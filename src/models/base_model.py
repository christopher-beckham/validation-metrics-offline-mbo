from typing import Callable, Dict, Tuple, Union
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
import math

from ..utils import suppress_stdout_stderr
from ..fid import fid_score
from .classifier import Classifier
import prdc

from ..setup_logger import get_logger
logger = get_logger(__name__)

# If this is enabled, only one iteration is done per epoch.
# This is for testing purposes only.
DRY_RUN = True if "DRY_RUN" in os.environ else False

class BaseModel:

    def __init__(self, 
                 n_out: int,
                 optim_kwargs: Dict,
                 use_ema: bool,
                 ema_rate: float,
                 rank: int = 0, 
                 verbose: bool = True):
        self.n_out = n_out
        self.optim_kwargs = optim_kwargs
        self.use_ema = use_ema
        self.ema_rate = ema_rate
        self.rank = rank
        self.verbose = verbose

        self.iteration = 0
    
    def _eval_loss_dict(self, loss_dict, return_str=True):
        loss = 0.0
        loss_str = []
        for key, val in loss_dict.items():
            if len(val) != 2:
                raise Exception("val must be a tuple of (coef, loss)")
            if val[0] != 0:
                # Only add the loss if the coef is != 0
                loss += val[0] * val[1]
            loss_str.append("{} * {}".format(val[0], key))
        if return_str:
            return loss, (" + ".join(loss_str))
        return loss

    @torch.no_grad()
    def sample(self, y, z=None, use_ema=False, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def sample_batched(self, batch_size, y, *args, **kwargs):
        n_examples = y.size(0)
        n_batches = int(math.ceil(n_examples / batch_size))
        buf = []
        for b in range(n_batches):
            #print(b)
            buf.append(
                self.sample(
                    y[(b*batch_size):(b+1)*batch_size], *args, **kwargs
                )
            )
        buf = torch.cat(buf, dim=0)        
        assert buf.size(0) == y.size(0)
        return buf

    @torch.no_grad()
    def sample_z(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def ema_src_network(self) -> nn.Module:
        """EMA source network""" 
        raise NotImplementedError()

    @property
    def ema_tgt_network(self) -> nn.Module:
        """EMA target network"""
        raise NotImplementedError()

    def update_ema(self):
        src_dict = self.ema_src_network.state_dict()
        tgt_dict = self.ema_tgt_network.state_dict()
        for key in src_dict.keys():
            # ema_param*ema_rate + original_param*(1-ema_rate)
            tgt_dict[key].data.copy_(
                tgt_dict[key].data * self.ema_rate + \
                    (1 - self.ema_rate) * src_dict[key].data
            ) 

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def _run_on_batch(self, 
                      batch: Tuple[torch.Tensor, ...], 
                      train: bool = True, 
                      savedir: str = None, 
                      **kwargs):
        raise NotImplementedError()

    def _run_on_loader(
        self, 
        run_on_batch_fn, 
        loader, 
        train, 
        log_every=None, 
        pbar=True, 
        return_buf=False,
        **kwargs
    ):
        buf = {}
        desc_str = "train" if train else "validate"
        if not hasattr(self, '_curr_time'):
            self._curr_time = time.time()
        for b, batch in enumerate(tqdm(loader, desc=desc_str, disable=not pbar)):

            verbose = False
            if log_every is not None and time.time() -  self._curr_time > log_every:
                verbose = True
                self._curr_time = time.time()
                logger.debug("{} secs have elapsed, set verbose in run_on_batch_fn".\
                    format(log_every))
            train_dict = run_on_batch_fn(
                batch, train=train, iter=b, verbose=verbose, **kwargs
            )
            for key in train_dict:
                if key not in buf:
                    buf[key] = []
                buf[key].append(train_dict[key])

            if DRY_RUN:
                logger.debug("env DRY_RUN is set, break early...")
                break

            self.iteration += 1

        pd_metrics = {}
        for key, val in buf.items():
            if type(val[0]) == torch.Tensor:
                pd_metrics[key] = torch.stack(val).detach().cpu().numpy()
            else:
                pd_metrics[key] = np.asarray(val)

        all_metrics = {k + "_mean": v.mean() for k, v in pd_metrics.items()}
        all_metrics.update({k + "_min": v.min() for k, v in pd_metrics.items()})
        all_metrics.update({k + "_max": v.max() for k, v in pd_metrics.items()})

        if return_buf:
            return pd_metrics
        else:
            return all_metrics

    def train_on_loader(self, 
                        loader: DataLoader, 
                        log_every: bool = None, 
                        **kwargs):
        return self._run_on_loader(
            self._run_on_batch, loader, train=True, log_every=log_every, **kwargs
        )

    def eval_on_loader(self, 
                       loader: DataLoader,
                       log_every: bool = None, 
                       **kwargs):
        return self._run_on_loader(
            self._run_on_batch, loader, train=False, log_every=log_every, **kwargs
        )

    @torch.no_grad()
    def score_on_dataset(self, 
                         dataset: Dataset,
                         classifier: Classifier,
                         fid_stats: Tuple[ torch.Tensor, Tuple[float, float] ], 
                         fid_kwargs: Union[ Dict, None] = None,
                         sample_fn: Callable = None,
                         batch_size: int = 512,
                         eval_gt: bool = True,
                         prefix: str = "", 
                         verbose: bool = True,
                         **kwargs):
        """Score the model on a dataset, which involves computing all of the 
        metrics of interest.

        Note that this differs from `eval_on_loader` (apart from the loader 
        aspect) in that the method only evaluates losses/metrics that are
        evaluated during training (`train_on_loader`). This method evaluates
        the following metrics: FID, precision, density, agreement, baseline
        agreement, ...

        Args:
            dataset: dataset
            classifier: features will be computed from this
            batch_size: batch size
            fid_stats: tuple where the first element are the features
              computed from the samples, and the second element is in
              turn a tuple denoting the (mean, sd).
            sample_fn: if this is not None, then a custom sample method will
              be used in place of the sample() that exists in the class
            prefix: metrics will be prefixed with this
        """

        cls = classifier

        self.eval()

        #if self.use_ema:
        #    eval_modes = [("", False), ("_ema", True)]
        #else:
        eval_modes = [("", False)]

        logger.info("eval modes: {}".format(eval_modes))

        score_dict = dict()

        if fid_kwargs is None:
            fid_kwargs = {}

        t0 = time.time()
        
        for (eval_postfix, eval_mode) in eval_modes:

            logger.info("scoring: {} - {}".format(eval_postfix, eval_mode))
            
            tgt_all_y = dataset.y.to(self.rank)
            if sample_fn is None:
                gen_samples = self.sample_batched(batch_size, tgt_all_y, use_ema=eval_mode)
            else:
                gen_samples = sample_fn(tgt_all_y)

            if not hasattr(cls, "features"):
                raise Exception("cls must have a method called features(x)")
            gen_samples_feats = cls.features(gen_samples, **fid_kwargs).cpu().numpy()
            #logger.debug("FID features shape: {}".format(gen_samples_feats.shape))

            #################################################
            # Find argmax_y fvalid(x), sample 10 candidates #
            # then compute mean score                       #
            #################################################

            if verbose:
                logger.info("  metric: mean reward")

            N_REPEATS = 10
            y_linspace = torch.linspace(tgt_all_y.min(), tgt_all_y.max(), tgt_all_y.size(0)).\
                view(-1, 1).to(self.rank)
            gen_samples_linspace = self.sample_batched(batch_size, y_linspace, use_ema=eval_mode)
            y_argmax = cls.predict(gen_samples_linspace).argmax(0)
            samples_y_argmax = self.sample_batched(batch_size,
                                                   tgt_all_y[y_argmax].repeat(N_REPEATS,1),
                                                   use_ema=eval_mode)
            samples_ymax_scored = cls.predict(samples_y_argmax)
            samples_ymax_scored_denorm = dataset.denorm_y(samples_ymax_scored)

            score_dict["{}_mean_score{}".format(prefix, eval_postfix)] = \
                -samples_ymax_scored.mean().item()
            score_dict["{}_mean_score_denorm{}".format(prefix, eval_postfix)] = \
                -samples_ymax_scored_denorm.mean().item()

            ###############
            # Compute FID #
            ###############

            if verbose:
                logger.info("  metric: FID")
            
            mu_tgt, sigma_tgt = fid_score.calculate_activation_statistics(
                gen_samples_feats
            )
            this_fid = fid_score.calculate_frechet_distance(
                fid_stats[1][0],
                fid_stats[1][1],
                mu_tgt,
                sigma_tgt,
            )
            score_dict["{}_fid{}".format(prefix, eval_postfix)] = this_fid
            if verbose:
                logger.debug(
                    "  {}_fid{}: {:.4f}".format(prefix, eval_postfix, this_fid)
                )

            ##############################################
            # Compute precision / recall / realism score #
            ##############################################

            if verbose:
                logger.info("  metric: DC (density + coverage)")

            with suppress_stdout_stderr():
                prdc_stats = prdc.compute_prdc(
                    real_features=fid_stats[0],
                    fake_features=gen_samples_feats,
                    nearest_k=3
                )
            prdc_stats["pr"] = (
                prdc_stats["precision"] + prdc_stats["recall"]
            )
            prdc_stats["dc"] = (
                prdc_stats["density"] + prdc_stats["coverage"]
            )
            
            #prdc_stats["realism"] = prdc_realism
            for prdc_key in prdc_stats.keys():
                # Make sure these stats are -ve because we want to
                # _max_ these metrics.
                score_dict[
                    "{}_{}{}".format(prefix, prdc_key, eval_postfix)
                ] = -prdc_stats[prdc_key] # <-- NOTE the -ve
                if verbose:
                    logger.debug(
                        "  {}_{}{}: {:.4f}".format(
                            prefix,
                            prdc_key,
                            eval_postfix,
                            prdc_stats[prdc_key],
                        )
                    )

            ##############################################
            # Compute agreement wrt to validation oracle #
            ##############################################

            if verbose:
                logger.info("  metric: agreement")

            if dataset.is_discrete:
                n_nonbinary = ((gen_samples > 0) & (gen_samples < 1)).float().mean()
                if n_nonbinary > 0:
                    # This is not one-hot
                    logger.warning("gen_samples is not a one-hot-encoded vector yet it will be passed " + \
                        "into the validation oracle")
            
            with torch.no_grad():
                pred_y = cls.predict(gen_samples).cpu().numpy().flatten()
                
            tgt_all_y = tgt_all_y.cpu().flatten().numpy()
            tgt_agg_norm = np.mean((pred_y - tgt_all_y) ** 2)
            tgt_agg_denorm = np.mean(
                (
                    dataset.denorm_y(tgt_all_y)
                    - dataset.denorm_y(pred_y)
                )
                ** 2
            )
            
            # Only compute this once, we want the baseline agreement.
            tgt_agg_baseline_denorm = np.mean(
                (
                    dataset.denorm_y(tgt_all_y)
                    - dataset.denorm_y(tgt_all_y.mean())
                )
                ** 2
            )
            score_dict[
                "{}_agg_baseline_denorm{}".format(prefix, eval_postfix)
            ] = tgt_agg_baseline_denorm
            if verbose:
                logger.debug(
                    "  {}_agg_baseline_denorm{}: {:.4f}".format(
                        prefix, eval_postfix, tgt_agg_baseline_denorm
                    )
                )

            score_dict[
                "{}_agg{}".format(prefix, eval_postfix)
            ] = tgt_agg_norm
            if verbose:
                logger.debug(
                    "  {}_agg{}: {:.4f}".format(
                        prefix, eval_postfix, tgt_agg_norm
                    )
                )

            score_dict[
                "{}_agg_denorm{}".format(prefix, eval_postfix)
            ] = tgt_agg_denorm
            if verbose:
                logger.debug(
                    "  {}_agg_denorm{}: {:.4f}".format(
                        prefix, eval_postfix, tgt_agg_denorm
                    )
                )

            ########################################
            # Compute agreement wrt to test oracle #
            ########################################

            if eval_gt:
                with suppress_stdout_stderr():
                    gt_pred = dataset.predict(gen_samples).flatten()
                tgt_all_y_denorm = dataset.denorm_y(tgt_all_y).flatten()
                score_dict[
                    "{}_gt-agg_denorm{}".format(prefix, eval_postfix)
                ] = np.mean((gt_pred-tgt_all_y_denorm)**2)

        time_taken = time.time() - t0
        logger.debug("Time taken for scoring: {:.4f} sec".format(time_taken))
        score_dict["{}_eval_time".format(prefix)] = time_taken

        return score_dict

    def get_state_dict(self):
        raise NotImplementedError()

    def set_state_dict(self, state_dict, load_opt=True, strict=True):
        raise NotImplementedError()
