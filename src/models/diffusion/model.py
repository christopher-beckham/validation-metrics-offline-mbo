import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW as Adam
from torch.autograd import grad

# from torch.cuda.amp import autocast, GradScaler
#from multiprocessing.sharedctypes import Value
#import os
from typing import Callable, Dict, Union
from collections import OrderedDict

from ..classifier import Classifier
from ... import utils as ut
from ...setup_logger import get_logger

logger = get_logger(__name__)

from ..base_model import BaseModel
#from .networks import EncoderDecoder
from .networks_unet_mlp import UNet_MLP
from .networks_unet_conv1d import UNet_Conv1d

# from .nerf_helpers import


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02, exponent=2):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** exponent


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class Diffusion(BaseModel):
    DEFAULT_ARGS = {}

    def _validate_args(self, dd):
        pass

    def setup(self, n_timesteps):
        self.betas = linear_beta_schedule(n_timesteps)
        self.n_timesteps = n_timesteps

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def __init__(
        self,
        n_classes: int,
        tau: float,
        w: float,
        gen_kwargs: Dict,
        arch: str = "mlp",
        oracle: Union[Classifier, None] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ARCH_MODES = ["mlp", "conv1d"]

        # Set up internal state for DDPM
        self.setup(n_timesteps=n_classes)

        if gen_kwargs is None:
            gen_kwargs = {}
        if arch == "mlp":
            model_class = UNet_MLP
        elif arch == "conv1d":
            model_class = UNet_Conv1d
        else:
            raise ValueError("`arch` must be one of: {}".format(self.ARCH_MODES))

        self.model = model_class(self.n_out, n_classes=n_classes, **gen_kwargs)
        self.model.to(self.rank)

        if self.use_ema:
            self.model_ema = model_class(self.n_out, n_classes=n_classes, **gen_kwargs)
            self.model_ema.to(self.rank)
        else:
            self.model_ema = None

        if self.verbose and self.rank == 0:
            logger.info("gen: {}".format(self.model))
            logger.info("# gen params: {}".format(ut.count_params(self.model)))

        # beta1 = optim_kwargs.pop("beta1")
        # beta2 = optim_kwargs.pop("beta2")
        
        if 'beta1' in self.optim_kwargs and 'beta2' in self.optim_kwargs:
            # HACK: older experiments used a separate beta1 and beta2 whereas
            # it should just be betas, so correct that here if that needs to
            # be done.
            logger.warning("beta1 and beta2 found in optim_kwargs, consolidating into betas=(beta1,beta)")
            beta1 = self.optim_kwargs['beta1']
            beta2 = self.optim_kwargs['beta2']
            self.optim_kwargs['betas'] = (beta1, beta2)
            del self.optim_kwargs['beta1']
            del self.optim_kwargs['beta2']
        
        self.opt_g = Adam(
            self.model.parameters(),
            **self.optim_kwargs
            # betas=(beta1, beta2),
            # **optim_kwargs
        )
        logger.info(self.opt_g)

        self.oracle = oracle

        self.tau = tau
        self.w = w

        self.ZERO_CONSTANT = -100

    def set_tau(self, tau):
        self.tau = tau

    def sample_z(self, bs):
        return torch.randn((bs, self.n_out))

    @torch.no_grad()
    def q_sample(self, x0, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def _p_sample_cfg(self, x, y, t, t_index, w):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        # Use Bernoulli expectation: tau*pred_eps_uncond + (1-tau)*pred_eps_cond
        pred_eps_cond = self.model(x, y, t)
        pred_eps_uncond = self.model(x, y * 0 + self.ZERO_CONSTANT, t)

        pred_eps = (w + 1) * pred_eps_cond - w * pred_eps_uncond

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pred_eps / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = self.sample_z(x.size(0)).to(x.device)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def _p_sample_cg(self, x, y, t, t_index, w):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        pred_eps_uncond = self.model(x, y * 0 + self.ZERO_CONSTANT, t)

        # log p(y|x;t)
        with torch.enable_grad():
            x_detached = x.detach().requires_grad_(True)
            log_density_cls = self.oracle.log_prob(y, x_detached, t=t)
            cls_grad = grad(log_density_cls.sum(), x_detached)[0]
            assert cls_grad.shape == x_detached.shape

        pred_eps = pred_eps_uncond - w * sqrt_one_minus_alphas_cumprod_t * cls_grad

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pred_eps / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = self.sample_z(x.size(0)).to(x.device)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample(self, y, **kwargs):
        w = kwargs.pop("w", None)
        if w is None:
            w = self.w
        bs = y.size(0)
        img = self.sample_z(bs).to(y.device)

        if self.oracle is None:
            p_sample_fn = self._p_sample_cfg
        else:
            p_sample_fn = self._p_sample_cg

        for i in reversed(range(1, self.n_timesteps)):
            # print(">>,", i)
            img = p_sample_fn(
                img, y, torch.full((bs,), i, device=self.rank, dtype=torch.long), i, w=w
            )
        return img

    @property
    def ema_src_network(self):
        return self.model

    @property
    def ema_tgt_network(self):
        return self.model_ema

    @torch.no_grad()
    def sample(self, y, z=None, use_ema=False, **kwargs):
        if z is not None:
            raise ValueError("A pre-defined z is not supported for this class")
        if use_ema:
            if not self.use_ema:
                raise Exception("use_ema was set but this model has no EMA weights")
            model = self.model_ema
        else:
            model = self.model
        # Do not delegate to model.sample here,
        # we should really do that in this class.
        return self.p_sample(y, **kwargs)

        # return model.sample(y, z, **kwargs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def _run_on_batch(self, batch, train=True, verbose=False, epoch=None, iter_=None, **kwargs):
        if train:
            self.train()
        else:
            self.eval()

        if train:
            self.opt_g.zero_grad()

        metrics = {}
        g_loss_dict = OrderedDict({})

        x_batch, y_batch_ = batch
        x_batch = x_batch.to(self.rank)
        y_batch_ = y_batch_.to(self.rank)

        bs = x_batch.size(0)
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.n_timesteps, (bs,), device=self.rank).long()
        # loss = p_losses(model, batch, t, loss_type="huber")

        noise = self.sample_z(bs).to(x_batch.device)

        # Classifier-free guidance, with probability tau drop out
        # some of the y's, i.e. set them to ZERO_CONSTANT.
        probs = (
            torch.zeros(
                bs,
            )
            .uniform_(0, 1)
            .view(-1, 1)
            .to(self.rank)
        )
        
        if self.tau == 0:
            y_batch__masked = y_batch_
        else:
            masks = probs <= self.tau
            y_batch__masked = y_batch_.clone()
            y_batch__masked[masks] = self.ZERO_CONSTANT  # don't use zero

        x_t = self.q_sample(x0=x_batch, t=t, noise=noise)
        predicted_noise = self.model(x_t, y_batch__masked, t)
        loss = F.smooth_l1_loss(noise, predicted_noise)

        with torch.no_grad():
            predicted_noise_cond = self.model(x_t, y_batch_, t)
            pred_diff = (predicted_noise_cond - predicted_noise) ** 2
            metrics["pred_diff"] = pred_diff.mean()
            # This is the 'actual' loss since it's all conditional, but
            # we don't want to optimise this one since we're doing CFG.
            loss_cond = F.smooth_l1_loss(noise, predicted_noise_cond)
            metrics["loss_cond"] = loss_cond.mean()

        g_loss_dict["dsm_loss"] = (1.0, loss)
        g_total_loss, g_total_loss_str = self._eval_loss_dict(g_loss_dict)
        if verbose and epoch == 0 and iter_ == 0:
            logger.info(
                "{}: G is optimising this total loss: {}".format(
                    self.rank, g_total_loss_str
                )
            )
            logger.debug("x_batch.shape = {}".format(x_batch.shape))
            logger.debug(
                "x_batch min max = {}, {}".format(x_batch.min(), x_batch.max())
            )

        if train:
            g_total_loss.backward()
            self.opt_g.step()
            self.update_ema()

        if verbose:
            logger.info("g_loss_dict: {}".format(g_loss_dict))
            logger.info("g_metrics: {}".format(metrics))

        with torch.no_grad():
            metrics = {k: v.detach() for k, v in metrics.items()}
            metrics.update({k: v[1].detach() for k, v in g_loss_dict.items()})

        return metrics

    def score_on_dataset(
        self,
        dataset: Dataset,
        classifier: Classifier,
        fid_stats: Dict,
        sample_fn: Callable = None,
        prefix: str = "",
        **kwargs
    ):
        score_dict = super().score_on_dataset(
            dataset=dataset,
            classifier=classifier,
            sample_fn=sample_fn,
            fid_stats=fid_stats,
            prefix=prefix,
            **kwargs
        )
        batch_size = kwargs.get("batch_size", 1)
        full_loader = DataLoader(dataset, batch_size=batch_size)

        eval_stats = self.eval_on_loader(full_loader, return_buf=True)
        dsm_loss = eval_stats["dsm_loss"]

        score_dict["{}_dsm_loss".format(prefix)] = dsm_loss.mean()

        return score_dict

    def get_state_dict(self):
        state_dict = {
            "gen": self.model.state_dict(),
            "opt_g": self.opt_g.state_dict(),
        }
        if self.use_ema:
            state_dict["gen_ema"] = self.model_ema.state_dict()

        return state_dict

    def _load_state_dict_with_mismatch(self, current_model_dict, chkpt_model_dict):
        # https://github.com/pytorch/pytorch/issues/40859
        # strict won't let you load in a state dict with
        # mismatch param shapes, so we do this hack here.
        new_state_dict = {
            k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
            for k, v in zip(current_model_dict.keys(), chkpt_model_dict.values())
        }
        return new_state_dict

    def set_state_dict(self, state_dict, load_opt=True, strict=True):
        if strict:
            self.model.load_state_dict(state_dict["gen"], strict=strict)
            if self.use_ema:
                self.model_ema.load_state_dict(state_dict["gen_ema"], strict=strict)
        else:
            self.model.load_state_dict(
                self._load_state_dict_with_mismatch(
                    current_model_dict=self.model.state_dict(),
                    chkpt_model_dict=state_dict["gen"],
                )
            )
            if self.use_ema:
                self.model_ema.load_state_dict(
                    self._load_state_dict_with_mismatch(
                        current_model_dict=self.model_ema.state_dict(),
                        chkpt_model_dict=state_dict["gen_ema"],
                    )
                )
        if load_opt:
            self.opt_g.load_state_dict(state_dict["opt_g"])
