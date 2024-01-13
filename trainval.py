import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np
import time
import argparse
import jstyleson as json
from dataclasses import field, dataclass
from omegaconf import OmegaConf as OC
from typing import Union, List

from src.models import Diffusion, Classifier, NoiseClassifier
from src.fid import fid_score
from src.datasets import get_dataset
from exp_configs.exp_utils import unflatten
# from src.datasets import DuplicateDatasetMTimes
from src.utils import (
    load_json_from_file,
    get_checkpoint,
    DuplicateDatasetMTimes,
    save_checkpoint
)

from src import setup_logger
logger = setup_logger.get_logger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('--datadir', type=str, default="")
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument(
        "--override_cfg",
        action="store_true",
        help="If this is set, then if there already exists a config.json "
        + "in the directory defined by savedir, load that instead of args.cfg. "
        + "This should be set so that SLURM does the right thing if the job is restarted.",
    )
    args = parser.parse_args()
    return args

# fmt: off
@dataclass
class Arguments:

    # Full path to the pretrained oracle checkpoint file. This is
    # the validation oracle, the thing which was trained on the
    # validation set.
    pretrained_oracle: str
    # Either gan, vae or diffusion
    model: str                          = "gan" 
    # TODO what is this for wrt to training diffuse
    oracle: str                         = "ResNet-v0"

    # -------------------
    # ** Dataset stuff **
    # -------------------
    dataset: str                        = "TFBind8"
    dataset_M: int                      = 0
    gain: float                         = 1.0
    gain_y: float                       = 1.0

    # -------------------
    # ** Network stuff **
    # -------------------
    # only applies to diffusion
    diffusion_kwargs: dict              = field(default_factory=lambda: {})
    
    # NOTE: only applies to GANs! No longer used for this codebase.
    disc_kwargs: dict                   = field(default_factory=lambda: {})
    gen_kwargs: dict                    = field(default_factory=lambda: {})

    # --------------------
    # ** Training stuff **
    # --------------------
    epochs: int                         = 5000
    batch_size: int                     = 512
    N_linspace: int = 100
    optim_kwargs: dict                  = field(
        default_factory=lambda: {'lr': 2e-4, 'betas': (0.0, 0.9), 'weight_decay': 0.}
    )
    # Save checkpoint every this many epochs. If None, the only checkpoint
    # saving will be at the end of each epoch (overwriting the main chkpt
    # file), or from the validation metric specific checkpoints which get
    # saved every time a new value for the validation metric is found.
    # (These are called `early stopping` checkpoints.)
    save_every: Union[int, None]        = None
    # Log metrics every this many seconds. Default is 30 sec.
    log_every: int                      = 30
    # NOTE: only for GANs
    update_g_every: Union[int, None]    = None
    use_ema: bool                       = True
    ema_rate: float                     = 0.9999

    # ----------------------
    # ** Validation stuff **
    # ----------------------
    fid_kwargs: dict                    = field(
        default_factory=lambda: {'all_features': True}
    )
    valid_metrics: List[str]            = field(
        default_factory=lambda: ["valid_fid_ema", "valid_agg_denorm_ema"]
    )
    eval_every: int                     = 10
    # If set to True, eval ground truth oracle in score_on_dataset.
    # One may want to disable this in the event that evaluating the
    # ground truth oracle during train/val takes too much time.
    eval_gt: bool                       = False
    # Only monitor validation metrics after this many epochs. This was
    # created because I found that the DC metric would give insanely 
    # good values at the start of training for no good reason.
    eval_after: int                     = 10
    eval_batch_size: Union[int, None]   = None

    # NOTE: only for GANs
    gamma: float = 1.0
    # NOTE: only for VAE
    beta: float = 1.0
# fmt: on

class FidWrapper(nn.Module):
    """Just a wrapper that conforms to the same
    interface as the Inception model used to
    compute FID.
    """

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return [self.f(x).unsqueeze(-1).unsqueeze(-1)]


def extra_validate_args(dd):
    # Load model and get checkpoint
    #if dd["gen_kwargs_str"] is not None:
    #    logger.info("gen_kwargs_str is set, overriding gen_kwargs...")
    #    dd["gen_kwargs"] = eval(dd["gen_kwargs_str"])
    #del dd["gen_kwargs_str"]
    if dd["model"] == "vae":
        if dd["disc_kwargs"] is not None:
            logger.warning("disc_kwargs is only defined for when model==gan")
        if dd["update_g_every"] is not None:
            logger.warning("update_g_every is only defined for when model==gan")
        if dd["gamma"] is not None:
            logger.warning("gamma is only defined for when model==gan")
    if dd["eval_batch_size"] is None:
        dd["eval_batch_size"] = dd["batch_size"]


"""
def trainval(exp_dict, savedir, args):
    validate_and_insert_defaults(exp_dict, DEFAULTS)
    extra_validate_args(exp_dict)

    # When haven saves exp_dict.json, it does not consider keys (default keys)
    # inserted after the experiment launches. So save a new exp dict
    # that overrides the one produced by Haven.
    with open("{}/exp_dict.json".format(savedir), "w") as f:
        f.write(json.dumps(exp_dict))
    _trainval(rank=0, exp_dict=exp_dict, savedir=savedir)
"""

@torch.no_grad()
def compute_fid_reference_stats(dataset, classifier, return_features=False, **fid_kwargs):
    tgt_X_cuda = dataset.X.to(classifier.rank)
    tgt_X_features = classifier.features(tgt_X_cuda, **fid_kwargs).cpu().numpy()
    logger.debug("tgt_X_features: {}".format(tgt_X_features.shape))
    tgt_mu, tgt_sigma = fid_score.calculate_activation_statistics(tgt_X_features)
    if return_features:
        return tgt_X_features, (tgt_mu, tgt_sigma)
    return tgt_mu, tgt_sigma


def _init_from_args(exp_dict, skip_model=False):
    """Return train/valid/test datasets and the model.
    This is a convenience class so that it can be called from
    other methods like eval.py.

    Args:
      rank: gpu rank
      exp_dict: the experiment dictionary
      skip_model: if true, do not return the model. Simply return
        Bprop, which is the 'dumb' model.
    """

    gain_y = exp_dict.get("gain_y", None)
    
    # Load dataset
    train_dataset = get_dataset(
        task_name=exp_dict["dataset"], oracle_name=exp_dict["oracle"],
        split="train",
        gain=exp_dict["gain"], gain_y=gain_y
    )

    valid_dataset = get_dataset(
        task_name=exp_dict["dataset"], oracle_name=exp_dict["oracle"],
        split="valid",
        gain=exp_dict["gain"], gain_y=gain_y

    )
    test_dataset = get_dataset(
        task_name=exp_dict["dataset"], oracle_name=exp_dict["oracle"],
        split="test",
        gain=exp_dict["gain"], gain_y=gain_y
    )

    name2model = {
        #"vae": VAE, "gan": GAN, 
        "diffusion": Diffusion,
    }
    
    model_class = name2model[exp_dict["model"]]
    base_kwargs = dict(
        n_out=train_dataset.n_in,
        optim_kwargs=exp_dict["optim_kwargs"],
        use_ema=exp_dict["use_ema"],
        ema_rate=exp_dict["ema_rate"],
    )

    if skip_model:
        #model = Bprop(
        #    classifier = None,
        #    **base_kwargs
        #)
        raise NotImplementedError
    else:   
        if model_class == "GAN":
            """
            model = model_class(
                gen_kwargs=exp_dict["gen_kwargs"],
                disc_kwargs=exp_dict["disc_kwargs"],            # TODO gen_kwargs
                update_g_every=exp_dict["update_g_every"],      # TODO gan_kwargs
                gamma=exp_dict["gamma"],                        # TODO gan_kwargs
                **base_kwargs
            )
            """
            raise NotImplementedError("To be done later")
        elif model_class == "VAE":
            """
            model = model_class(
                gen_kwargs=exp_dict["gen_kwargs"],
                beta=exp_dict["beta"],          # TODO this needs to be a vae_kwarg
                **base_kwargs
            )
            """
            raise NotImplementedError("To be done later")
        elif model_class == Diffusion:
            diffusion_kwargs = exp_dict["diffusion_kwargs"]

            cg_oracle = None
            if 'pretrained_cg' in diffusion_kwargs:
                logger.debug("Found `pretrained_cg` in `diffusion_kwargs` so " + \
                    "assume classifier guidance instead...")
                if 'w_cg' not in diffusion_kwargs:
                    raise ValueError("If `pretrained_cg` is defined then `w_cg` " + \
                        "classifier guidance scale) also needs to be defined")
                # Load the classifier guidance oracle
                cg_exp_dict = load_json_from_file(
                    "{}/exp_dict.json".format(os.path.dirname(diffusion_kwargs["pretrained_cg"]))
                )
                logger.debug("Loading classifier guidance oracle: {}, use w_cg={}".format(
                    diffusion_kwargs["pretrained_cg"],
                    diffusion_kwargs["w_cg"],
                ))
                cg_oracle = NoiseClassifier(
                    n_in=train_dataset.n_in,
                    model_kwargs=cg_exp_dict["model_kwargs"],
                    optim_kwargs=cg_exp_dict["optim_kwargs"],
                    **cg_exp_dict["classifier_kwargs"]
                )
                cg_oracle.set_state_dict(torch.load(diffusion_kwargs["pretrained_cg"]))
                if diffusion_kwargs["tau"] != 1.0:
                    raise ValueError("tau should be == 1 when using classifier guidance " + \
                        "(we don't want to use labels at all)")
            
            model = model_class(
                n_classes=diffusion_kwargs["n_classes"],
                tau=diffusion_kwargs["tau"],
                w=diffusion_kwargs.get("w", 0.),
                arch=diffusion_kwargs.get("arch", "mlp"),
                gen_kwargs=exp_dict["gen_kwargs"],
                # This is the training oracle to be used with classifier guidance
                # if it is enabled. Not to be confused with the validation oracle,
                # which is `pretrained_oracle`.
                oracle=cg_oracle,
                **base_kwargs
            )
        else:
            raise ValueError("Unknown model class: {}".format(model_class))

    return (train_dataset, valid_dataset, test_dataset), model

def run(exp_dict, savedir):

    exp_dict = vars(exp_dict)
    extra_validate_args(exp_dict)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    rank = torch.cuda.current_device()

    datasets, model = _init_from_args(exp_dict)
    train_dataset, valid_dataset, test_dataset = datasets

    # ---------------------------
    # Load our validation oracle.
    # ---------------------------
    
    val_oracle_exp_dict = load_json_from_file(
        "{}/exp_dict.json".format(os.path.dirname(exp_dict["pretrained_oracle"]))
    )

    # HACK: backwards compatibility with older trained
    # models.
    if "n_in" in val_oracle_exp_dict["model_kwargs"]:
        del val_oracle_exp_dict["model_kwargs"]['n_in']
    
    val_oracle = Classifier(
        n_in=train_dataset.n_in,
        model_kwargs=val_oracle_exp_dict["model_kwargs"],
        optim_kwargs=val_oracle_exp_dict["optim_kwargs"],
    )
    val_oracle.set_state_dict(torch.load(exp_dict["pretrained_oracle"]))

    # Safety here since older experiments didn't have these
    # args set.
    cls_gain = val_oracle_exp_dict.get("gain", 1.0)
    cls_gain_y = val_oracle_exp_dict.get("gain_y", 1.0)

    # ----------------------------
    # Load diffusion model weights
    # ----------------------------

    # Explicitly set what gpu to put the weights on.
    # If map_location is not set, each rank (gpu) will
    # load these onto presumably gpu0, causing an OOM
    # if we run this code under a resuming script.
    chk_dict = get_checkpoint(
        savedir,
        return_model_state_dict=True,
        map_location=lambda storage, loc: storage.cuda(rank),
    )
    if len(chk_dict["model_state_dict"]):
        model.set_state_dict(chk_dict["model_state_dict"], strict=True)

    #if val_oracle_exp_dict["postprocess"] != exp_dict["postprocess"]:
    ##    raise ValueError(
    #        "Postprocess flags are inconsistent. This would cause bad "
    #        + "FID estimates since real / fake x statistics would be inconsistent"
    #    )
    
    if cls_gain != exp_dict["gain"]:
        raise ValueError("Classifier experiment has a different gain (cls={} vs {})".\
            format(cls_gain, exp_dict["gain"]))
    elif cls_gain_y != exp_dict["gain_y"]:
        raise ValueError("Classifier experiment has a different gain_y (cls={} vs {})".\
            format(cls_gain_y, exp_dict["gain_y"]))

    dataset_M = exp_dict["dataset_M"]
    if dataset_M > 0:
        train_dataset_ = DuplicateDatasetMTimes(train_dataset, M=dataset_M)
    else:
        train_dataset_ = train_dataset
    train_loader = DataLoader(
        train_dataset_, shuffle=True, batch_size=exp_dict["batch_size"],
        #pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, shuffle=True, batch_size=exp_dict["batch_size"],
        #pin_memory=True
    )

    # ---------------------------------------------
    # Pre-compute FID stats for valid and test sets
    # ---------------------------------------------

    fid_stats = {}
    for tgt_dataset, tgt_name in zip(
        [train_dataset, valid_dataset, test_dataset], ["train", "valid", "test"]
    ):
        logger.info("Computing FID reference stats for: {}".format(tgt_name))
        fid_stats[tgt_name] = compute_fid_reference_stats(
            tgt_dataset, val_oracle, True, **exp_dict["fid_kwargs"]
        )

    # ------------------
    # Run Train-Val loop
    # ------------------
    
    max_epochs = exp_dict["epochs"]
    save_every = exp_dict["save_every"]
    eval_every = exp_dict["eval_every"]
    eval_after = exp_dict["eval_after"]

    valid_metrics = exp_dict["valid_metrics"]
    record_metrics = {k: np.inf for k in valid_metrics}
    logger.info("metrics for checkpoint saving: {}".format(valid_metrics))
    if len(chk_dict["score_list"]) == 0:
        for key in record_metrics.keys():
            record_metrics[key] = np.inf
    else:
        # If we're resuming from a pre-trained checkpoint, find what the
        # minimum value is meant to be for each of the metrics in
        # `valid_metrics`.
        for key in record_metrics.keys():
            this_scores = [
                score[key] for score in chk_dict["score_list"] if key in score
            ]
            if len(this_scores) == 0:
                record_metrics[key] = np.inf
            else:
                record_metrics[key] = min(this_scores)
                logger.debug("record_metrics[{}] = {}".format(key, min(this_scores)))


    logger.info("Starting epoch: {}".format(chk_dict["epoch"]))
    for epoch in range(chk_dict["epoch"], max_epochs):

        t0 = time.time()

        #score_dict.update(
        model.score_on_dataset(
            dataset=valid_dataset, 
            classifier=val_oracle, 
            fid_stats=fid_stats["train"], 
            fid_kwargs=exp_dict["fid_kwargs"],
            eval_gt=exp_dict["eval_gt"],
            prefix="valid",
            batch_size=exp_dict["eval_batch_size"]
        )
        #)

        if rank == 0:
            score_dict = {}
            score_dict["epoch"] = epoch

        train_dict_ = model.train_on_loader(
            train_loader,
            epoch=epoch,
            savedir=savedir,
            log_every=exp_dict["log_every"],
            # pbar=world_size <= 1,
            pbar=False
        )
        train_dict = {("train_" + key): val for key, val in train_dict_.items()}

        valid_dict_ = model.eval_on_loader(
            valid_loader,
            epoch=epoch,
            savedir=savedir,
            log_every=exp_dict["log_every"],
            # pbar=world_size <= 1,
            pbar=False
        )
        valid_dict = {("valid_" + key): val for key, val in valid_dict_.items()}

        score_dict.update(train_dict)
        score_dict.update(valid_dict)
        score_dict["time"] = time.time() - t0

        logger.info(
            {k:v for k,v in score_dict.items() if k.endswith("_mean") or k=="epoch"}
        )

        if eval_every > 0 and epoch % eval_every == 0 and epoch > eval_after:

            for this_dataset, this_split in [
                (train_dataset, "train"), # training set for debugging purposes
                (valid_dataset, "valid"), # validation metrics computed on valid set
                (test_dataset, "test")    # this split is not needed
            ]:
            
                score_dict.update(
                    model.score_on_dataset(
                        dataset=this_dataset, 
                        classifier=val_oracle, # valid metrics use validation oracle
                        fid_stats=fid_stats[this_split], 
                        fid_kwargs=exp_dict["fid_kwargs"],
                        eval_gt=exp_dict["eval_gt"],
                        prefix=this_split,
                        batch_size=exp_dict["eval_batch_size"]
                    )
                )

            chk_dict["score_list"] += [score_dict]

            for metric in record_metrics.keys():
                if score_dict[metric] < record_metrics[metric]:
                    logger.info(
                        "New best metric {}: from {:.4f} to {:.4f}".format(
                            metric, record_metrics[metric], score_dict[metric]
                        )
                    )
                    # save the new best metric
                    record_metrics[metric] = score_dict[metric]
                    save_checkpoint(
                        savedir,
                        fname_suffix="." + metric,
                        score_list=chk_dict["score_list"],
                        model_state_dict=model.get_state_dict()
                    )
        else:
            chk_dict["score_list"] += [score_dict]

        # Save checkpoint
        save_checkpoint(
            savedir,
            score_list=chk_dict["score_list"],
            model_state_dict=model.get_state_dict(),
        )

        # If `save_every` is defined, save every
        # this many epochs.
        if save_every is not None:
            if epoch > 0 and epoch % save_every == 0:
                save_checkpoint(
                    savedir,
                    fname_suffix="." + str(epoch),
                    score_list=chk_dict["score_list"],
                    model_state_dict=model.get_state_dict(),
                )

    print("Experiment completed")

if __name__ == "__main__":
    args = parse_args()

    saved_cfg_file = os.path.join(args.savedir, "config.json")
    if os.path.exists(saved_cfg_file) and not args.override_cfg:
        cfg_file = json.loads(open(saved_cfg_file, "r").read())
        logger.debug("Found config in exp dir, loading instead...")
    else:
        cfg_file = json.loads(open(args.cfg, "r").read())

    cfg_file = unflatten(cfg_file)

    # structured() allows type checking
    conf = OC.structured(Arguments(**cfg_file))

    # Since type checking is already done, convert
    # it back ito a (dot-accessible) dictionary.
    # (OC.to_object() returns back an Arguments object)
    run(OC.to_object(conf), args.savedir)
