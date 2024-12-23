import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import time
import jstyleson as json
import argparse
from dataclasses import asdict, dataclass, field
from exp_utils import unflatten
from omegaconf import OmegaConf as OC
from src.models import Classifier, NoiseClassifier
from src.datasets import get_dataset
# from src.datasets import DuplicateDatasetMTimes
from src.utils import DuplicateDatasetMTimes, save_checkpoint
from typing import Union

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
    dataset: str                = "TFBind8"
    # Create a version of the dataset duplicated M times. This can
    # be useful if you want to mitigate the rate at which data loaders
    # are destroyed/re-created.
    dataset_M: int              = 0
    
    # If true, train on the entire dataset (still set aside X% for
    # internal validation). Set this when there is already a ground
    # truth oracle that comes with the dataset.
    mode: str                    = "train+valid"
    
    # X is normalised between [-1, 1] by default, `gain` instead will
    # make it so it is instead [-gain, +gain].
    gain: float                 = 1.0 
    # Same as above but for y.
    gain_y: Union[float, None]  = None 

    # Set if you want to model the conditional distn p(y|x_t), for
    # any noisy timestep t.
    use_noise: bool             = False

    # This can be anything here, since we won't be using a pre-trained
    # oracle (this script is for training our own version).
    oracle: str                 = "ResNet-v0"
    batch_size: int             = 512
    epochs: int                 = 5000
    model_kwargs: dict          = field(default_factory=lambda: {})
    
    # This is meant to be used if one chooses the ClassifierGuidance
    # class, which requires some extra args like the number of timesteps
    # and what noise schedule to use.
    classifier_kwargs: dict     = field(default_factory=lambda: {})

    optim_kwargs: dict          = field(default_factory=lambda: {
        'lr': 2e-4, 'betas': (0.0, 0.9), 'weight_decay': 0.0
    })

    # number of workers for data loader
    num_workers: int            = 2

    save_every: Union[int, None]= None
    eval_every: int             = 10
# fmt: on

"""
NONETYPE = type(None)
DEFAULTS = {
    "dataset": Argument("dataset", "TFBind8", [str]),
    "dataset_M": Argument("dataset_M", 0, [int]),

    "use_noise": Argument("use_noise", False, [bool]),

    # If true, train on the entire dataset (still set aside X% for
    # internal validation). Set this when there is already a ground
    # truth oracle that comes with the dataset.
    "test_oracle": Argument("test_oracle", False, [bool]),

    "gain": Argument("gain", 1.0, [float]),
    "gain_y": Argument("gain_y", None, [float, NONETYPE]),

    "postprocess": Argument("postprocess", False, [bool]),
    "oracle": Argument("oracle", "ResNet-v0", [str]),
    "batch_size": Argument("batch_size", 512, [int]),
    "epochs": Argument("epochs", 5000, [int]),
    "model_kwargs": Argument("model_kwargs", {}, [dict]),

    # This is meant to be used if one chooses the ClassifierGuidance
    # class, which requires some extra args like the number of timesteps
    # and what noise schedule to use.
    "classifier_kwargs": Argument("classifier_kwargs", {}, [dict]),
   
    "optim_kwargs": {
        "lr": Argument("lr", 2e-4, [float]),
        "beta1": Argument("beta1", 0.0, [float]),
        "beta2": Argument("beta2", 0.9, [float]),
        "weight_decay": Argument("weight_decay", 0.0, [float])
    },
    "save_every": Argument("save_every", None, [int, NONETYPE]),
    "eval_every": Argument("eval_every", 10, [int])
}
"""

# wtf?
# gym.error.DependencyNotInstalled: numpy.core.multiarray failed to import (auto-generated because you didn't call 'numpy.import_array()' after cimporting numpy; use '<void>numpy._import_array' to disable if you are certain you don't need it).. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)
# Doing the below import seems to fix it.
import numpy.core.multiarray


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


def run(exp_dict, savedir):
    exp_dict = vars(exp_dict)  # convert back to dict

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open("{}/exp_dict.json".format(savedir), "w") as f:
        f.write(json.dumps(exp_dict))

    # Load dataset
    train_dataset = get_dataset(
        task_name=exp_dict["dataset"],
        oracle_name=exp_dict["oracle"],
        split="train",
        gain=exp_dict["gain"],
        gain_y=exp_dict["gain_y"],
    )
    valid_dataset = get_dataset(
        task_name=exp_dict["dataset"],
        oracle_name=exp_dict["oracle"],
        split="valid",
        gain=exp_dict["gain"],
        gain_y=exp_dict["gain_y"],
    )
    n_features = train_dataset.n_in
    mode = exp_dict["mode"]
    logger.info("mode={}".format(mode))
    if mode == "train+valid+test":
        # You should only use this option to train a proper "test" oracle, i.e. if
        # there is no actual ground truth (exact) oracle which exists.
        logger.warning("Training using the full dataset, so training test oracle...")
        test_dataset = get_dataset(
            task_name=exp_dict["dataset"],
            oracle_name=exp_dict["oracle"],
            split="test",
            gain=exp_dict["gain"],
            gain_y=exp_dict["gain_y"],
        )
        dataset = ConcatDataset((train_dataset, valid_dataset, test_dataset))
        train_dataset = dataset
        valid_dataset = dataset
    elif mode == "train+valid":
        # By default, the train and validation sets are merged together
        # and a small internal validation set is set aside.
        dataset = ConcatDataset((train_dataset, valid_dataset))
        rnd_state = np.random.RandomState(0)
        indices = np.arange(0, len(dataset))
        rnd_state.shuffle(indices)
        train_indices = indices[0 : int(0.95 * len(indices))]
        valid_indices = indices[int(0.95 * len(indices)) : :]

        train_dataset = Subset(dataset, indices=train_indices)
        valid_dataset = Subset(dataset, indices=valid_indices)
    elif mode == "train":
        # Only train on the training set. Set aside a small % of the
        # training set for validation (5%).
        rnd_state = np.random.RandomState(0)
        indices = np.arange(0, len(train_dataset))
        rnd_state.shuffle(indices)
        train_indices = indices[0 : int(0.95 * len(indices))]
        valid_indices = indices[int(0.95 * len(indices)) : :]
        train_train_dataset = Subset(train_dataset, indices=train_indices)
        valid_train_dataset = Subset(train_dataset, indices=valid_indices)

        train_dataset = train_train_dataset
        valid_dataset = valid_train_dataset
    else:
        raise Exception("mode must be either train, train+valid, train+valid+test")

    dataset_M = exp_dict["dataset_M"]
    if dataset_M > 0:
        logger.info("Duplicating dataset M={} times".format(dataset_M))
        train_dataset_ = DuplicateDatasetMTimes(train_dataset, M=dataset_M)
    else:
        train_dataset_ = train_dataset

    logger.debug("len of train: {}".format(len(train_dataset)))
    logger.debug("len of valid: {}".format(len(valid_dataset)))

    train_loader = DataLoader(
        train_dataset_, shuffle=True, batch_size=exp_dict["batch_size"],
        num_workers=exp_dict['num_workers']
    )

    valid_loader = DataLoader(
        valid_dataset, shuffle=True, batch_size=exp_dict["batch_size"],
        num_workers=exp_dict['num_workers']
    )

    classifier_class = NoiseClassifier if exp_dict["use_noise"] else Classifier
    model = classifier_class(
        n_in=n_features,
        model_kwargs=exp_dict["model_kwargs"],
        optim_kwargs=exp_dict["optim_kwargs"],
        **exp_dict["classifier_kwargs"]
    )

    best_metric = np.inf
    chk_metric = "valid_loss"
    max_epochs = exp_dict["epochs"]
    eval_every = exp_dict["eval_every"]
    score_list = []
    for epoch in range(0, max_epochs):
        t0 = time.time()

        score_dict = {}
        score_dict["epoch"] = epoch

        # if train_sampler is not None:
        #    train_sampler.set_epoch(epoch)
        # if dev_sampler is not None:
        #    dev_sampler.set_epoch(epoch)

        # (1) Train GAN.
        train_dict_ = model.train_on_loader(
            train_loader,
            pbar=False
            # epoch=epoch,
            # pbar=world_size <= 1,
            # pbar=False if 'DISABLE_PBAR' in os.environ else True
        )
        train_dict = {("train_" + key): val for key, val in train_dict_.items()}

        valid_dict_ = model.val_on_loader(
            valid_loader,
            pbar=False
            # epoch=epoch,
            # pbar=world_size <= 1,
            # pbar=False if 'DISABLE_PBAR' in os.environ else True
        )
        valid_dict = {("valid_" + key): val for key, val in valid_dict_.items()}

        score_dict.update(train_dict)
        score_dict.update(valid_dict)
        time_taken = time.time() - t0
        score_dict["time"] = time_taken

        if eval_every > 0 and epoch % eval_every == 0:
            log_dict = {
                **{'epoch': epoch, 'time': time_taken}, **train_dict, **valid_dict
            }
            logger.info(log_dict)

            if score_dict[chk_metric] < best_metric:
                logger.info(
                    "new best metric: from {}={:.4f} to {}={:.4f}".format(
                        chk_metric, best_metric, chk_metric, score_dict[chk_metric]
                    )
                )
                best_metric = score_dict[chk_metric]
                save_checkpoint(
                    savedir,
                    fname_suffix="." + chk_metric,
                    score_list=score_list,
                    model_state_dict=model.get_state_dict()
                )

        score_list += [score_dict]

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
