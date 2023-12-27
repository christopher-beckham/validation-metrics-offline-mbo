import numpy as np
#from haven import haven_utils as hu
import numpy as np
import os
import json
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import Dataset
from typing import Dict, List
from prdc import prdc
import pickle
import torch

from .setup_logger import get_logger

logger = get_logger(__name__)


def load_json_from_file(filename):
    return json.loads(open(filename, "r").read())


def save_checkpoint(savedir: str, 
                    score_list: List, 
                    model_state_dict: Dict, 
                    fname_suffix: str = ""):
    # save score_list
    score_list_fname = os.path.join(savedir, "score_list%s.pkl" % fname_suffix)
    with open(score_list_fname, "wb") as f:
        pickle.dump(score_list, f)
    # save model
    if model_state_dict is not None:
        model_state_dict_fname = os.path.join(savedir, "model%s.pth" % fname_suffix)
        #hu.torch_save(model_state_dict_fname, model_state_dict)
        torch.save(model_state_dict, model_state_dict_fname)


class DuplicateDatasetMTimes(Dataset):
    """Cleanly duplicate a dataset M times. This is to avoid
    the massive overhead associated with data loader resetting
    for small dataset sizes, e.g. the support set which only
    has k examples per class.
    """

    def __init__(self, dataset, M):
        self.dataset = dataset
        self.N_actual = len(dataset)
        self.M = M

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx % self.N_actual)

    def __len__(self):
        return self.N_actual * self.M


def str2bool(st):
    if st.lower() == "true":
        return True
    return False


def get_checkpoint(savedir, return_model_state_dict=False, map_location=None):
    chk_dict = {}

    # score list
    score_list_fname = os.path.join(savedir, "score_list.pkl")
    if os.path.exists(score_list_fname):
        #score_list = hu.load_pkl(score_list_fname)
        score_list = pickle.load(open(score_list_fname, "rb"))
    else:
        score_list = []

    chk_dict["score_list"] = score_list
    if len(score_list) == 0:
        chk_dict["epoch"] = 0
    else:
        chk_dict["epoch"] = score_list[-1]["epoch"] + 1

    model_state_dict_fname = os.path.join(savedir, "model.pth")
    if return_model_state_dict:
        if os.path.exists(model_state_dict_fname):
            chk_dict["model_state_dict"] = torch.load(
                model_state_dict_fname, map_location=map_location
            )

        else:
            chk_dict["model_state_dict"] = {}

    return chk_dict


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def count_params(module, trainable_only=True):
    """Count the number of parameters in a
    module.
    :param module: PyTorch module
    :param trainable_only: only count trainable
      parameters.
    :returns: number of parameters
    :rtype:
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num

# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def dict2name(x: Dict, keys: List[str], sep: str = ",") -> str:
    """Convert a dictionary to a string representation based on a list of provided keys"""
    return sep.join([str(a) + "=" + str(b) for a, b in x.items() if a in keys])