import argparse
import json
import os
from typing import Dict, Union
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal

#from haven import haven_utils as hu

# from torchvision.utils import save_image
import numpy as np
import random
from tqdm import tqdm
import pickle

from src.fid import fid_score

from torch.utils.data import Dataset

# from src.models import utils_viz
from src.models import Diffusion, Classifier
from src.models.base_model import BaseModel

from src import setup_logger
from src.utils import load_json_from_file, suppress_stdout_stderr

logger = setup_logger.get_logger(__name__)

from src.datasets import get_dataset

import torch.distributed as dist

from trainval import compute_fid_reference_stats, _init_from_args

@torch.no_grad()
def test_oracle(valid_loader, test_loader):
    dataset = valid_loader.dataset
    # task = dataset.task
    mse = []
    for X_batch, y_batch in valid_loader:
        y_batch_np = y_batch[:, 0].numpy()
        y_batch_np = dataset.denorm_y(y_batch_np)
        pred_y = dataset.predict(X_batch)
        mse.append(np.mean((pred_y - y_batch_np) ** 2))
    print(np.mean(mse), "+/-", np.std(mse))


def load_pkl(filename):
    with open(filename, "r") as f:
        return pickle.load(f)

def get_oracle_predictions(
    model: BaseModel,
    classifier: Classifier,
    dataset: Dataset,
    use_ema: bool = False,
    sample_kwargs: Union[None, dict] = None,
    N_evals_per_y: int = 32,
    N_linspace: int = 100,
    verbose: bool = True,
) -> Dict:
    """For some range of y, generate candidates with the model x ~ p(x|y) and
      run these through both the validation oracle (ours) and the test
      oracle (Design-Bench).

    Args:
        model (BaseModel): _description_
        classifier (Classifier): validation classifier
        classifier_test: test classifier (ground truth oracle)
        dataset (torch.Dataset): _description_
        N_evals_per_y (int, optional): _description_. Defaults to 32.
        N_linspace (int, optional): _description_. Defaults to 100.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        a dictionary of metrics
    """

    # We assume that the generative model takes normalised
    # values of y, so these will be normalised.
    y_min = dataset.y.min().item()
    y_max = dataset.y.max().item()
    
    y_linspace = np.linspace(y_min, y_max, num=N_linspace)
    zeros = torch.zeros((N_evals_per_y, 1)).to(model.rank)

    if sample_kwargs is None:
        sample_kwargs = {}
    else:
        logger.debug("sample_kwargs: {}".format(sample_kwargs))

    logger.debug("N_linspace: {}".format(N_linspace))

    TABLE_HEADER = "{}\t\t\t{}\t\t\t{}\t\t\t{}\t\t\t{}"
    TABLE_ROW = "{:.2f} ({:.2f})\t\t{:.2f} ({:.2f})\t\t{:.2f} ({:.2f})\t\t{:.2f} ({:.2f})\t\t{:.2f} ({:.2f})"
    
    print(TABLE_HEADER.format(
        "y", "valid µ", "valid σ", "test µ", "test σ"
    ))
    print(TABLE_HEADER.format(
        *(["-------"]*5)
    ))

    BUFFER = dict(
        pred_y=[], pred_y_gt=[], pred_y_denorm=[], pred_y_gt_denorm=[]
    )

    #print("y\t\t norm\t\tvalid µ\t\tvalid σ\t\ttest µ\t\ttest σ\t")
    
    for yval in y_linspace:

        # Our model expects normalised y, so do that here
        this_ycond = zeros + yval
        this_xfake = model.sample(this_ycond, 
                                  use_ema=use_ema,
                                  **sample_kwargs)

        # Predict our sampled candidate with validation oracle
        # We run the prediction through denorm_y() because our
        # own oracle outputs normalised y's.

        # valid preds are normalised already since it's our oracle,
        # so we must denormalise them.
        with torch.no_grad():
            this_valid_pred = classifier.predict(this_xfake).cpu().numpy()
        this_valid_pred_denorm = dataset.denorm_y(this_valid_pred)
        this_valid_pred = this_valid_pred.flatten()
        this_valid_pred_denorm = this_valid_pred_denorm.flatten()

        # test preds are already de-normed so we must normalise
        # them.
        with suppress_stdout_stderr():
            # Weird stuff gets printed out on the DB side so
            # don't print them here.
            this_test_pred_denorm = dataset.predict(this_xfake)
        this_test_pred = dataset.norm_y(this_test_pred_denorm)

        # Predict our sampled candidate with test oracle.
        # Do not run the prediction through any denorm, design-bench
        # handles this stuff internally.
        #with suppress_stdout_stderr():
        #    this_test_pred = dataset.predict(this_xfake.cpu()).flatten()

        if verbose:
            print(TABLE_ROW.format(
                dataset.denorm_y(yval), yval,
                np.mean(this_valid_pred_denorm), np.mean(this_valid_pred),
                np.std(this_valid_pred_denorm), np.std(this_valid_pred),
                np.mean(this_test_pred_denorm), np.mean(this_test_pred),
                np.std(this_test_pred_denorm), np.std(this_test_pred),
            ))

        BUFFER['pred_y'].append(this_valid_pred)
        BUFFER['pred_y_gt'].append(this_test_pred)
        BUFFER['pred_y_denorm'].append(this_valid_pred_denorm)
        BUFFER['pred_y_gt_denorm'].append(this_test_pred_denorm)

    BUFFER = {k:np.asarray(v) for k,v in BUFFER.items()}

    # EXTRA: compute the regular agreement over the validation distribution
    # How training code does agreement:

    # Compute agreement
    """
    tgt_all_y = dataset.y.to(model.rank)
    gen_samples = model.sample(tgt_all_y, use_ema=use_ema)

    with torch.no_grad():
        pred_y = classifier.predict(gen_samples).cpu().numpy().flatten()
    tgt_all_y = tgt_all_y.cpu().flatten().numpy()
    #tgt_agg_norm = np.mean((pred_y - tgt_all_y) ** 2)
    tgt_agg_denorm = np.mean(
        (dataset.denorm_y(tgt_all_y) - dataset.denorm_y(pred_y)) ** 2
    )
    """
    
    """

    #this_xfake = model.sample(dataset.y.to(model.rank))
    #this_pred = dataset.denorm_y(
    #    classifier.predict(this_xfake).cpu().numpy()
    #).flatten()
    #actual_y = dataset.denorm_y(dataset.y.numpy().flatten())
    """
    #agreement = tgt_agg_denorm
    #logger.info("agreement over p(y): {:.4f}".format(agreement))
    #BUFFER["agr_denorm"] = agreement

    BUFFER["y"] = y_linspace
    BUFFER["y_denorm"] = dataset.denorm_y(y_linspace)
    return BUFFER


def _run(exp_dir, savedir, method, method_kwargs, seed, args):

    if seed is not None:
        print("Setting seed to: {}".format(seed))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Setup.
    print("Loading {}...".format(exp_dir))
    exp_dict = json.loads(open("{}/exp_dict.json".format(exp_dir)).read())

    if args.model is not None:

        datasets, model = _init_from_args(exp_dict=exp_dict)

        train_dataset = datasets[0]

        chk_dict = torch.load("{}/{}".format(exp_dir, args.model))
        score_list = load_pkl(
            "{}/{}".format(
                exp_dir, args.model.replace("model", "score_list").replace(".pth", ".pkl")
            )
        )
        score_list = [x for x in score_list if exp_dict["valid_metrics"][0] in x]
        logger.warning(
            "This checkpoint was saved at EPOCH={}".format(score_list[-1]["epoch"])
        )

        model.set_state_dict(chk_dict, strict=True)

    else:

        logger.warning("Using Bprop for model class!!!")

        # if model is None, load in the backprop model.
        datasets, model = _init_from_args(rank=0, exp_dict=exp_dict, skip_model = True)

    train_dataset = datasets[0]
    valid_dataset = datasets[1]
    test_dataset = datasets[2]

    # Load our own pre-trained oracle for FID
    # computation purposes.
    logger.info("Loading pretrained valid oracle...")
    cls_exp_dict = load_json_from_file(
        "{}/exp_dict.json".format(os.path.dirname(exp_dict["pretrained_oracle"]))
    )
    cls = Classifier(
        model_kwargs=cls_exp_dict["model_kwargs"],
        optim_kwargs=cls_exp_dict["optim_kwargs"],
    )
    cls.set_state_dict(torch.load(exp_dict["pretrained_oracle"]))

    if type(model) == Bprop:
        model.set_classifier(cls)

    """
    logger.info("Loading pretrained test oracle...")
    cls_test_exp_dict = load_json_from_file(
        "{}/exp_dict.json".format(os.path.dirname(args.test_oracle))
    )
    cls_test = Classifier(
        model_kwargs=cls_test_exp_dict["model_kwargs"],
        optim_kwargs=cls_test_exp_dict["optim_kwargs"],
    )
    cls_test.set_state_dict(torch.load(args.test_oracle))
    """

    logger.info("len of valid dataset: {}".format(len(valid_dataset)))
    logger.info("len of test dataset: {}".format(len(test_dataset)))

    # returns (path, basename)
    expdir_split = os.path.split(os.path.abspath(args.experiment))
    expdir_name = "{}/{}".format(os.path.basename(expdir_split[0]), expdir_split[1])
    tmp_savedir = "{}/{}".format(args.savedir, expdir_name)

    metadata = vars(args)

    if not os.path.exists(tmp_savedir):
        os.makedirs(tmp_savedir)

    # valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    # test_oracle(valid_loader, test_loader)

    # logger.info("Method invoked: {}".format(args.method))
    # invoke_method = METHODS[args.method]
    valid_stats = compute_fid_reference_stats(valid_dataset, cls, return_features=True)
    test_stats = compute_fid_reference_stats(test_dataset, cls, return_features=True)

    if method == "plot":

        # ------------------------
        # Dump predictions to disk
        # ------------------------

        pkl_buf = {}
        file_ext = method_kwargs.pop('file_ext', 'pkl')
        print("File ext:", file_ext)
        for dataset, dataset_name in zip(
            [train_dataset, valid_dataset], ["train", "valid"]
        ):
            this_preds = get_oracle_predictions(model=model, 
                                                classifier=cls, 
                                                dataset=dataset, 
                                                use_ema=args.use_ema,
                                                sample_kwargs=args.sample_kwargs,
                                                **method_kwargs)
            pkl_buf[dataset_name] = this_preds
        if type(model) == Bprop:
            out_file = "{}/preds-bprop.{}".format(tmp_savedir, file_ext)
        else:
            out_file = "{}/preds-{}.{}".format(tmp_savedir, 
                                               args.model.replace(".pth", ""), 
                                               file_ext)
        logger.info("Saving to pkl file: {}".format(out_file))
        with open(out_file, "wb") as f:
            pickle.dump(pkl_buf, f)

    else:

        raise Exception("args.method not recognised")
