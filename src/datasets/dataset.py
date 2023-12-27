from typing import Union
from ..utils import suppress_stdout_stderr
with suppress_stdout_stderr():
    import design_bench

import torch
from torch.utils.data import Dataset
import numpy as np

from ..setup_logger import get_logger

logger = get_logger(__name__)


class GenericDataset(Dataset):
    """
    Dataset which wraps Design Bench's Task class.

    Inputs are preprocessed to conform with what the task oracle
    expects. Optionally, this class may implements its own post-
    processing methods. If any method is called which invokes the
    oracle, e.g. `score` or `predict`, the data's post-processing
    step will be inverted so that it is compatible with what the
    task oracle expects.
    """

    def _mask_xy(self, x, y, mask, invert=True):
        """If mask is None, just return x and y"""
        if mask is None:
            return x, y
        else:
            if invert:
                mask = (~mask).tolist()
            else:
                mask = mask.tolist()
            return x[mask], y[mask]

    def _print_summary_stats(self, X):
        for j in range(X.shape[1]):
            logger.info("{}\tμ={:.4f}\tσ={:.4f}\tmin={:.4f}\tmax={:.4f}".format(
                j, X[:,j].mean(), X[:,j].std(), X[:,j].min(), X[:,j].max()
            ))

    def __init__(self, task_name, split="train", test_size=0.5, gain=1.0, gain_y=None, subsample_flags=None):
        super().__init__()
        assert split in ("train", "valid", "test")

        task = design_bench.make(task_name)

        if subsample_flags is not None:
            logger.debug("subsample flags set: {}".format(subsample_flags))
            task.dataset.subsample(**subsample_flags)

        # Collect the data.
        if split == "train":
            # The entire dataset is what we call the test set.
            # The training set is a special subsample of the test set,
            # where we only consider y's below some threshold y_max.
            X, y = task.x, task.y
            # invert because we want the non-invalid examples            
            X, y = self._mask_xy(X, y, self.mask_dataset(X, y), invert=True)
        else:
            X, y = task.oracle.internal_dataset.x, task.oracle.internal_dataset.y
            # invert because we want the non-invalid examples
            X, y = self._mask_xy(X, y, self.mask_dataset(X, y), invert=True)
            # X_all and y_all comprise the entire dataset.
            # First we subset it so that all the y's are above
            # that of task.y.
            X = X[(y > task.y.max()).flatten()]
            y = y[(y > task.y.max()).flatten()]
            # From this subset, assign half to be validation
            # and half to be testing.
            
            rnd_state = np.random.RandomState(42)
            idcs = np.arange(0, len(X))
            rnd_state.shuffle(idcs)
            valid_idcs = idcs[0 : int(len(idcs) * test_size)]
            test_idcs = idcs[int(len(idcs) * test_size) :]
            if split == "valid":
                X = X[valid_idcs]
                y = y[valid_idcs]
            else:
                X = X[test_idcs]
                y = y[test_idcs]

        """
        The oracle has its own methods for dealing with oracle x,y and dataset x,y,
        e.g. dataset_to_oracle_{x,y} and oracle_to_dataset_{x,y}, and we need not
        interfere with these. When oracle.predict gets called, the following
        steps happen:

        x = dataset_to_oracle_x(x)
        y = predict(x)
        y = oracle_to_dataset_y(y)
        return y

        This means that from our pov, all we need to do is make sure
        our own denorm_x gets called before we input it to the oracle.
        """

        logger.debug("split            = {}".format(split))
        logger.debug("  X.shape        = {}".format(X.shape))
        logger.debug("  gain           = {}".format(gain))
        logger.debug("  gain_y         = {}".format(gain_y))
        logger.debug("  X min max      = {:.4f} {:.4f}".format(X.min(), X.max()))
        logger.debug("  y min max      = {:.4f} {:.4f}".format(y.min(), y.max()))
        
        # self.min_X, self.max_X = np.min(task.x), np.max(task.x) # compute statistics based on train

        # MUST USE task.x, task.y, not (X,y)
        self.set_normalisation_parameters(task.x, task.y)
        self.gain = gain  
        if gain_y is None:
            logger.debug("gain_y = None so set gain_y = gain = {}".format(gain))
            self.gain_y = self.gain
        else:
            self.gain_y = gain_y

        X_orig = X
        y_orig = y

        #self._print_summary_stats(X_orig)
        
        X = self.norm_x(X)
        y = self.norm_y(y)
        logger.debug(
            "  after norm              = {:.4f} {:.4f}, {}".format(
                X.min(), X.max(), X.shape
            )
        )
        logger.debug(
            "  after norm              = {:.4f} {:.4f}, {}".format(
                y.min(), y.max(), y.shape
            )
        )

        # Ensure that our normalisation and denormalisation works
        if split == "train":
            assert np.isclose(self.denorm_x(X), X_orig, atol=1e-4).all() 
            assert np.isclose(self.denorm_y(y), y_orig, atol=1e-4).all()

        self.task = task
        self.oracle = task.oracle

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

        # logger.info("oracle.internal_dataset.y range: ({}, {})".format(
        #    self.oracle.internal_dataset.y.min(),
        #    self.oracle.internal_dataset.y.max()
        # ))
        self.split = split


    def mask_dataset(self, x: np.ndarray, y: np.ndarray) -> Union[None, np.ndarray]:
        return None

    def get_custom_oracle(self, all_x: np.ndarray, all_y: np.ndarray):
        """Return a custom test oracle here. By default, this simply
        returns None. If it returns not None, then self.oracle will
        be overriden with this.

        Args:
            all_x: entire dataset x's
            all_y: entire dataset y's

        Returns:
            something that has a predict() method.
        """
        return None
    
    def set_normalisation_parameters(self, task_x, task_y):
        raise NotImplementedError()
    
    def norm_x(self, x):
        """For a discrete dataset, this should map discrete tokens to continuous vectors"""
        raise NotImplementedError()

    def denorm_x(self, x_normed):
        """For a discrete dataset, this should map continuous vectors back to discrete tokens"""
        raise NotImplementedError()

    def norm_y(self, y):
        """Normalise y values"""
        raise NotImplementedError()

    def denorm_y(self, y_normed):
        """Denormalise y values"""
        raise NotImplementedError()

    @property
    def n_in(self):
        return self.X.size(1)

    @torch.no_grad()
    def score(self, x: torch.Tensor, y: torch.Tensor):
        """Predict the MSE between predicted return for x
        and actual y. This method will internally handle
        any conversions needed for X before it is input to
        the task oracle. For example, if `post_process` is
        True, then this will be undone on x before it gets
        input to the oracle.

        Args:
            x (torch.Tensor): mini-batch x in torch format.
            y (torch.Tensor): mini-batch y, in torch format.

        Returns:
            numpy.array: an array of residuals (y_pred-y)**2
        """
        assert type(x) == torch.Tensor
        assert type(y) == torch.Tensor
        self._assert_shape_if_x_is_continuous(x)
        assert len(x) == len(y)
        if x.is_cuda:
            x = x.cpu()
        if y.is_cuda:
            y = y.cpu()
        x = x.numpy()
        y = y.numpy()
        x = self.denorm_x(x)
        y = self.denorm_y(y)
        assert len(x) == len(y)
        y_pred = self.oracle.predict(x).flatten()
        mse = (y_pred - y.flatten()) ** 2
        # return np.mean(mse), np.std(mse)
        return mse

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.array:
        """Predict the score for a candidate using the oracle embedded in 
          `self.task` (that is, the ground-truth oracle Design Bench provides).
          The oracle here for all intents and purposes fulfils the purpose of a
          'test set', so this should only be used for final evaluation.
        
        Note that this method automatically handles GPU to CPU, denormalisation,
          as well as conversion back into numpy format before being passed
          into the task oracle.

        Args:
            x: candidate to predict

        Returns:
            y_pred: a numpy array of predictions.
        """
        assert type(x) == torch.Tensor
        self._assert_shape_if_x_is_continuous(x)
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
        x = self.denorm_x(x)
        y_pred = self.oracle.predict(x).flatten()
        return y_pred

class ContinuousDataset(GenericDataset):

    @property
    def is_discrete(self):
        return False

    def set_normalisation_parameters(self, task_x, task_y):
        self.min_X = task_x.min(axis=0, keepdims=True)
        self.max_X = task_x.max(axis=0, keepdims=True)
        self.min_y, self.max_y = np.min(task_y), np.max(task_y)
    
    def norm_x(self, X):
        """Convert X into a format amenable to training"""
        # X = (X-self.mean_X) / self.std_X
        # X = (X-0.5) / 0.5
        X = (X - self.min_X) / (self.max_X - self.min_X + 1e-6)
        X = ((X - 0.5) / 0.5) * self.gain
        return X

    def denorm_x(self, x_normed):
        """Denormalisation for Design Bench test oracle"""
        # x_normed = x_normed*0.5 + 0.5
        x_normed = (x_normed * 0.5 + (0.5*self.gain)) / self.gain
        return x_normed * (self.max_X - self.min_X + 1e-6) + self.min_X

    def norm_y(self, y):
        """Convert y into a format amenable to training"""
        y = (y - self.min_y) / (self.max_y - self.min_y)
        y = y*self.gain_y
        return y

    def denorm_y(self, y_normed):
        """Denormalisation for Design Bench test oracle"""
        y_normed = y_normed / self.gain_y
        return (y_normed * (self.max_y - self.min_y)) + self.min_y

class DiscreteDataset(GenericDataset):

    @property
    def is_discrete(self):
        return True

    def set_normalisation_parameters(self, task_x, task_y):
        self.min_y, self.max_y = np.min(task_y), np.max(task_y)

    def norm_y(self, y):
        """Convert y into a format amenable to training"""
        y = (y - self.min_y) / (self.max_y - self.min_y)
        y = y*self.gain_y
        return y

    def denorm_y(self, y_normed):
        """Denormalisation for Design Bench test oracle"""
        y_normed = y_normed / self.gain_y
        return (y_normed * (self.max_y - self.min_y)) + self.min_y