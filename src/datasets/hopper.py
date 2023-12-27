from .dataset import ContinuousDataset
import numpy as np

class Hopper50(ContinuousDataset):
    def __init__(self, oracle_name, *args, **kwargs):
        super().__init__(
            "{}-{}".format("HopperController", oracle_name),
            *args, 
            **kwargs,
            subsample_flags=dict(max_samples=10000, 
                                 distribution="uniform", 
                                 min_percentile=0, 
                                 max_percentile=50)
        )
        
    def _assert_shape_if_x_is_continuous(self, x):
        assert len(x.shape) == 2 and x.shape[1]==5126