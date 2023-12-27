from .dataset import ContinuousDataset

class AntMorphology(ContinuousDataset):
    def __init__(self, oracle_name, *args, **kwargs):
        super().__init__(
            "{}-{}".format("AntMorphology", oracle_name),
            *args, 
            **kwargs
        )
    def _assert_shape_if_x_is_continuous(self, x):
        assert len(x.shape) == 2 and x.shape[1]==60