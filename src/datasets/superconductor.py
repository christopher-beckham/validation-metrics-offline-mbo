from .dataset import ContinuousDataset

class Superconductor(ContinuousDataset):
    def __init__(self, oracle_name, *args, **kwargs):
        super().__init__(
            "{}-{}".format("Superconductor", oracle_name),
            *args, 
            **kwargs
        )
    def _assert_shape_if_x_is_continuous(self, x):
        assert len(x.shape) == 2 and x.shape[1]==86