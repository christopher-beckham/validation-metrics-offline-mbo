from .dataset import DiscreteDataset

import numpy as np

from ..setup_logger import get_logger
logger = get_logger(__name__)

class TfBind8(DiscreteDataset):
    def __init__(self, oracle_name, *args, **kwargs):
        super().__init__(
            "{}-{}".format("TFBind8", oracle_name),
            *args, 
            **kwargs
        )

    def get_custom_oracle(self, all_x_discrete: np.ndarray, all_y: np.ndarray):
        from sklearn.neighbors import KNeighborsRegressor
        knn = KNeighborsRegressor(n_neighbors=1)
        knn.fit(all_x_discrete, all_y)
        logger.debug("custom test oracle trained: {}".format(knn))
        # It should fit the data perfectly
        assert knn.score(all_x_discrete, all_y) == 1.
        return knn

    def mask_dataset(self, x: np.ndarray, y: np.ndarray):
        return None

    def norm_x(self, x: np.ndarray):
        assert type(x) == np.ndarray
        # check if it's discrete shape first
        self._assert_shape_if_x_is_discrete(x)
        x_onehot = np.eye(4)[x]
        x_onehot = x_onehot.reshape(-1, 8*4)
        return x_onehot
    
    def denorm_x(self, x: np.ndarray):
        assert type(x) == np.ndarray
        # check if it's continuous shape first
        self._assert_shape_if_x_is_continuous(x)
        return x.reshape(-1, 8, 4).argmax(2)

    def _assert_shape_if_x_is_discrete(self, x):
        assert len(x.shape) == 2 and x.shape[1]==8

    def _assert_shape_if_x_is_continuous(self, x):
        assert len(x.shape) == 2 and x.shape[1]==8*4
        
if __name__ == '__main__':
    ds = TfBind8("FullyConnected-v0")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    x_batch, y_batch = iter(loader).next()
    print(x_batch.shape, y_batch.shape, y_batch.min(), y_batch.max())
    x_batch_d = ds.to_discrete(x_batch.numpy())
    assert np.mean(ds.to_onehot(x_batch_d) == x_batch.numpy()) == 1.0
    print("test scoring function:")
    mse_mu, mse_std = ds.score(x_batch_d, y_batch.numpy())
    print(mse_mu, mse_std)