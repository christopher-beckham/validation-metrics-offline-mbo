import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW as Adam

from ..setup_logger import get_logger

logger = get_logger(__name__)


class MLPClassifier(nn.Module):
    def __init__(self, n_in, n_hidden, n_layers):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            # nn.BatchNorm1d(n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )
        hidden = []
        for _ in range(n_layers):
            hidden.append(
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    nn.LayerNorm(n_hidden),
                    # nn.BatchNorm1d(n_hidden),
                    nn.LeakyReLU(0.2),
                )
            )
        self.hidden = nn.ModuleList(hidden)
        self.out = nn.Linear(n_hidden, 1)
        self.n_in = n_in

    def features(self, x, all_features=True, **kwargs):
        buf = []
        h = self.pre(x)
        for j in range(len(self.hidden)):
            h = self.hidden[j](h)
            buf.append(h)
        if all_features:
            return torch.cat(buf, dim=1)
        else:
            return buf[-1]

    def forward(self, x):
        h = self.pre(x)
        for j in range(len(self.hidden)):
            h = self.hidden[j](h)
        pred_y = self.out(h)
        return pred_y


# model definition
class Classifier:
    def __init__(
        self,
        n_in,
        model_kwargs,
        optim_kwargs,
        rank=0,
        verbose=False,
    ):

        if 'n_in' in model_kwargs:
            model_kwargs.pop('n_in')
        
        self.model = MLPClassifier(n_in, **model_kwargs)
        self.model.to(rank)

        self.rank = rank

        logger.info("model: {}".format(self.model))

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if 'beta1' in optim_kwargs and 'beta2' in optim_kwargs:
            # HACK: older experiments used a separate beta1 and beta2 whereas
            # it should just be betas, so correct that here if that needs to
            # be done.
            logger.warning("beta1 and beta2 found in kwargs, consolidating into betas=(beta1,beta)")
            beta1 = optim_kwargs['beta1']
            beta2 = optim_kwargs['beta2']
            optim_kwargs['betas'] = (beta1, beta2)
            del optim_kwargs['beta1']
            del optim_kwargs['beta2']

        print(optim_kwargs, "<<<<<<<<<<,")
        self.opt = Adam(params, **optim_kwargs)

        logger.info("optim: {}".format(self.opt))

    def set_state_dict(self, state_dict, load_opt=True):
        self.model.load_state_dict(state_dict["model"])
        if load_opt:
            self.opt.load_state_dict(state_dict["opt"])

    def get_state_dict(self):
        state_dict = {"model": self.model.state_dict(), "opt": self.opt.state_dict()}
        return state_dict

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def features(self, x, **kwargs):
        self.eval()
        return self.model.features(x, **kwargs)

    def predict(self, x):
        self.eval()
        return self.model(x)

    # train the model
    def train_on_loader(self, loader, pbar: bool = True):
        self.train()

        losses = []

        for _, batch in enumerate(tqdm(loader, desc="Training", disable=not pbar)):
            self.opt.zero_grad()

            x_batch, y_batch = batch
            x_batch = x_batch.to(self.rank)
            y_batch = y_batch.to(self.rank)

            y_pred = self.model(x_batch)
            loss = torch.mean((y_pred - y_batch) ** 2)

            loss.backward()
            self.opt.step()

            losses.append(loss.item())
            # accs.append(acc.item())

        return {"loss": np.mean(losses)}

    # evaluate the model
    @torch.no_grad()
    def val_on_loader(self, loader, desc="Validating", pbar: bool = True):
        self.eval()

        losses = []

        for _, batch in enumerate(tqdm(loader, desc=desc, disable=not pbar)):
            x_batch, y_batch = batch
            x_batch = x_batch.to(self.rank)
            y_batch = y_batch.to(self.rank)

            y_pred = self.model(x_batch)
            loss = torch.mean((y_pred - y_batch) ** 2)

            losses.append(loss.item())

        return {"loss": np.mean(losses)}
