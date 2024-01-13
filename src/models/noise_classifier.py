import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from tqdm import tqdm

# from .resnet18 import Resnet18

from torch.optim import AdamW as Adam

from ..setup_logger import get_logger

logger = get_logger(__name__)

# from ..models import utils as ut

from .classifier import Classifier


class MLPWithTimestepClassifier(nn.Module):
    def __init__(self, n_timesteps, n_in, n_hidden, n_layers):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            # nn.BatchNorm1d(n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )
        self.temb = nn.Embedding(n_timesteps, n_hidden)
        hidden = []
        for _ in range(n_layers):
            hidden.append(
                nn.Sequential(
                    nn.Linear(n_hidden * 2, n_hidden),
                    nn.LayerNorm(n_hidden),
                    # nn.BatchNorm1d(n_hidden),
                    nn.LeakyReLU(0.2),
                )
            )
        self.hidden = nn.ModuleList(hidden)
        self.out = nn.Linear(n_hidden, 1)
        self.n_in = n_in

    def features(self, x, all_features=True, **kwargs):
        raise NotImplementedError()
        buf = []
        h = self.pre(x)
        for j in range(len(self.hidden)):
            h = self.hidden[j](h)
            buf.append(h)
        if all_features:
            return torch.cat(buf, dim=1)
        else:
            return buf[-1]

    def forward(self, x, t):
        h = self.pre(x)
        t_features = self.temb(t)
        for j in range(len(self.hidden)):
            ht = torch.cat((h, t_features), dim=1)
            h = self.hidden[j](ht)
        pred_y = self.out(h)
        return pred_y


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class NoiseClassifier(Classifier):
    def __init__(self, n_in, model_kwargs, optim_kwargs, n_timesteps, rank=0):
        
        model_kwargs.update(dict(n_in=n_in))
        self.model = MLPWithTimestepClassifier(n_timesteps, **model_kwargs)
        self.model.to(rank)

        self.rank = rank

        logger.info("model: {}".format(self.model))

        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if 'beta1' in optim_kwargs and 'beta2' in optim_kwargs:
            # HACK: older experiments used a separate beta1 and beta2 whereas
            # it should just be betas, so correct that here if that needs to
            # be done.
            logger.warning("beta1 and beta2 found in optim_kwargs, consolidating into betas=(beta1,beta)")
            beta1 = optim_kwargs['beta1']
            beta2 = optim_kwargs['beta2']
            optim_kwargs['betas'] = (beta1, beta2)
            del optim_kwargs['beta1']
            del optim_kwargs['beta2']

        self.opt = Adam(params, **optim_kwargs)
        logger.info("optim: {}".format(self.opt))

        self.setup(n_timesteps=n_timesteps)

        # TODO: cleanup, just setting this here to keep bw compat
        # for now.
        self.n_out = n_in

    def sample_z(self, bs):
        return torch.randn((bs, self.n_out))

    # TODO: This is stolen from diffusion class, but really we shouldn't have
    # duplicate code...
    # TODO: do we need static methods?
    def setup(self, n_timesteps):
        self.betas = linear_beta_schedule(n_timesteps)
        self.n_timesteps = n_timesteps

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) \
        #    / (1. - self.alphas_cumprod)

    @torch.no_grad()
    def q_sample(self, x0, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def log_prob(self, y, x, **kwargs):
        """Compute log p(y|x), where p(y|x) = Normal(y; mu=f(x), 1)"""
        self.eval()
        mean = self.model(x, **kwargs)
        dist = Normal(mean, torch.ones_like(mean))
        return dist.log_prob(y)

    # train the model
    def train_on_loader(
        self,
        loader,
        pbar=True
    ):
        self.train()

        losses = []

        for _, batch in enumerate(tqdm(loader, desc="Training", disable=not pbar)):
            self.opt.zero_grad()

            x_batch, y_batch = batch
            x_batch = x_batch.to(self.rank)
            y_batch = y_batch.to(self.rank)

            bs = x_batch.size(0)
            t = torch.randint(0, self.n_timesteps, (bs,), device=self.rank).long()
            noise = self.sample_z(bs).to(x_batch.device)

            x_batch = self.q_sample(x_batch, t, noise)

            y_pred = self.model(x_batch, t=t)
            loss = torch.mean((y_pred - y_batch) ** 2)

            loss.backward()
            self.opt.step()

            losses.append(loss.item())
            # accs.append(acc.item())

        return {"loss": np.mean(losses)}

        # evaluate the model

    @torch.no_grad()
    def val_on_loader(self, loader, desc="Validating", pbar=True):
        self.eval()

        losses = []

        for _, batch in enumerate(tqdm(loader, desc=desc, disable=not pbar)):
            x_batch, y_batch = batch
            x_batch = x_batch.to(self.rank)
            y_batch = y_batch.to(self.rank)

            t_zeros = torch.zeros((x_batch.size(0),)).long().to(self.rank)
            y_pred = self.model(x_batch, t=t_zeros)
            loss = torch.mean((y_pred - y_batch) ** 2)

            losses.append(loss.item())

        return {"loss": np.mean(losses)}
