from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import warnings
warnings.simplefilter("ignore")