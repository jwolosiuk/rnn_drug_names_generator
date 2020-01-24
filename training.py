import numpy as np
import pandas as pd
from collections import Counter
from unidecode import unidecode
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from livelossplot import PlotLosses

