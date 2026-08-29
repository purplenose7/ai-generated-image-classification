import numpy as np
import torch
from pathlib import Path

SEED = 77
CLASS_NAMES = ['real','gan','diffusion']
CLASS_TO_IDX = {'real':0,'gan':1,'diffusion':2}
PROJECT_DIR = Path.cwd().resolve()
FIGURE_DIR = PROJECT_DIR/'figures'

def set_global_seed(seed):
    random.seed(SEED)
    torch.random.seed(SEED)
    np.random.set_global_seed(SEED)
