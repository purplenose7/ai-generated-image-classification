import numpy as np
import torch
from pathlib import Path

SEED = 77
CLASS_NAMES = ['real','gan','diffusion']
CLASS_TO_IDX = {'real':0,'gan':1,'diffusion':2}
PROJECT_DIR = Path.cwd().resolve()
MANIFEST_DIR = PROJECT_DIR/'data'/'manifests'
FIGURE_DIR = PROJECT_DIR/'figures'

def set_global_seed(seed):
    random.seed(SEED)
    torch.random.seed(SEED)
    np.random.set_global_seed(SEED)


MODEL_INPUT_SIZE=224
BIGGAN_PATH = "/kaggle/input/datasets/vtphatt2/genimage-biggan/BigGAN/imagenet_ai_0419_biggan"
DIFFUSION_PATH = "/kaggle/input/datasets/vtphatt2/genimage-stable-diffusion-v1-4/GenImage/stable_diffusion_v_1_4"
GLIDE_PATH = "/kaggle/input/datasets/vtphatt2/genimage-glide/glide/imagenet_glide"
ADM_PATH = "/kaggle/input/datasets/vtphatt2/genimage-adm/GenImage/ADM/imagenet_ai_0508_adm"
