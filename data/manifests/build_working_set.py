import numpy as np
import pandas as pd
import os 
from config.settings import *

set_global_seed(SEED)

def list_source_images(subset_root, split, label_folder):
    """Returns a sorted list of paths of all source images

    Args:
        subset_root (Path): _description_
        split (Path): _description_
        label_folder (Path): _description_

    Returns:
        list
    """
    path = subset_root/split/label_folder
    
    li = []
    for filename in os.listdir(path):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png") or filename.lower().endswith(".jpeg"):
            li.append(path/filename)
    return sorted(li)

def sample_paths(paths,n,seed):
    """Returns list of randomly selected n paths

    Args:
        paths (list(Path)): List of all paths
        n (int): Number of paths required
        seed (int): Random seed

    Returns:
        list(str)
    """
    
    generator = np.random.default_rng(seed)
    if len(paths)<n:
        print(f"Warning: Number of paths({len(paths)}) less than n={n}")
        return paths
    return [str(p) for p in generator.choice(paths, n)]

def build_working_set(class_spec, out_csv, seed):
    rows = []
    for source in class_spec:
        paths = []
        for split in ['train','val']:
            paths.extend(list_source_images(source['root'], split, source['label_folder']))
        sampled = sample_paths(paths, source['n'], seed)
        for path in sampled:
            rows.append({'filepath': path,
                         'class_name': source['class_name'],
                         'class_idx': CLASS_TO_IDX[source['class_name']],
                         'generator': source['generator']})
    df = pd.DataFrame(rows)
    counts = df['class_name'].value_counts()
    assert len(counts.unique()) == 1, f"Class imbalance detected: {counts.to_dict()}"
    df.to_csv(out_csv, index=False)
    return df

def build_ood_set(glide_root, id_csv, n_ai, n_real, out_csv, seed):
    id_df = pd.read_csv(id_csv)
    id_paths = set(id_df['filepath'].tolist())

    glide_paths = list_source_images(glide_root, 'train', 'ai')
    sampled_glide = sample_paths(glide_paths, n_ai, seed)
    glide_rows = [{'filepath': p,
                   'class_name': 'diffusion',
                   'class_idx': CLASS_TO_IDX['diffusion'],
                   'generator': 'glide',
                   'source_split': 'train',
                   'is_ood': True} for p in sampled_glide]

    nature_paths = []
    nature_paths.extend(list_source_images(glide_root, 'train', 'nature'))
    nature_paths.extend(list_source_images(glide_root, 'val', 'nature'))
    nature_paths_filtered = [str(p) for p in nature_paths if str(p) not in id_paths]
    sampled_nature = sample_paths(nature_paths_filtered, n_real, seed)
    nature_rows = [{'filepath': p,
                    'class_name': 'real',
                    'class_idx': CLASS_TO_IDX['real'],
                    'generator': 'nature',
                    'source_split': 'train',
                    'is_ood': True} for p in sampled_nature]

    df = pd.concat([pd.DataFrame(glide_rows), pd.DataFrame(nature_rows)], ignore_index=True)
    df.to_csv(out_csv, index=False)
    return df