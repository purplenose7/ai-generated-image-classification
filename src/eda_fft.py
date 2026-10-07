import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_avg_spectrum(manifest_csv,class_name,n):
    df = pd.read_csv(manifest_csv)
    df = df[df.class_name == class_name].reset_index(drop=True)

    if df.shape[0]<n:
        print(f"Warning: Number of paths({len(paths)}) less than n={n}")
        return None

    sampled_df = df.sample(n=n,random_state=SEED)
    samples = []
    for row in sampled_df.iterrows():
        img = Image.open(row.filepath).convert('L').resize((MODEL_INPUT_SIZE,MODEL_INPUT_SIZE)) #colour adds unnecessary complexity without much insight
        fft_img = np.fft.fft2(np.array(img,dtype=np.float32))
        fft_shifted_img = np.fft.fftshift(fft_img)
        result = np.log1p(np.abs(fft_shifted_img))
        samples.append(result)
    
    return np.stack(samples,axis=0).mean(axis=0)

def plot_class_spectra(spectra_by_class, out_path):
    all_values = np.concatenate([s.flatten() for s in spectra_by_class.values()])
    vmin, vmax = all_values.min(), all_values.max()
    
    fig, axes = plt.subplots(1, len(spectra_by_class), figsize=(6*len(spectra_by_class), 5))
    for ax, (class_name, spectrum) in zip(axes, spectra_by_class.items()):
        im = ax.imshow(spectrum, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(class_name)
        ax.axis('off')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Average Log-Magnitude FFT Spectrum by Class')
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()
    print(f"Saved to {out_path}")
    

def show_samples(manifest_csv, n_per_class, out_path):
    df = pd.read_csv(manifest_csv)
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class,
                             figsize=(3*n_per_class, 3*len(CLASS_NAMES)))

    for row_idx, class_name in enumerate(CLASS_NAMES):
        class_df = df[df['class_name'] == class_name].sample(n=n_per_class, random_state=SEED)
        for col_idx, (_, data_row) in enumerate(class_df.iterrows()):
            img = Image.open(data_row['filepath']).convert('RGB')
            axes[row_idx][col_idx].imshow(img)
            axes[row_idx][col_idx].axis('off')
            if col_idx == 0:
                axes[row_idx][col_idx].set_ylabel(class_name, fontsize=13, rotation=90, labelpad=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()
    print(f"Saved to {out_path}")