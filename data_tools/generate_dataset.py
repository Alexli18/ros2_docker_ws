import numpy as np
import os
import matplotlib.pyplot as plt

# ---- CONFIGURATION -------------------------------------------------
SAVE_PNG = False      # Set True only when you need visual examples
# --------------------------------------------------------------------

def generate_bscan(with_mine=False):
    bscan = np.random.normal(0, 0.04, (256, 512))

    # добавим синусоиды
    for i in range(3):
        freq = np.random.uniform(0.01, 0.05)
        amp = np.random.uniform(0.02, 0.05)
        bscan += amp * np.sin(np.linspace(0, 2 * np.pi * freq * 512, 512))[None, :]

    for i in range(2):
        freq = np.random.uniform(0.01, 0.03)
        amp = np.random.uniform(0.01, 0.02)
        bscan += amp * np.sin(np.linspace(0, 2 * np.pi * freq * 256, 256))[:, None]

    if with_mine:
        x = np.arange(512)
        offset = np.random.randint(100, 150)
        width = np.random.uniform(200, 400)
        amplitude = np.random.uniform(0.6, 1.0)
        center = np.random.randint(100, 400)
        shape_type = np.random.choice(['gaussian', 'box', 'sine'])
        if shape_type == 'gaussian':
            anomaly = amplitude * np.exp(-((x - center)**2) / width)
            bscan[offset:offset+10] += anomaly
        elif shape_type == 'box':
            bscan[offset:offset+10, center-5:center+5] += amplitude
        elif shape_type == 'sine':
            anomaly = amplitude * np.sin(np.linspace(0, 10*np.pi, 512))
            bscan[offset:offset+10] += anomaly

    # add occasional hard‑background spikes (edge echoes) for negative class
    if not with_mine and np.random.rand() < 0.3:
        spike_amp = 0.4 * np.random.rand()
        bscan[:, :20]  += spike_amp
        bscan[:, -20:] += spike_amp

    return bscan

def save_example(bscan, label, index):
    out_dir = f"dataset/{label}"
    os.makedirs(out_dir, exist_ok=True)

    # always save npy (fast and lightweight)
    np.save(f"{out_dir}/bscan_{index}.npy", bscan)

    # optionally save PNG preview
    if SAVE_PNG:
        plt.imsave(f"{out_dir}/bscan_{index}.png", bscan, cmap='viridis')


if __name__ == "__main__":
    for label in ['no_mine', 'with_mine']:
        for i in range(500):   # 500 examples per class (faster debug)
            bscan = generate_bscan(with_mine=(label == 'with_mine'))
            save_example(bscan, label, i)