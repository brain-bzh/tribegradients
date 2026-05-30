"""
ridge_regression_subject.py

For a given subject, fit a SINGLE ridge regression across ALL movies combined,
per layer. The train/test split (last 20% of chunks per movie) is done
per-movie first, then all train chunks are pooled across movies before fitting.

Pipeline per layer:
  - For each movie: load features, align to predictions, resample to fMRI TR,
    z-score per chunk, split last 20% chunks as test
  - Pool all train chunks across movies → fit one RidgeCV
  - Evaluate on test chunks of each movie separately → save R² per movie
  - Also save R² on the combined test set

Usage
-----
python ridge_regression_subject.py \
    --subject 1 \
    --basedir /path/to/algonauts2025/download \
    --features_dir /path/to/npz_features \
    --output_dir ./ridge_results \
    [--movies bourne wolf figures life] \
    [--test_ratio 0.2] \
    [--tr 1.49]

Output
------
  <output_dir>/sub-<subject>/
    split_info.npz                           — train/test chunk indices per movie
    <layer_name>_r2_<movie>.npy             — R² per vertex, tested on that movie
    <layer_name>_r2_combined.npy            — R² on all test chunks pooled
    all_r2_scores.npz                       — everything in one file
    summary_r2_by_layer.png                 — mean R² by depth (one bar per movie + combined)
"""

import argparse
import os
import numpy as np
import torch
from torch import nn
from scipy.signal import resample
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.fmri_utils import load_fmri

# ─────────────────────────────────────────────────────────────────────────────
# Layer list
# ─────────────────────────────────────────────────────────────────────────────
FEATURES_LIST = [
    'projector_text',
    'projector_audio',
    'projector_video',
    'encoder_layer0_ffn',
    'encoder_layer1_ffn',
    'encoder_layer2_ffn',
    'encoder_layer3_ffn',
    'encoder_layer4_ffn',
    'encoder_layer5_ffn',
    'encoder_layer6_ffn',
    'encoder_layer7_ffn',
    'low_rank_head',
    
]

# ─────────────────────────────────────────────────────────────────────────────
# Alignment utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_chunk_boundaries(segments):
    starts = np.array([np.round(seg.start).astype(int) for seg in segments])
    boundaries = np.where(np.diff(starts) < 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(segments)]])
    return boundaries, starts


def align_features_to_predictions_chunked(feat_array, segments, predictions,
                                           pool_to=100):
    pooler = nn.AdaptiveAvgPool1d(pool_to)

    fa = feat_array.copy()
    if fa.shape[2] == 200:
        pass
    elif fa.shape[1] == 200:
        fa = fa.transpose(0, 2, 1)
    else:
        raise ValueError(
            f"Cannot locate time dimension (size 200) in shape {feat_array.shape}"
        )

    pooled = pooler(torch.tensor(fa.astype(np.float32))).numpy()
    pooled_flat = pooled.transpose(0, 2, 1).reshape(-1, pooled.shape[1])

    chunk_boundaries, starts = get_chunk_boundaries(segments)
    n_chunks = len(chunk_boundaries) - 1

    chunks_feat, chunks_pred = [], []
    hook_batch_offset = 0

    for c in range(n_chunks):
        seg_s, seg_e   = chunk_boundaries[c], chunk_boundaries[c + 1]
        chunk_starts   = starts[seg_s:seg_e]
        chunk_len      = int(chunk_starts[-1]) + 1
        n_batches      = int(np.ceil(chunk_len / pool_to))

        hook_start     = hook_batch_offset * pool_to
        global_indices = hook_start + chunk_starts

        valid = global_indices < len(pooled_flat)
        if not np.all(valid):
            print(f"    [align] Chunk {c}: {(~valid).sum()} indices clipped (padding)")

        chunks_feat.append(pooled_flat[global_indices[valid]])
        chunks_pred.append(predictions[seg_s:seg_e][valid])
        hook_batch_offset += n_batches

    return chunks_feat, chunks_pred, chunk_boundaries


def zscore(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0,  keepdims=True)
    std[std < 1e-8] = 1.0
    return (arr - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load and align all chunks for one movie, return per-chunk arrays
# ─────────────────────────────────────────────────────────────────────────────
def load_movie_chunks(movie, fmri_chunks, features_dir, layer_name, test_ratio):
    """
    Load features for one movie × one layer, align + resample + z-score.

    Returns
    -------
    chunks_X   : list of np.ndarray  (T_fmri, C)  — one per chunk
    chunks_y   : list of np.ndarray  (T_fmri, V)  — z-scored fMRI, one per chunk
    train_idx  : list of int
    test_idx   : list of int
    """
    feat_path = os.path.join(features_dir, f"{movie}_allfeatures.npz")
    pred_path = os.path.join(features_dir, f"{movie}_predictions.npz")

    npz_feat    = np.load(feat_path)
    npz_pred    = np.load(pred_path, allow_pickle=True)
    predictions = npz_pred['preds']
    segments    = npz_pred['segments']

    feat_array = npz_feat[layer_name]

    chunks_feat, _, _ = align_features_to_predictions_chunked(
        feat_array, segments, predictions
    )

    n_chunks = min(len(chunks_feat), len(fmri_chunks))

    # Train / test split: last ceil(test_ratio * n_chunks) → test
    n_test    = max(1, int(np.ceil(test_ratio * n_chunks)))
    n_train   = n_chunks - n_test
    train_idx = list(range(n_train))
    test_idx  = list(range(n_train, n_chunks))

    # Z-score + resample each chunk to match its fMRI counterpart
    chunks_X, chunks_y = [], []
    for c in range(n_chunks):
        T_fmri       = fmri_chunks[c].shape[0]
        cf_resampled = resample(zscore(chunks_feat[c]), T_fmri, axis=0)
        chunks_X.append(cf_resampled)
        chunks_y.append(zscore(fmri_chunks[c]))

    return chunks_X, chunks_y, train_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Fit one ridge across all movies, evaluate per movie
# ─────────────────────────────────────────────────────────────────────────────
def fit_and_evaluate_layer(layer_name, movies, fmri_stack, features_dir,
                            split_info, test_ratio):
    """
    For one layer:
      1. Pool train chunks from all movies → fit RidgeCV
      2. Evaluate on test chunks of each movie separately
      3. Also evaluate on the combined test set

    Parameters
    ----------
    split_info : dict  movie → {'train_idx': [...], 'test_idx': [...],
                                'chunks_X': [...], 'chunks_y': [...]}
        Pre-loaded from load_movie_chunks; passed in to avoid re-loading
        features for every layer (caller loops over layers).

    Returns
    -------
    r2_per_movie   : dict  movie → np.ndarray (n_vertices,)
    r2_combined    : np.ndarray (n_vertices,)
    """
    # ── Build combined train set ──────────────────────────────────────────────
    X_train_parts, y_train_parts = [], []
    for movie in movies:
        info = split_info[movie]
        for c in info['train_idx']:
            X_train_parts.append(info['chunks_X'][c])
            y_train_parts.append(info['chunks_y'][c])

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    print(f"  [{layer_name:30s}]  "
          f"X_train={X_train.shape}  y_train={y_train.shape}",
          end='', flush=True)

    # ── Fit ───────────────────────────────────────────────────────────────────
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
    ridge.fit(X_train, y_train)

    # ── Evaluate per movie ────────────────────────────────────────────────────
    r2_per_movie = {}
    X_test_all, y_test_all = [], []

    for movie in movies:
        info = split_info[movie]
        X_test_m = np.concatenate([info['chunks_X'][c] for c in info['test_idx']], axis=0)
        y_test_m = np.concatenate([info['chunks_y'][c] for c in info['test_idx']], axis=0)

        y_pred_m = ridge.predict(X_test_m)
        r2_per_movie[movie] = r2_score(y_test_m, y_pred_m, multioutput='raw_values')

        X_test_all.append(X_test_m)
        y_test_all.append(y_test_m)

    # ── Evaluate on combined test set ─────────────────────────────────────────
    X_test_all = np.concatenate(X_test_all, axis=0)
    y_test_all = np.concatenate(y_test_all, axis=0)
    y_pred_all = ridge.predict(X_test_all)
    r2_combined = r2_score(y_test_all, y_pred_all, multioutput='raw_values')

    means = "  ".join(
        f"{m}={r2_per_movie[m].mean():.4f}" for m in movies
    )
    print(f"  →  {means}  combined={r2_combined.mean():.4f}")

    return r2_per_movie, r2_combined


# ─────────────────────────────────────────────────────────────────────────────
# Summary plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary(all_r2_per_movie, all_r2_combined, movies, output_dir):
    """
    Horizontal bar chart: one column per movie + one combined column.
    Rows = layers.
    """
    n_cols  = len(movies) + 1
    fig, axes = plt.subplots(1, n_cols,
                              figsize=(5 * n_cols, 6),
                              sharey=True, squeeze=False)

    columns = movies + ['combined']
    for col, label in enumerate(columns):
        ax = axes[0, col]
        layers = [l for l in FEATURES_LIST
                  if l in (all_r2_combined if label == 'combined'
                           else all_r2_per_movie.get(label, {}))]
        if label == 'combined':
            means = [all_r2_combined[l].mean() for l in layers]
            color = 'tomato'
        else:
            means = [all_r2_per_movie[label][l].mean() for l in layers]
            color = 'steelblue'

        ax.barh(layers, means, color=color, alpha=0.85)
        ax.invert_yaxis()
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel("Mean R² (test chunks)")
        ax.set_title(label)

    plt.suptitle("Ridge R² by encoder depth  —  single model fit across all movies",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, 'summary_r2_by_layer.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSummary plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Ridge regression (one model across all movies) from TRIBE v2 '
                    'features to fMRI, per subject.'
    )
    parser.add_argument('--subject',      type=int,   required=True)
    parser.add_argument('--basedir',      type=str,   required=True,
                        help='Root of Algonauts2025 download (passed to load_fmri)')
    parser.add_argument('--features_dir', type=str,   required=True,
                        help='Directory containing <movie>_allfeatures.npz and '
                             '<movie>_predictions.npz')
    parser.add_argument('--output_dir',   type=str,   default='./ridge_results')
    parser.add_argument('--movies',       type=str,   nargs='+',
                        default=['bourne', 'wolf', 'figures', 'life'])
    parser.add_argument('--test_ratio',   type=float, default=0.2,
                        help='Fraction of chunks per movie held out for test')
    parser.add_argument('--tr',           type=float, default=1.49)
    args = parser.parse_args()

    subj_out = os.path.join(args.output_dir, f"sub-{args.subject:02d}")
    os.makedirs(subj_out, exist_ok=True)

    # ── Load fMRI ─────────────────────────────────────────────────────────────
    print(f"\nLoading fMRI for subject {args.subject} ...")
    fmri = load_fmri(args.basedir, args.subject, average=False)

    fmri_stack = {}
    for movie in args.movies:
        keys = sorted(k for k in fmri.keys() if movie in k)
        fmri_stack[movie] = [fmri[k] for k in keys]
        print(f"  {movie}: {len(keys)} chunks  "
              f"{[fmri[k].shape for k in keys]}")

    # ── Skip movies with missing files ────────────────────────────────────────
    valid_movies = []
    for movie in args.movies:
        feat_ok = os.path.exists(
            os.path.join(args.features_dir, f"{movie}_allfeatures.npz"))
        pred_ok = os.path.exists(
            os.path.join(args.features_dir, f"{movie}_predictions.npz"))
        fmri_ok = len(fmri_stack.get(movie, [])) > 0
        if feat_ok and pred_ok and fmri_ok:
            valid_movies.append(movie)
        else:
            print(f"  [SKIP] {movie}: feat={feat_ok} pred={pred_ok} fmri={fmri_ok}")

    if not valid_movies:
        raise RuntimeError("No valid movies found — check paths.")

    # ── Loop over layers ──────────────────────────────────────────────────────
    # Pre-load chunks for all movies × all layers up front to avoid redundant
    # I/O inside the layer loop. Memory is manageable: features are (T, 1152)
    # floats, not the raw fMRI volumes.

    print(f"\nLoading and aligning chunks for all layers × movies ...")
    # split_info[layer][movie] = {'chunks_X', 'chunks_y', 'train_idx', 'test_idx'}
    split_info_by_layer = {}
    saved_split = {}  # save train/test indices (same for all layers of a movie)

    for layer_name in FEATURES_LIST:
        split_info_by_layer[layer_name] = {}
        for movie in valid_movies:
            chunks_X, chunks_y, train_idx, test_idx = load_movie_chunks(
                movie         = movie,
                fmri_chunks   = fmri_stack[movie],
                features_dir  = args.features_dir,
                layer_name    = layer_name,
                test_ratio    = args.test_ratio,
            )
            split_info_by_layer[layer_name][movie] = {
                'chunks_X' : chunks_X,
                'chunks_y' : chunks_y,
                'train_idx': train_idx,
                'test_idx' : test_idx,
            }
            # train/test split is the same regardless of layer — save once
            if movie not in saved_split:
                saved_split[movie] = {
                    'train_idx': train_idx,
                    'test_idx' : test_idx,
                    'n_chunks' : len(chunks_X),
                }

        print(f"  {layer_name}: loaded")

    # Save split info per movie
    np.savez(os.path.join(subj_out, 'split_info.npz'), **{
        f"{movie}__{k}": np.array(v)
        for movie, info in saved_split.items()
        for k, v in info.items()
    })

    # ── Fit one ridge per layer, across all movies ────────────────────────────
    print(f"\nFitting ridge regression (one model across all movies) ...")

    all_r2_per_movie = {movie: {} for movie in valid_movies}
    all_r2_combined  = {}

    for layer_name in FEATURES_LIST:
        r2_per_movie, r2_combined = fit_and_evaluate_layer(
            layer_name   = layer_name,
            movies       = valid_movies,
            fmri_stack   = fmri_stack,
            features_dir = args.features_dir,
            split_info   = split_info_by_layer[layer_name],
            test_ratio   = args.test_ratio,
        )

        all_r2_combined[layer_name] = r2_combined

        # Save per-movie R² arrays
        for movie, r2 in r2_per_movie.items():
            all_r2_per_movie[movie][layer_name] = r2
            np.save(
                os.path.join(subj_out, f"{layer_name}_r2_{movie}.npy"), r2
            )

        # Save combined R²
        np.save(
            os.path.join(subj_out, f"{layer_name}_r2_combined.npy"), r2_combined
        )

    # ── Summary plot ─────────────────────────────────────────────────────────
    plot_summary(all_r2_per_movie, all_r2_combined, valid_movies, subj_out)

    # ── Save everything in one npz ────────────────────────────────────────────
    save_dict = {}
    for movie, res in all_r2_per_movie.items():
        for layer_name, r2 in res.items():
            save_dict[f"{movie}__{layer_name}"] = r2
    for layer_name, r2 in all_r2_combined.items():
        save_dict[f"combined__{layer_name}"] = r2

    np.savez_compressed(os.path.join(subj_out, 'all_r2_scores.npz'), **save_dict)
    print(f"\nAll R² scores → {os.path.join(subj_out, 'all_r2_scores.npz')}")
    print("Keys: <movie>__<layer>  and  combined__<layer>")
    print("\nDone.")


if __name__ == '__main__':
    main()
