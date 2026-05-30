"""
ridge_regression_subject.py

For a given subject, run ridge regression to predict fMRI responses from
TRIBE v2 intermediate features (all layers), across all movies.

For each movie:
  - Load pre-extracted features (.npz) and model predictions
  - Align features to predictions (chunk-aware, handles padding)
  - Resample features to match fMRI TR
  - Z-score features and fMRI per chunk
  - Split chunks: last 20% → test, rest → train
  - Fit RidgeCV on train, evaluate R² on test
  - Save per-layer R² arrays to output directory

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
  <output_dir>/sub-<subject>/<movie>/<layer_name>_r2.npy   — R² per vertex
  <output_dir>/sub-<subject>/<movie>/split_info.npz        — train/test chunk indices
"""

import argparse
import os
import numpy as np
import torch
from torch import nn
from scipy.signal import resample
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
# ── local utils (same repo structure as notebooks) ───────────────────────────
from utils.fmri_utils import load_fmri

# ─────────────────────────────────────────────────────────────────────────────
# Layer list (FFN outputs only — one per transformer depth + projectors + head)
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
# Alignment utilities (identical to notebook)
# ─────────────────────────────────────────────────────────────────────────────
def get_chunk_boundaries(segments):
    """
    Detect chunk boundaries from the Segment objects returned by model.predict().
    .start resets to 0 at the beginning of each independent recording run.

    Returns
    -------
    boundaries : np.ndarray  shape (n_chunks+1,)
        Indices into the flat segments/predictions array.
    starts : np.ndarray  shape (n_total_trs,)
        Integer .start value for every TR.
    """
    starts = np.array([np.round(seg.start).astype(int) for seg in segments])
    boundaries = np.where(np.diff(starts) < 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(segments)]])
    return boundaries, starts


def align_features_to_predictions_chunked(feat_array, segments, predictions,
                                           pool_to=100):
    """
    Pool + align hook-captured features to model.predict() output,
    preserving per-chunk structure.

    Parameters
    ----------
    feat_array : np.ndarray
        Shape (n_batches, C, 200)  — already in (batches, channels, time) order,
        OR (n_batches, 200, C)     — will be transposed automatically.
    segments : array-like of Segment
        From model.predict(), .start resets per chunk.
    predictions : np.ndarray  shape (n_total_trs, n_vertices)
        Flat predictions output from model.predict().
    pool_to : int
        Target temporal resolution (default 100 = 1 Hz).

    Returns
    -------
    chunks_feat : list of np.ndarray  (chunk_len, C)
    chunks_pred : list of np.ndarray  (chunk_len, n_vertices)
    chunk_boundaries : np.ndarray
    """
    pooler = nn.AdaptiveAvgPool1d(pool_to)

    fa = feat_array.copy()
    if fa.shape[2] == 200:
        pass                         # (n_batches, C, 200) — correct
    elif fa.shape[1] == 200:
        fa = fa.transpose(0, 2, 1)  # (n_batches, 200, C) → (n_batches, C, 200)
    else:
        raise ValueError(
            f"Cannot locate time dimension (size 200) in shape {feat_array.shape}"
        )

    # Pool: (n_batches, C, 200) → (n_batches, C, pool_to)
    pooled = pooler(torch.tensor(fa.astype(np.float32))).numpy()

    # Flatten: (n_batches, C, pool_to) → (n_batches * pool_to, C)
    pooled_flat = pooled.transpose(0, 2, 1).reshape(-1, pooled.shape[1])

    chunk_boundaries, starts = get_chunk_boundaries(segments)
    n_chunks = len(chunk_boundaries) - 1

    chunks_feat = []
    chunks_pred = []
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-movie pipeline
# ─────────────────────────────────────────────────────────────────────────────
def zscore(arr):
    """Z-score columns (axis=0), avoid division by zero."""
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0,  keepdims=True)
    std[std < 1e-8] = 1.0
    return (arr - mean) / std


def process_movie(movie, subject, fmri_stack, features_dir, output_dir,
                  test_ratio=0.2, tr=1.49):
    """
    Full pipeline for one movie × one subject.

    Returns dict:  layer_name → r2_array  (shape: n_vertices)
    """
    print(f"\n{'='*60}")
    print(f"  Movie: {movie}  |  Subject: {subject}")
    print(f"{'='*60}")

    # ── Load pre-extracted features + predictions ─────────────────────────────
    feat_path = os.path.join(features_dir, f"{movie}_allfeatures.npz")
    pred_path = os.path.join(features_dir, f"{movie}_predictions.npz")

    if not os.path.exists(feat_path):
        print(f"  [SKIP] Features not found: {feat_path}")
        return None
    if not os.path.exists(pred_path):
        print(f"  [SKIP] Predictions not found: {pred_path}")
        return None

    npz_feat = np.load(feat_path)
    npz_pred = np.load(pred_path, allow_pickle=True)
    predictions = npz_pred['preds']    # (n_total_trs, 20484)
    segments    = npz_pred['segments'] # (n_total_trs,) Segment objects

    # ── fMRI chunks for this movie ────────────────────────────────────────────
    fmri_chunks = fmri_stack.get(movie, [])
    if len(fmri_chunks) == 0:
        print(f"  [SKIP] No fMRI data found for movie '{movie}'")
        return None

    # ── Detect chunks & compute train/test split (by chunk index) ─────────────
    # Use the predictor layer just to get the chunk count
    dummy_feat = npz_feat[FEATURES_LIST[-1]]
    _, _, chunk_boundaries = align_features_to_predictions_chunked(
        dummy_feat, segments, predictions
    )
    n_chunks = len(chunk_boundaries) - 1

    if n_chunks != len(fmri_chunks):
        print(f"  [WARN] Chunk count mismatch: "
              f"features={n_chunks}, fMRI={len(fmri_chunks)}. "
              f"Using min({n_chunks}, {len(fmri_chunks)}).")
        n_chunks = min(n_chunks, len(fmri_chunks))

    # Last ceil(test_ratio * n_chunks) chunks → test
    n_test  = max(1, int(np.ceil(test_ratio * n_chunks)))
    n_train = n_chunks - n_test
    train_idx = list(range(n_train))
    test_idx  = list(range(n_train, n_chunks))

    print(f"  {n_chunks} chunks total  →  train: {train_idx}  |  test: {test_idx}")

    # Save split info
    movie_out = os.path.join(output_dir, movie)
    os.makedirs(movie_out, exist_ok=True)
    np.savez(os.path.join(movie_out, 'split_info.npz'),
             n_chunks=n_chunks, train_idx=train_idx, test_idx=test_idx)

    # ── Loop over layers ──────────────────────────────────────────────────────
    results = {}

    for layer_name in FEATURES_LIST:
        if layer_name not in npz_feat.files:
            print(f"  [SKIP layer] {layer_name} not in .npz")
            continue

        feat_array = npz_feat[layer_name]

        # Align to predictions (chunk-aware)
        chunks_feat, _, _ = align_features_to_predictions_chunked(
            feat_array, segments, predictions
        )
        chunks_feat = chunks_feat[:n_chunks]

        # Z-score + resample each chunk to match fMRI TR
        chunks_processed = []
        for c in range(n_chunks):
            cf       = zscore(chunks_feat[c])            # (T_feat, C)
            fmri_ts  = fmri_chunks[c]                    # (T_fmri, n_vertices)
            T_fmri   = fmri_ts.shape[0]
            cf_resampled = resample(cf, T_fmri, axis=0)  # match fMRI length
            chunks_processed.append(cf_resampled)

        # Build train / test matrices
        X_train = np.concatenate([chunks_processed[c] for c in train_idx], axis=0)
        y_train = np.concatenate(
            [zscore(fmri_chunks[c]) for c in train_idx], axis=0
        )

        X_test  = np.concatenate([chunks_processed[c] for c in test_idx],  axis=0)
        y_test  = np.concatenate(
            [zscore(fmri_chunks[c]) for c in test_idx],  axis=0
        )

        print(f"  [{layer_name:30s}]  "
              f"X_train={X_train.shape}  X_test={X_test.shape}  "
              f"y_train={y_train.shape}  y_test={y_test.shape}")

        # Fit Ridge
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)

        r2 = r2_score(y_test, y_pred, multioutput='raw_values')  # (n_vertices,)
        results[layer_name] = r2

        # Save immediately (so partial results survive crashes)
        out_path = os.path.join(movie_out, f"{layer_name}_r2.npy")
        np.save(out_path, r2)

        print(f"    R² mean={r2.mean():.4f}  median={np.median(r2):.4f}  "
              f"max={r2.max():.4f}  →  saved {out_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary plot: R² by layer depth, per movie
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary(all_results, output_dir):
    """
    all_results : dict  movie → dict  layer_name → r2_array
    """
    movies = [m for m in all_results if all_results[m]]
    if not movies:
        return

    fig, axes = plt.subplots(1, len(movies),
                              figsize=(5 * len(movies), 5),
                              sharey=True, squeeze=False)

    for col, movie in enumerate(movies):
        res   = all_results[movie]
        layers = [l for l in FEATURES_LIST if l in res]
        means  = [res[l].mean() for l in layers]

        ax = axes[0, col]
        ax.barh(layers, means, color='steelblue')
        ax.invert_yaxis()
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel("Mean R² (test chunks)")
        ax.set_title(movie)

    plt.suptitle("Ridge R² by encoder depth", fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, 'summary_r2_by_layer.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSummary plot saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Ridge regression from TRIBE v2 features to fMRI, per subject.'
    )
    parser.add_argument('--subject',      type=int,   required=True,
                        help='Subject index (integer, e.g. 1)')
    parser.add_argument('--basedir',      type=str,   required=True,
                        help='Root of Algonauts2025 download (passed to load_fmri)')
    parser.add_argument('--features_dir', type=str,   required=True,
                        help='Directory containing <movie>_allfeatures.npz and '
                             '<movie>_predictions.npz')
    parser.add_argument('--output_dir',   type=str,   default='./ridge_results',
                        help='Where to write R² arrays and plots')
    parser.add_argument('--movies',       type=str,   nargs='+',
                        default=['bourne', 'wolf', 'figures', 'life'],
                        help='Movie names to process')
    parser.add_argument('--test_ratio',   type=float, default=0.2,
                        help='Fraction of chunks (last N) held out for test')
    parser.add_argument('--tr',           type=float, default=1.49,
                        help='fMRI repetition time in seconds')
    args = parser.parse_args()

    # ── Output directory ──────────────────────────────────────────────────────
    subj_out = os.path.join(args.output_dir, f"sub-{args.subject:02d}")
    os.makedirs(subj_out, exist_ok=True)

    # ── Load fMRI for this subject (all movies at once) ───────────────────────
    print(f"\nLoading fMRI for subject {args.subject} from {args.basedir} ...")
    fmri = load_fmri(args.basedir, args.subject, average=False)

    fmri_stack = {}
    for movie in args.movies:
        keys = sorted(k for k in fmri.keys() if movie in k)
        fmri_stack[movie] = [fmri[k] for k in keys]
        print(f"  {movie}: {len(keys)} chunks  "
              f"{[fmri[k].shape for k in keys]}")

    # ── Process each movie ────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend for script use
    import matplotlib.pyplot as plt

    all_results = {}
    for movie in args.movies:
        movie_out    = os.path.join(subj_out, movie)
        all_results[movie] = process_movie(
            movie       = movie,
            subject     = args.subject,
            fmri_stack  = fmri_stack,
            features_dir= args.features_dir,
            output_dir  = subj_out,
            test_ratio  = args.test_ratio,
            tr          = args.tr,
        )

    # ── Summary plot across layers ────────────────────────────────────────────
    plot_summary(all_results, subj_out)

    # ── Save all R² results in one npz for easy downstream use ───────────────
    save_dict = {}
    for movie, res in all_results.items():
        if res is None:
            continue
        for layer_name, r2 in res.items():
            save_dict[f"{movie}__{layer_name}"] = r2
    np.savez_compressed(
        os.path.join(subj_out, 'all_r2_scores.npz'),
        **save_dict
    )
    print(f"\nAll R² scores saved → {os.path.join(subj_out, 'all_r2_scores.npz')}")
    print("Keys follow the pattern:  <movie>__<layer_name>")
    print("\nDone.")


if __name__ == '__main__':
    main()
