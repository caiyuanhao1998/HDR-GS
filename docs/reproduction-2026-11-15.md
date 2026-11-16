# Reproduction on Linux — 2026-11-15

The released data, models and README commands were exercised in a fresh checkout. Full training uses the twelve released YAML configurations at their default 30,000 iterations. Each saved final model was then loaded with `--test_only` and evaluated again.

## Environment

Both independent Conda installations built the original CUDA rasterizer and kNN submodules and passed all nine regression tests. The second installation also rendered the released bathroom model and evaluated all 85 test images, with exactly the same metrics as the first installation.

- NVIDIA RTX A5000 (24 GB), CUDA toolkit 11.8, GCC 11.4.
- Python 3.7.12, PyTorch 1.12.1 (CUDA 11.6), TorchVision 0.13.1.
- NumPy 1.21.6, SciPy 1.7.3, ImageIO 2.31.1, OpenCV 4.8.1.
- Both Hugging Face repositories were downloaded using Git LFS and passed `git lfs fsck`.

## Runtime fixes

Validated changes: [setup, training and checkpoint fixes](https://github.com/caiyuanhao1998/HDR-GS/commit/821ea47) and [direct checkpoint evaluation](https://github.com/caiyuanhao1998/HDR-GS/commit/fa9a975).

- The batch scripts now call the existing training entry points and stop if a scene fails.
- The environment includes the imaging/YAML/scientific dependencies and an MKL version compatible with the released PyTorch binary. CUDA extensions are built after PyTorch and a compatible host compiler are available.
- Explicit CLI options override YAML defaults, and the actual final iteration is saved.
- New resumable checkpoints preserve the three learned RGB tone mappers as well as Gaussian and optimizer state. Legacy checkpoints require the matching standalone model weights.
- Evaluation reuses LPIPS networks. PSNR, SSIM and LPIPS for both released pretrained models are identical before and after this change.

## Released pretrained models

Each model was evaluated on all 85 test images, and its spiral video was rendered.

| Model | LDR-OE PSNR | LDR-NE PSNR | HDR PSNR | Metric parity |
| --- | ---: | ---: | ---: | --- |
| bathroom | 41.3119 | 40.5054 | 35.8559 | Exact |
| sofa_2 | 41.4113 | 25.6474 | 42.3113 | Exact |

## Full training and saved-model reload

OE denotes observed exposures; NE denotes held-out exposures. The released real scenes have no HDR reference images, so HDR PSNR is reported for synthetic scenes.

| Scene | Iterations | LDR-OE PSNR | LDR-NE PSNR | HDR PSNR | Gaussian count | Reload |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bathroom | 30,000 | 41.1444 | 40.4460 | 35.0143 | 307,923 | Passed |
| bear | 30,000 | 43.1632 | 42.4514 | 43.8283 | 146,702 | Passed |
| chair | 30,000 | 35.7308 | 34.8508 | 37.6682 | 353,021 | Passed |
| desk | 30,000 | 40.1657 | 38.3049 | 44.0332 | 396,693 | Passed |
| diningroom | 30,000 | 43.8945 | 42.2196 | 42.1153 | 317,783 | Passed |
| dog | 30,000 | 42.0381 | 39.5420 | 36.8066 | 295,480 | Passed |
| sofa | 30,000 | 41.5399 | 25.7095 | 42.3783 | 384,790 | Passed |
| sponza | 30,000 | 39.5451 | 26.1701 | 27.7488 | 409,362 | Passed |
| box | 30,000 | 36.5903 | 30.1250 | — | 1,776,774 | Passed |
| computer | 30,000 | 35.7317 | 32.1288 | — | 2,037,652 | Passed |
| flower | 30,000 | 33.0431 | 29.6885 | — | 1,676,278 | Passed |
| luckycat | 30,000 | 36.4952 | 34.4433 | — | 2,669,902 | Passed |

All PLY fields and tone-mapper weights are finite. All 24 final-training/reload videos decode successfully, contain 120 frames, and run at 30 fps. The largest absolute difference between a final training metric and its reloaded-model metric is `0`.

## Checkpoint continuation

A checkpoint from synthetic bathroom training was saved at iteration 1,000, restored in a separate process, and trained through iteration 1,010. Evaluation, video rendering, final model export and a new resumable checkpoint all completed. Regression tests also verify that the next optimizer step matches after a complete Gaussian/tone-mapper checkpoint round trip.

Direct `--test_only --start_checkpoint` evaluation was also exercised on all 85 test images at reduced resolution.

Complete PSNR/SSIM/LPIPS, model sizes, video metadata and reload differences are recorded in [the machine-readable results](reproduction-2026-11-15.json).
