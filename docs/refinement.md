# Scene training recipes

The commands in [Training](../README.md#4-training) run each scene's complete recipe through `train_synthetic.py` or `train_real.py`. Use the [training environment](../README.md#1-create-environment) and put the scene data in `data_hdr/`.

Each scene YAML sets `refinement.enabled` and points to its recipe in [config/refinement](../config/refinement). The recipe records the initialization, seeds, learning rates, loss weights, training stages, reference test metrics and SHA-256 hashes. Set `refinement.enabled: false` for base training.

The release recipes for Bathroom and Sofa start from their original published checkpoint pairs. The training script caches the files from the pinned Model revision in the PyTorch Hub cache and verifies their hashes. The other ten recipes start from the checkpoint saved by 30,000 iterations of base training.

To refine an existing base checkpoint or use a local copy of a published initialization, set `load_path` in the scene's `refinement` block. The path points to a directory containing `point_cloud.ply` and `tone_mapper.pth`.

```yaml
refinement:
  enabled: true
  plan: refinement/bear.yaml
  load_path: output/base/bear/point_cloud/iteration_30000
```

Run the same scene command from the repository root. The `plan` path is relative to the scene YAML, and `load_path` is relative to the working directory. To resume a later refinement stage, set `start_stage` to its one-based index and set `load_path` to the preceding stage's `weights/` directory.

Each stage saves a model pair in `refinement/stage_XX/weights/` under the training output directory. The final path is printed and recorded in `refinement/run.json`. With `--eval`, `refinement/benchmark.json` reports PSNR, SSIM and Alex LPIPS on the published test split. Synthetic HDR metrics use the per-image normalization and µ-law mapping in the synthetic evaluator.

The reference metrics in each recipe come from reloading the selected checkpoints. Use the recorded initialization for the closest comparison. Retraining the base model or switching CUDA environments can change the final metrics. Use [Evaluation](../README.md#evaluation) to measure the released weights directly.
