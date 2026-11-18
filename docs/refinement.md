# Scene training recipes

The commands in [Training](../README.md#4-training) run each scene's complete recipe through `train_synthetic.py` or `train_real.py`. Use the [training environment](../README.md#1-create-environment) and put the scene data in `data_hdr/`.

Each scene YAML sets `refinement.enabled` and points to its recipe in [config/refinement](../config/refinement). The recipe records the initialization, seeds, learning rates, loss weights, training stages, reference test metrics and SHA-256 hashes. Set `refinement.enabled: false` for base training.

All default scene configs initialize the Gaussians from the SfM point clouds in the released dataset and train on the scene images. Refinement starts from the checkpoint saved after 30,000 base iterations. The defaults for Bathroom and Sofa select `refinement/bathroom-from-scratch.yaml` and `refinement/sofa-from-scratch.yaml`.

To reproduce the published Bathroom and Sofa checkpoint pairs, set `refinement.plan` in the scene YAML to `refinement/bathroom.yaml` or `refinement/sofa.yaml`. These recipes refine their original published weights. The training script fetches the files from the pinned Model revision, checks their hashes and caches them in the PyTorch Hub cache.

To refine an existing base checkpoint or use a local copy of a published initialization, set `load_path` in the scene's `refinement` block. The path points to a directory containing `point_cloud.ply` and `tone_mapper.pth`.

```yaml
refinement:
  enabled: true
  plan: refinement/bear.yaml
  load_path: output/base/bear/point_cloud/iteration_30000
```

Run the same scene command from the repository root. The `plan` path is relative to the scene YAML, and `load_path` is relative to the working directory. To resume a later refinement stage, set `start_stage` to its one-based index and set `load_path` to the preceding stage's `weights/` directory.

Each stage saves a model pair in `refinement/stage_XX/weights/` under the training output directory. The final path is printed and recorded in `refinement/run.json`. With `--eval`, `refinement/benchmark.json` reports PSNR, SSIM and Alex LPIPS on the published test split. Synthetic HDR metrics use the per-image normalization and µ-law mapping in the synthetic evaluator.

Each recipe records metrics from reloading its final model. Bathroom and Sofa have separate measurements for their default training recipes and published checkpoints. Use the recorded initialization for the closest comparison. Retraining the base model or switching CUDA environments can change the final metrics. Use [Evaluation](../README.md#evaluation) to measure the released weights directly.
