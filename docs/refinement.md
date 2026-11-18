# Reproduce the selected HDR-GS checkpoints

The twelve YAML files in [config/refinement](../config/refinement) specify the initialization, learning rates, loss weights, seeds and training stages for the selected checkpoints. They record the test metrics and SHA-256 hashes of the weights used for the released visualizations. The same training entry point uses HDR-GS's existing renderer, camera loader and PLY/tone-mapper serialization for every scene.

Use the HDR-GS environment from [Create Environment](../README.md#1-create-environment) and place the original scene data under data_hdr as described in the README.

Bathroom and Sofa start from their original published checkpoint pairs, pinned in each YAML's initialization section. That revision preserves the exact inputs to refinement, including when the current Model repository points to the improved versions. For the other ten scenes, use the matching final point_cloud/iteration_30000 directory produced by the original training script. The initialization section records the script, original YAML, arguments, revision and checksum of that starting checkpoint.

For example, after obtaining the original Sofa weights:

    python train_refinement.py --config config/refinement/sofa.yaml \
      --load_path pretrained_weights/sofa_original \
      --output output/refined/sofa --gpu_id 0 --evaluate

Select a different scene by changing the YAML, the matching initialization directory and the output directory. Each stage saves point_cloud.ply and tone_mapper.pth under stage_XX/weights. The final benchmark.json reports PSNR, SSIM and Alex LPIPS on the published test split. The reported HDR metrics use the synthetic evaluator's normalization and mu-law mapping.

The expected values in the YAML come from reloading the actual selected checkpoints. Exact comparisons should use the recorded initial weights. Recreating the initial training run or switching CUDA environments can introduce small numerical differences. Test views are used for evaluation. Refinement trains on the released training views and exposure groups.

Source code revision and reproduction inputs are recorded alongside the model artifacts. Stage settings, starting hashes and evaluation outputs are also preserved in each local run directory.
