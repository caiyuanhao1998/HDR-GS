# Rendering and camera paths

Render an LDR video with the learned tone mapper. Without `--brightness`, brightness increases on the forward leg and decreases on the backward leg, using the scene-specific project-page limits:

```shell
python render_ldr.py --scene bathroom --model_path pretrained_weights/synthetic/bathroom --output output/videos/bathroom --gpu_id 0
```

Add `--brightness 0.35` for fixed mean luminance instead of a gradient. Brightness is a target in `(0, 1)`, achieved by solving the learned tone mapper's exposure, not by multiplying rendered pixels.

Render the HDR video along the same camera path:

```shell
python render_hdr.py --scene bathroom --model_path pretrained_weights/synthetic/bathroom --output output/videos/bathroom --gpu_id 0
```

The output contains `ldr.mp4`, `hdr.mp4`, corresponding posters, interactive LDR/HDR Gaussian previews, `tonemap.json`, camera poses and render metadata. Open the generated `viewer/index.html?scene=render&mode=ldr` or `mode=hdr` through a local HTTP server. The HDR video uses OpenCV/NumPy extended Reinhard tone mapping and an sRGB display transform, with one calibration for the whole sequence. Photomatix and Luminance HDR are not required. The MP4 is an SDR visualization for ordinary displays; `frames_hdr/*.exr` preserves original float32 HDR radiance. Add `--save_exr` to retain every HDR frame; otherwise three representative EXRs are saved.

Both commands default to 8 seconds at 30 fps. They retain the project page's spatial camera paths and traverse them twice as fast as the original 16-second clips; smooth turn transitions scale with duration. Spirals keep one winding direction through smooth forward/backward transitions. Use `--scene` to choose `bathroom`, `bear`, `chair`, `desk`, `diningroom`, `dog`, `sofa`, `sponza`, `flower`, `computer`, `box`, or `luckycat`; pass its checkpoint with `--model_path`. Download the pretrained checkpoints for all 12 scenes using [Section 3 of the README](../README.md#3-testing). `--model_path` accepts a PLY, checkpoint directory, or training run. `--tone_mapper` selects a separate matching tone mapper. `--trajectory path.json` loads a custom path; `--duration`, `--fps`, `--max_size`, and `--preview_only` control rendering. Explicit trajectory JSON keeps its own timing unless `--duration` or `--fps` is supplied. HDR display controls include `--exposure` in EV, `--key`, `--white_percentile` and `--saturation`.

### Local Gradio demo

Install both environments using [Section 1 of the README](../README.md#1-create-environment), then launch from the activated `hdr_gs` environment:

```shell
conda run --no-capture-output -n hdr_gs_demo python demo/app.py --renderer-python "$CONDA_PREFIX/bin/python" --models-root pretrained_weights --gpu-id 0 --port 7876
```

Open `http://127.0.0.1:7876` on the machine running the demo. Point `--models-root` to the directory containing the downloaded `synthetic/` and `real/` folders. Keep each scene's `point_cloud.ply` and `tone_mapper.pth` together under `synthetic/<scene>/` or `real/<scene>/`. All 12 scenes are selectable. Alternatively, upload an HDR-GS PLY and its matching tone mapper. Preview a spiral, straight, horizontal-sway, vertical-sway or figure-eight path; adjust the forward distance and lateral amplitudes; then render both LDR and HDR videos. The demo defaults to 8 seconds at 30 fps. Camera axes show X right, Z forward and −Y up relative to the starting camera. Viewer controls sit below each point cloud. Clicking Render clears previous results and shows a centered progress bar with the stage, frame count and elapsed time. The demo shows separate interactive Gaussian viewers and offers videos, visualization splats, original PLY/tone mapper, EXR samples and camera JSON downloads.

Custom trajectory JSON uses OpenCV camera-to-world matrices (`+X right, +Y down, +Z forward`), positive `camera_intrinsics` (`width`, `height`, `fx`, `fy`), `duration_seconds`, `fps`, and ordered `keyframes` containing `time_seconds` and `camera_to_world`. The first keyframe is at zero and the last at `duration_seconds`, with matching start/end position and orientation. Paths must move forward first and then return; backward-first paths are rejected. The demo includes an editable JSON example and exports the complete sampled path. Its closed final keyframe is omitted from the video to avoid a duplicate loop frame.


### Reusing Gaussian previews

The trained Gaussian geometry is loaded for inference; rendering does not retrain or reconstruct the scene. Browser point clouds, videos and camera data are organized in the [HF Dataset visualizations directory](https://huggingface.co/datasets/CaiYuanhao/HDR-GS/tree/main/visualizations).

Point-cloud exports use a content-addressed `.viewer-cache` beside the output directories; `--viewer_cache` selects another location. The HDR geometry/radiance export is reused for the same source model, reference camera and point budget. LDR previews share that geometry and only recompute learned tone-mapper colors when brightness or the tone mapper changes. Repeating the same brightness reuses those colors too. Changing video duration, frame rate or the rest of the camera path does not invalidate these preview files when the reference camera is unchanged. HDR display calibration/exposure metadata is updated independently of cached Gaussian bytes.
