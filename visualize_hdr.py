"""Convert EXR/HDR images to viewable PNGs without an external GUI."""
import argparse
import json
from pathlib import Path
from utils.hdr_visualization import fit_reinhard, load_settings, read_hdr, save_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='EXR/HDR file or directory')
    parser.add_argument('--output', required=True, type=Path, help='PNG path or output directory')
    parser.add_argument('--settings', type=Path, help='Reuse a saved calibration JSON')
    parser.add_argument('--exposure', type=float, default=0.0, help='Exposure offset in EV')
    parser.add_argument('--key', type=float, default=0.18)
    parser.add_argument('--white_percentile', type=float, default=99.95)
    parser.add_argument('--saturation', type=float, default=1.0)
    args = parser.parse_args()
    directory = args.input.is_dir()
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in ['.exr', '.hdr']) if directory else [args.input]
    if not paths or not all(p.is_file() for p in paths):
        parser.error('No HDR images found')
    settings = load_settings(args.settings) if args.settings else fit_reinhard(
        (read_hdr(p) for p in paths), key=args.key,
        white_percentile=args.white_percentile, saturation=args.saturation)
    for path in paths:
        destination = args.output / (path.stem + '.png') if directory else args.output
        save_preview(destination, read_hdr(path), settings, args.exposure)
        print(destination)
    settings_path = args.output / 'tonemap.json' if directory else args.output.with_suffix('.tonemap.json')
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + '\n')


if __name__ == '__main__':
    main()
