"""Load training configuration before applying explicit command-line options."""
import yaml


def parse_training_args(parser, argv=None):
    preliminary, _ = parser.parse_known_args(argv)
    try:
        with open(preliminary.config, "r") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        parser.error("Cannot read configuration {}: {}".format(preliminary.config, error))
    if not isinstance(config, dict):
        parser.error("The configuration must contain a mapping of parameter names to values")
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.test_only and not args.load_path:
        parser.error("--test_only requires --load_path pointing to a saved scene model")
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    return args
