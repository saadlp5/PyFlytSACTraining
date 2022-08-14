import argparse
import os
import sys
import time
import warnings

import numpy as np
import wandb
import yaml

from utils.helpers import get_device


def shutdown_handler(*_):
    print("ctrl-c invoked")
    exit(0)


def arg_parser():
    parser = argparse.ArgumentParser(description="IDK yet.")

    parser.add_argument(
        "--wandb",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Log to W and B.",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Activate debug mode.",
    )

    parser.add_argument(
        "--display",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Set to display mode.",
    )

    parser.add_argument(
        "--train",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Train the network.",
    )

    parser.add_argument(
        "--evaluate",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Evaluate the network.",
    )

    parser.add_argument(
        "--shutdown",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Shutdown after training.",
    )

    parser.add_argument(
        "--wandb_name",
        type=str,
        nargs="?",
        const=False,
        default="",
        help="Name of the run in wandb.",
    )

    parser.add_argument(
        "--notes",
        type=str,
        nargs="?",
        const=False,
        default="",
        help="Description of the run in wandb.",
    )

    parser.add_argument(
        "--id",
        type=str,
        nargs="?",
        const=False,
        default="",
        help="ID of the run in wandb.",
    )

    parser.add_argument(
        "--net_version",
        type=str,
        nargs="?",
        const=False,
        default="",
        help="Network weights number.",
    )

    return parser.parse_args()


def parse_set():
    """
    Fundamentally there's two things we need to deal with:
        - settings.yaml
        - args

    Depending on the output of args, we also may need to do it through wandb
    """

    # parse arguments
    args = arg_parser()

    # parse settings
    with open(os.path.join(os.path.dirname(__file__), "./settings.yaml")) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # format settings a bit
    settings["device"] = get_device()
    settings["buffer_size"] = (
        settings["buffer_size_debug"] if args.debug else settings["buffer_size"]
    )
    # when formatting net_version, assert that only either args or settings
    # file have a value, not both
    if settings["net_version"] == "" and args.net_version == "":
        settings["net_version"] = str(np.random.randint(999999))
    elif settings["net_version"] != "" and args.net_version == "":
        settings["net_version"] = settings["net_version"]
    elif settings["net_version"] == "" and args.net_version != "":
        settings["net_version"] = args.net_version
    else:
        raise AssertionError(
            "net_version cannot be set in both settings.yaml and in args."
        )
    if not args.debug:
        args.net_version = settings["net_version"]
    else:
        args.net_version = "debug"

    # merge args and settings
    settings = dict({**settings, **vars(args)})

    # set depending on whether wandb is enabled
    set = None
    if args.wandb and args.train:
        wandb.init(
            project=settings["wandb_project"],
            entity="availab",
            config=settings,
            name=args.wandb_name + ", v=" + settings["net_version"]
            if args.wandb_name != ""
            else settings["net_version"],
            notes=args.notes,
            id=args.id if args.id != "" else None,
        )

        # also save the code if wandb
        wandb.run.log_code(".", exclude_fn=lambda path: "venv" in path)

        # set to be consistent with wandb config
        set = wandb.config
    else:
        # otherwise just merge settings with args
        set = argparse.Namespace(**settings)

    return set


def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return (
        getattr(sys, "base_prefix", None)
        or getattr(sys, "real_prefix", None)
        or sys.prefix
    )


def check_venv():
    if get_base_prefix_compat() == sys.prefix:
        time.sleep(10)
        warnings.warn("Not in venv.")
