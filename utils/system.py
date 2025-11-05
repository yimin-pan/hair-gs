# The code is modified based on original 3DGS which is under the following license:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

from argparse import ArgumentParser, Namespace


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def search_for_max_interation(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def clean_output_path(args: ArgumentParser):
    if args.model_path:
        print("Cleaning output folder: {}".format(args.model_path))
        os.system("rm -rf {}".format(args.model_path))


def prepare_output_path(args: ArgumentParser):
    if not args.model_path:
        # use name of the source folder
        source_path = args.source_path
        elems = -2
        if source_path.endswith("/"):
            elems = -3
        source_folder = source_path.split("/")[elems:]
        args.model_path = os.path.join("./output/", source_folder[0], source_folder[1])
    # Create output folder
    print("Selected output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
