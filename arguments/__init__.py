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

import sys
import os

from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.sh_degree = 0
        self._resolution = -1
        self.data_device = "cuda"
        self.eval = False  # This will keep some views for evaluation
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Common
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.iterations
        self.scaling_lr = 0.005
        self.feature_lr = 0.025
        self.opacity_lr = 0.05
        self.mask_lr = 0.01
        self.lambda_dssim = 0.2
        self.lambda_orientation = 100.0
        self.lambda_mask = 0.01
        self.pval = 0.05
        self.bidirectional_eval = True  # True -> bidirectional angle difference
        # GS specific
        self.rotation_lr = 0.001
        # Hair-GS specific
        self.lambda_smooth = 0.005
        self.lambda_magnet = 0.0  # Disabled
        self.bidirectional_merge = False  # False -> only merge from strand tip
        self.num_points_strand = 80
        self.merge_interval = 100
        self.merge_dist_th_init = 2e-3
        self.merge_dist_th_final = 4e-3
        self.merge_angle_th_init = 20
        self.merge_angle_th_final = 40
        self.growth_interval = 100000
        self.growth_averaging_points = 3
        # Densification
        self.percent_dense = 0.01
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = self.iterations * 0.9
        self.densification_interval = 100
        self.prune_max_radii_2d = 1000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")


class GeneralParams(ParamGroup):
    def __init__(self, parser):
        self.quiet = False
        self.logger = "tensorboard"  # options: wandb, tensorboard, None
        self.ip = "127.0.0.1"
        self.port = 6009
        self.vis2d = False
        self.update_vis2d_frequency = 30000
        self.vis3d = False
        self.save_frequency = 5000
        self.eval_frequency = 30000
        super().__init__(parser, "General Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
