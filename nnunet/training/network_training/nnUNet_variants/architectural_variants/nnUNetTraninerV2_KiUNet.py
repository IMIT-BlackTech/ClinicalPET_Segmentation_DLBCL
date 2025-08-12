from typing import Tuple

import numpy as np
import torch
from torch import nn
from nnunet.network_architecture.generic_modular_residual_UNet import FabiansUNet, get_default_network_config

from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper

from my_traning.generic_KiUNet_3_depth import generic_KiUNet, Upsample, ConvSampleNormNonlin, ConvNormNonlinSampleResiSkip


class nnUNetTraninerV2_KiUNet(nnUNetTrainerV2):
    def initialize_network(self):
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in") 
            """
            {'conv_op': torch.nn.modules.conv.Conv3d,
             'dropout_op': None,
             'norm_op': torch.nn.modules.instancenorm.InstanceNorm3d,
             'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
             'dropout_op_kwargs': {'p': 0, 'inplace': True},
             'conv_op_kwargs': {'stride': 1, 'dilation': 1, 'bias': True},
             'nonlin': torch.nn.modules.activation.LeakyReLU,
             'nonlin_kwargs': {'negative_slope': 0.01, 'inplace': True}}
            """
        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        stage_plans = self.plans['plans_per_stage'][self.stage]
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']
#         blocks_per_stage_encoder = stage_plans['num_blocks_encoder']
#         blocks_per_stage_decoder = stage_plans['num_blocks_decoder']
        
        pool_op_kernel_sizes = [[2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2],
                                [2, 2, 2]]

        conv_op_kernel_sizes = [[3, 3, 3],
                                [3, 3, 3],
                                [3, 3, 3],
                                [3, 3, 3]]
        
        self.num_classes = 2
        final_nonlin = lambda x:x
        self.network = generic_KiUNet(
                input_channels=2, base_num_features=16, num_classes=2, num_pool=len(pool_op_kernel_sizes), num_conv_per_stage=2,
                feat_map_mul_on_downscale=2, conv_op=torch.nn.Conv3d,
                norm_op=torch.nn.InstanceNorm3d, norm_op_kwargs={'eps': 1e-5, 'affine': True, 'momentum': 0.1},
                downsample_op=torch.nn.MaxPool3d, downsample_op_kwargs=None,
                upsample_op=Upsample, upsample_op_kwargs=None,
                nonlin=torch.nn.ReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True,
                final_nonlin=final_nonlin, weightInitializer=InitWeights_He(1e-2),
                pool_op_kernel_sizes=pool_op_kernel_sizes, conv_kernel_sizes=None,
                upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                max_num_features=None, basic_block=ConvSampleNormNonlin, crfb_block=ConvNormNonlinSampleResiSkip,
                seg_output_use_bias=False
            )


        if torch.cuda.is_available():
            self.network.cuda()
        # self.network.inference_apply_nonlin = softmax_helper
        self.network.inference_apply_nonlin = final_nonlin
        # print("init:", self.network.conv_op)
        # assert 1==0

    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        self.net_num_pool_op_kernel_sizes = [[2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2],
                                             [2, 2, 2]]

        super().setup_DA_params()
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z=0,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = nnUNetTrainer.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size, save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name,
                                     debug=debug, all_in_gpu=all_in_gpu,
                                     segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds)
        self.network.do_ds = ds
        return ret
    
    # predict case
    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
                                                                             mirror_axes=mirror_axes,
                                                                             use_sliding_window=use_sliding_window,
                                                                             step_size=step_size,
                                                                             use_gaussian=use_gaussian,
                                                                             pad_border_mode=pad_border_mode,
                                                                             pad_kwargs=pad_kwargs,
                                                                             all_in_gpu=all_in_gpu,
                                                                             verbose=verbose,
                                                                             mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = nnUNetTrainer.run_training(self)
        self.network.do_ds = ds
        return ret