from caffe.model_libs import *


class Params:

    def __init__(self, resize_height, resize_width, base_lr, iter_size, solver_mode, device_id, test_iter, num_classes,
                 share_location, background_label_id, output_result_dir, label_map_file, num_test_image, code_type,
                 loc_weight, train_on_diff_gt, neg_pos_ratio, normalization_mode):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.base_lr = base_lr
        self.iter_size = iter_size
        self.solver_mode = solver_mode
        self.device_id = device_id
        self.test_iter = test_iter
        self.num_classes = num_classes
        self.share_location = share_location
        self.background_label_id = background_label_id
        self.output_result_dir = output_result_dir
        self.label_map_file = label_map_file
        self.num_test_image = num_test_image
        self.code_type = code_type
        self.loc_weight = loc_weight
        self.train_on_diff_gt = train_on_diff_gt
        self.neg_pos_ratio = neg_pos_ratio
        self.normalization_mode = normalization_mode

    def params(self):
        train_transform_param = {
            'mirror': False,
            'mean_value': [104, 117, 123],
            'force_color': True,
            'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': self.resize_height,
                'width': self.resize_width,
                'interp_mode': [
                    P.Resize.LINEAR,
                    P.Resize.AREA,
                    P.Resize.NEAREST,
                    P.Resize.CUBIC,
                    P.Resize.LANCZOS4,
                ],
            },
            'emit_constraint': {
                'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }
        test_transform_param = {
            'mean_value': [104, 117, 123],
            'force_color': True,
            'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': self.resize_height,
                'width': self.resize_width,
                'interp_mode': [P.Resize.LINEAR],
            },
        }

        solver_param = {
            # Train parameters
            'base_lr': self.base_lr,
            'weight_decay': 0.0005,
            'lr_policy': "step",
            'stepsize': 60000,
            'gamma': 0,
            'momentum': 0.9,
            'iter_size': self.iter_size,
            'max_iter': 10000,
            'snapshot': 50,
            'display': 10,
            'average_loss': 10,
            'type': "SGD",
            'solver_mode': self.solver_mode,
            'device_id': self.device_id,
            'debug_info': False,
            'snapshot_after_train': True,
            # Test parameters
            'test_iter': [self.test_iter],
            'test_interval': 50,
            'eval_type': "detection",
            'ap_version': "11point",
            'test_initialization': False,
        }

        # parameters for generating detection output.
        det_out_param = {
            'num_classes': self.num_classes,
            'share_location': self.share_location,
            'background_label_id': self.background_label_id,
            'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
            'save_output_param': {
                'output_directory': self.output_result_dir,
                'output_name_prefix': "comp4_det_test_",
                'output_format': "VOC",
                'label_map_file': self.label_map_file,
                # 'name_size_file': name_size_file,
                'num_test_image': self.num_test_image,
            },
            'keep_top_k': 200,
            'confidence_threshold': 0.01,
            'code_type': self.code_type,
        }

        # parameters for evaluating detection results.
        det_eval_param = {
            'num_classes': self.num_classes,
            'background_label_id': self.background_label_id,
            'overlap_threshold': 0.5,
            'evaluate_difficult_gt': False,
            # 'name_size_file': name_size_file,
        }
        multibox_loss_param = {
            'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
            'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
            'loc_weight': self.loc_weight,
            'num_classes': self.num_classes,
            'share_location': self.share_location,
            'match_type': P.MultiBoxLoss.PER_PREDICTION,
            'overlap_threshold': 0.5,
            'use_prior_for_matching': True,
            'background_label_id': self.background_label_id,
            'use_difficult_gt': self.train_on_diff_gt,
            'do_neg_mining': True,
            'neg_pos_ratio': self.neg_pos_ratio,
            'neg_overlap': 0.5,
            'code_type': self.code_type,
        }
        loss_param = {
            'normalization': self.normalization_mode,
        }

        return train_transform_param, test_transform_param, solver_param, det_eval_param, det_out_param, loss_param, multibox_loss_param
