from __future__ import print_function
import sys

sys.path.insert(0, '/home/sweekar/SDN_main/caffe/python/')
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

from top_layers import TopLayer
from sampler import Sampler
from parameters import Params
from directories import Dir

import math
import os
import shutil
import stat
import subprocess

caffe_root = os.getcwd()

run_soon = True
resume_training = True
remove_old_models = False

top_layer = True

# Specify the batch sampler.
resize_width = 300
resize_height = 300
resize = "{}x{}".format(resize_width, resize_height)

# If true, use batch norm for all newly added layers.
# Currently only the batch norm version has been tested.
use_batchnorm = True
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.004
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.000004

pretrain_model = "/home/sweekar/SDN_main/services/SDN/Model/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
label_map_file = "/home/sweekar/SDN_main/services/SDN/labelmap_voc.prototxt"

# MultiBoxLoss parameters.
num_classes = 2
share_location = True
background_label_id = 0
train_on_diff_gt = True
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.

# minimum dimension of input image
min_dim = 300

if top_layer:
    # Proposed Layer
    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_1', 'pool6', 'conv7_1', 'pool7', 'conv8_1', 'pool8', 'conv9_1',
                          'pool9', 'conv10_1', 'pool10', 'conv11_1', 'pool11']
else:
    # VGG only
    mbox_source_layers = ['conv4_3', 'pool5', 'fc7']

# in percent %
min_ratio = 20
max_ratio = 95
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
max_sizes = []
for ratio in range(min_ratio, max_ratio + 1, step):
    min_sizes.append(min_dim * ratio / 100.)
    max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
max_sizes = [[]] + max_sizes

if top_layer:
    # Proposed Layer
    aspect_ratios = [[2, 3, 5, 7, 10], [2, 3, 5, 7, 10], [2, 3, 5, 7, 10], [2, 3, 5, 7, 10], [2, 3, 5, 7, 10],
                     [2, 3, 5, 7, 10], [2, 3, 5, 7, 10], [2, 3, 5, 7, 10], [2, 3, 5, 7, 10], [2, 3, 5, 7, 10],
                     [2, 3, 5, 7, 10],
                     [2, 3, 5, 7, 10], [2, 3, 5, 7, 10], [2, 3, 5, 7, 10]]
    normalizations = [20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

else:
    # VGG only
    aspect_ratios = [[2, 3], [2, 3], [2, 3]]
    normalizations = [20, -1, -1]

# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
    prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
    prior_variance = [0.1]
flip = False
clip = True

# Solver parameters.
# Defining which GPUs to use.
gpus = "0"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
batch_size = 5
accum_batch_size = 32
iter_size = accum_batch_size // batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
    batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
    iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
    solver_mode = P.Solver.GPU
    device_id = int(gpulist[0])

if normalization_mode == P.Loss.BATCH_SIZE:
    base_lr /= iter_size
elif normalization_mode == P.Loss.NONE:
    base_lr /= batch_size_per_device * iter_size
elif normalization_mode == P.Loss.VALID:
    base_lr *= 25. / loc_weight / iter_size
elif normalization_mode == P.Loss.FULL:
    base_lr *= 2000. / iter_size

# Which layers to freeze (no backward) during training.
freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']

# Evaluate on whole test set.
num_test_image = 193
test_batch_size = 1
test_iter = num_test_image // test_batch_size

# Import directories and parameters
# ____________________________________________

Dir = Dir(resize)
train_data, test_data, model_name, output_result_dir, train_net_file, test_net_file, deploy_net_file, solver_file, snapshot_prefix, task_file, save_dir,task_dir,snapshot_dir = Dir.dir()

Params = Params(resize_height, resize_width, base_lr, iter_size, solver_mode, device_id, test_iter, num_classes,
                share_location, background_label_id, output_result_dir, label_map_file, num_test_image, code_type,
                loc_weight, train_on_diff_gt, neg_pos_ratio, normalization_mode)
train_transform_param, test_transform_param, solver_param, det_eval_param, det_out_param, loss_param, multibox_loss_param = Params.params()

# ____________________________________________

check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(label_map_file)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(task_dir)
make_if_not_exist(snapshot_dir)


# Train Net
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
                                               train=True, output_label=True, label_map_file=label_map_file,
                                               transform_param=train_transform_param,
                                               batch_sampler=Sampler.batch_sampler)

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
           dropout=False, freeze_layers=freeze_layers)
if top_layer:
    Layer = TopLayer(net, use_batchnorm)
    net = Layer.Layers()

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
                                 use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
                                 aspect_ratios=aspect_ratios, normalizations=normalizations,
                                 num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
                                 prior_variance=prior_variance, kernel_size=[1, 5], pad=[0, 2])

# MultiBoxLossLayer.
name = "mbox_loss"
mbox_layers.append(net.label)
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
                           loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
                           propagate_down=[True, True, False, False])

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Test Net
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
                                               train=False, output_label=True, label_map_file=label_map_file,
                                               transform_param=test_transform_param)

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
           dropout=False, freeze_layers=freeze_layers)
if top_layer:
    Layer = TopLayer(net, use_batchnorm)
    net = Layer.Layers()

mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
                                 use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
                                 aspect_ratios=aspect_ratios, normalizations=normalizations,
                                 num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
                                 prior_variance=prior_variance, kernel_size=[1, 5], pad=[0, 2])

conf_name = "mbox_conf"
if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
    reshape_name = "{}_reshape".format(conf_name)
    net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
    softmax_name = "{}_softmax".format(conf_name)
    net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
    flatten_name = "{}_flatten".format(conf_name)
    net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
    mbox_layers[1] = net[flatten_name]
elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
    sigmoid_name = "{}_sigmoid".format(conf_name)
    net[sigmoid_name] = L.Sigmoid(net[conf_name])
    mbox_layers[1] = net[sigmoid_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
                                      detection_output_param=det_out_param,
                                      include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
                                         detection_evaluate_param=det_eval_param,
                                         include=dict(phase=caffe_pb2.Phase.Value('TEST')))

with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

# Deploy Net
# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)

# Solver
solver = caffe_pb2.SolverParameter(
    train_net=train_net_file,
    test_net=[test_net_file],
    snapshot_prefix=snapshot_prefix,
    **solver_param)
with open(solver_file, 'w') as f:
    print(solver, file=f)

max_iter = 0
# Snapshot
for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(model_name))[1])
        if iter > max_iter:
            max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
    if max_iter > 0:
        train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
    # Remove any snapshots smaller than max_iter.
    for file in os.listdir(snapshot_dir):
        if file.endswith(".solverstate"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(model_name))[1])
            if max_iter > iter:
                os.remove("{}/{}".format(snapshot_dir, file))
        if file.endswith(".caffemodel"):
            basename = os.path.splitext(file)[0]
            iter = int(basename.split("{}_iter_".format(model_name))[1])
            if max_iter > iter:
                os.remove("{}/{}".format(snapshot_dir, file))

# Task File
with open(task_file, 'w') as f:
    f.write('cd {}\n'.format(caffe_root))
    f.write('/home/sweekar/SDN_main/caffe/build/tools/caffe train \\\n')
    f.write('--solver="{}" \\\n'.format(solver_file))
    f.write(train_src_param)
    if solver_param['solver_mode'] == P.Solver.GPU:
        f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, task_dir, model_name))
    else:
        f.write('2>&1 | tee {}/{}.log\n'.format(task_dir, model_name))

# Copy the python script to task_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, task_dir)

# Run the task.
os.chmod(task_file, stat.S_IRWXU)
if run_soon:
    subprocess.call(task_file, shell=True)
