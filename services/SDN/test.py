import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import xml.dom.minidom

# %matplotlib inline

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/sweekar/SDN_main/caffe'  # this file is expected to be in {caffe_root}/examples
os.chdir(caffe_root)
import sys

sys.path.insert(0, 'python')

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()


model_def = '/home/sweekar/SDN_main/models/VGG/300x300/deploy.prototxt'
model_weights = '/home/sweekar/SDN_main/models/VGG/300x300/VGG_300x300_iter_6200.caffemodel'


scales = ((500, 500),)

net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
print(net.blobs['data'].data.shape)

test_list = open('/home/sweekar/SDN_main/evaluation/test_list_ha.txt')
save_dir = '/home/sweekar/SDN_main/evaluation/dt/'
for line in test_list.readlines():
    line = line.strip()
    image_name = line
    image_path = '/home/sweekar/SDN_main/dataset/test_dataset/' + line
    save_detection_path = save_dir + 'vgg_' + line[0:len(line) - 4] + '.txt'
    image = caffe.io.load_image(image_path)
    image_height, image_width, channels = image.shape
    detection_result = open(save_detection_path, 'wt')
    for scale in scales:
        image_resize_height = scale[0]
        image_resize_width = scale[1]
        transformer = caffe.io.Transformer({'data': (1, 3, image_resize_height, image_resize_width)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        net.blobs['data'].reshape(1, 3, image_resize_height, image_resize_width)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        plt.clf()
        plt.imshow(image)
        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            xmin = max(1, xmin)
            ymin = max(1, ymin)
            xmax = min(image.shape[1] - 1, xmax)
            ymax = min(image.shape[0] - 1, ymax)
            score = top_conf[i]
            result = str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\r\n'
            detection_result.write(result)

            name = '%.2f' % (score)
            coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
            face_color_opacity = 0.4
            currentAxis.add_patch(plt.Rectangle(*coords, fill=True, facecolor=(1, 0, 0, face_color_opacity)))

    detection_result.close()
    plt.savefig('/home/sweekar/SDN_main/dataset/test_plots/' + image_name)
test_list.close()
print('success')
