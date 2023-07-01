from caffe.model_libs import *

# Architecture
# _____________________________________________________________
# conv4_3 --> 38 x 38
# fc7 --> 19 x 19
# conv6_1 --> 17 x 17
# pool6 --> 16 x 16
# conv7_1 --> 14 x 14
# pool7 --> 13 x 13
# conv8_1 --> 11 x 11
# pool8 --> 10 x 10
# conv9_1 --> 8 x 8
# pool9 --> 7 x 7
# conv10_1 --> 5 x 5
# pool10 --> 4 x 4
# conv11_1 --> 2 x 2
# pool11 --> 1 x 1
# ____________________________________________________________

class TopLayer:

    def __init__(self, net, use_batchnorm):
        self.net = net
        self.use_batchnorm = use_batchnorm

    def Layers(self):
        use_relu = True
        # Additional Top Layers

        from_layer = self.net.keys()[-1]
        out_layer = "conv6_1"
        ConvBNLayer(self.net, from_layer, out_layer, self.use_batchnorm, use_relu, 16, kernel_size=3, stride=1, pad=0)
        self.net.pool6 = L.Pooling(self.net[out_layer], pool=P.Pooling.MAX, kernel_size=2, stride=1)

        from_layer = self.net.keys()[-1]
        out_layer = "conv7_1"
        ConvBNLayer(self.net, from_layer, out_layer, self.use_batchnorm, use_relu, 32, kernel_size=3, stride=1, pad=0)
        self.net.pool7 = L.Pooling(self.net[out_layer], pool=P.Pooling.MAX, kernel_size=2, stride=1)

        from_layer = self.net.keys()[-1]
        out_layer = "conv8_1"
        ConvBNLayer(self.net, from_layer, out_layer, self.use_batchnorm, use_relu, 64, kernel_size=3, stride=1, pad=0)
        self.net.pool8 = L.Pooling(self.net[out_layer], pool=P.Pooling.MAX, kernel_size=2, stride=1)

        from_layer = self.net.keys()[-1]
        out_layer = "conv9_1"
        ConvBNLayer(self.net, from_layer, out_layer, self.use_batchnorm, use_relu, 128, kernel_size=3, stride=1, pad=0)
        self.net.pool9 = L.Pooling(self.net[out_layer], pool=P.Pooling.MAX, kernel_size=2, stride=1)

        from_layer = self.net.keys()[-1]
        out_layer = "conv10_1"
        ConvBNLayer(self.net, from_layer, out_layer, self.use_batchnorm, use_relu, 256, kernel_size=3, stride=1, pad=0)
        self.net.pool10 = L.Pooling(self.net[out_layer], pool=P.Pooling.MAX, kernel_size=2, stride=1)

        from_layer = self.net.keys()[-1]
        out_layer = "conv11_1"
        ConvBNLayer(self.net, from_layer, out_layer, self.use_batchnorm, use_relu, 512, kernel_size=3, stride=1, pad=0)
        self.net.pool11 = L.Pooling(self.net[out_layer], pool=P.Pooling.MAX, kernel_size=2, stride=1)

        return self.net
