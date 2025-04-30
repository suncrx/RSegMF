'''

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn import Dropout, Softmax, Conv2d, LayerNorm


class v_Fusenet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self, img_inchannels=3, dsm_inchannels=1, out_channels=2):
        super(v_Fusenet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(img_inchannels, 64, 3, padding=1)
        self.dsm_conv1_1 = nn.Conv2d(dsm_inchannels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(128, out_channels, 3, padding=1)

        self.apply(self.weight_init)

    def forward(self, x, y):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)

        y = self.conv1_1_bn(F.relu(self.dsm_conv1_1(y)))
        y = self.conv1_2_bn(F.relu(self.conv1_2(y)))
        y, dsm_mask1 = self.pool(y)


        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)

        y = self.conv2_1_bn(F.relu(self.conv2_1(y)))
        y = self.conv2_2_bn(F.relu(self.conv2_2(y)))
        y, dsm_mask2 = self.pool(y)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)

        y = self.conv3_1_bn(F.relu(self.conv3_1(y)))
        y = self.conv3_2_bn(F.relu(self.conv3_2(y)))
        y = self.conv3_3_bn(F.relu(self.conv3_3(y)))
        y, dsm_mask3 = self.pool(y)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        y = self.conv4_1_bn(F.relu(self.conv4_1(y)))
        y = self.conv4_2_bn(F.relu(self.conv4_2(y)))
        y = self.conv4_3_bn(F.relu(self.conv4_3(y)))
        y, dsm_mask4 = self.pool(y)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        y = self.conv5_1_bn(F.relu(self.conv5_1(y)))
        y = self.conv5_2_bn(F.relu(self.conv5_2(y)))
        y = self.conv5_3_bn(F.relu(self.conv5_3(y)))
        y, dsm_mask5 = self.pool(y)

        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        y = self.unpool(y, dsm_mask5)
        y = self.conv5_3_D_bn(F.relu(self.conv5_3_D(y)))
        y = self.conv5_2_D_bn(F.relu(self.conv5_2_D(y)))
        y = self.conv5_1_D_bn(F.relu(self.conv5_1_D(y)))

        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        y = self.unpool(y, dsm_mask4)
        y = self.conv4_3_D_bn(F.relu(self.conv4_3_D(y)))
        y = self.conv4_2_D_bn(F.relu(self.conv4_2_D(y)))
        y = self.conv4_1_D_bn(F.relu(self.conv4_1_D(y)))

        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        y = self.unpool(y, dsm_mask3)
        y = self.conv3_3_D_bn(F.relu(self.conv3_3_D(y)))
        y = self.conv3_2_D_bn(F.relu(self.conv3_2_D(y)))
        y = self.conv3_1_D_bn(F.relu(self.conv3_1_D(y)))

        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        y = self.unpool(y, dsm_mask2)
        y = self.conv2_2_D_bn(F.relu(self.conv2_2_D(y)))
        y = self.conv2_1_D_bn(F.relu(self.conv2_1_D(y)))


        # Decoder block 1
        x = self.unpool(x, mask1)
        x_out = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        y = self.unpool(y, dsm_mask1)
        y_out = self.conv1_2_D_bn(F.relu(self.conv1_2_D(y)))
        x = torch.cat((x_out, y_out),dim=1)
        #x = F.log_softmax(self.conv1_1_D(x))
        x = F.log_softmax(self.conv1_1_D(x), dim=1)
        return x

if __name__=='__main__':
    model = v_Fusenet(img_inchannels=3, dsm_inchannels=1, 
                      out_channels=1)
    a = torch.randn(1, 3, 512, 512)
    b = torch.randn(1, 1, 512, 512)
    c = model(a, b)
    print(c.shape)
