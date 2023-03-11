import torch
import spconv.pytorch as spconv
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from modules.netvlad import NetVLADLoupe


class BottleneckSparse2D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super(BottleneckSparse2D, self).__init__()
        self.conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels // 4, 1),
            torch.nn.BatchNorm1d(out_channels // 4), torch.nn.ReLU(),
            spconv.SubMConv2d(out_channels // 4, out_channels // 4,
                              kernel_size),
            torch.nn.BatchNorm1d(out_channels // 4), torch.nn.ReLU(),
            spconv.SubMConv2d(out_channels // 4, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels))
        self.shotcut_conv = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels))
        self.relu = spconv.SparseSequential(torch.nn.ReLU())

    def forward(self, x):
        y = self.conv(x)
        shortcut = self.shotcut_conv(x)
        y = spconv.functional.sparse_add(y, shortcut)
        return self.relu(y)



class FeatureFuse(torch.nn.Module):

    def __init__(self, feature_dim, num_heads=1) -> None:
        super(FeatureFuse, self).__init__()
        self.mutihead_attention = AttentionalPropagation(
            feature_dim, num_heads)
    def forward(self, x, source):
        return (x + self.mutihead_attention(x, source))


class backbone(torch.nn.Module):
    def __init__(self, inchannels=64) -> None:
        super(backbone, self).__init__()
        self.dconv_down1 = BottleneckSparse2D(inchannels, inchannels * 2, 11)
        self.dconv_down1_1 = BottleneckSparse2D(inchannels * 2, inchannels * 2,
                                                11)
        self.dconv_down2 = BottleneckSparse2D(inchannels * 2, inchannels * 4,
                                              7)
        self.dconv_down2_1 = BottleneckSparse2D(inchannels * 4, inchannels * 4,
                                                7)
        self.dconv_down3 = BottleneckSparse2D(inchannels * 4, inchannels * 8,
                                              5)
        self.dconv_down3_1 = BottleneckSparse2D(inchannels * 8, inchannels * 8,
                                                5)
        self.dconv_down4 = spconv.SubMConv2d(inchannels * 8,
                                             inchannels * 16,
                                             3,
                                             bias=True)
        self.maxpool1 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up1')
        self.maxpool2 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up2')
        self.maxpool3 = spconv.SparseMaxPool2d(3, 2, 1, indice_key='up3')

    def forward(self, x):
        x = spconv.SparseConvTensor.from_dense(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool1(conv1)
        x = self.dconv_down1_1(x)
        conv2 = self.dconv_down2(x)
        x = self.maxpool2(conv2)
        x = self.dconv_down2_1(x)
        conv3 = self.dconv_down3(x)
        x = self.maxpool3(conv3)
        x = self.dconv_down3_1(x)
        x = self.dconv_down4(x)
        return x.dense()


class vlad_head(torch.nn.Module):
    def __init__(self) -> None:
        super(vlad_head, self).__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, 1, 1),
                                        torch.nn.ReLU6(),
                                        torch.nn.Conv2d(512, 512, 3, 1, 1))
        self.vlad = NetVLADLoupe(
            feature_size=512,
            max_samples=1024,
            cluster_size=32,
            output_dim=1024,
            gating=True,
            add_batch_norm=True,
            is_training=True)

    def forward(self, x):
        x = self.conv(x)
        return self.vlad(x.reshape(x.shape[0], x.shape[1], -1, 1))


class overlap_head(torch.nn.Module):
    def __init__(self, inchannels=64) -> None:
        super(overlap_head, self).__init__()
        self.fusenet16 = FeatureFuse(inchannels * 16)
        self.last_conv16 = spconv.SparseSequential(
            spconv.SubMConv2d(inchannels * 16, inchannels * 8, 3, bias=True),
            torch.nn.BatchNorm1d(inchannels * 8), torch.nn.ReLU(),
            spconv.SubMConv2d(inchannels * 8, 1, 3, bias=True))

    def forward(self, x):
        x40 = spconv.SparseConvTensor.from_dense(x)
        feature = x40.features
        mask0 = (x40.indices[:, 0] == 0)
        mask1 = (x40.indices[:, 0] == 1)
        fea1 = self.fusenet16(feature[mask0].permute(1, 0).unsqueeze(0),
                              feature[mask1].permute(
                                  1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        fea2 = self.fusenet16(feature[mask1].permute(1, 0).unsqueeze(0),
                              feature[mask0].permute(
                                  1, 0).unsqueeze(0)).squeeze(0).permute(1, 0)
        x40 = x40.replace_feature(torch.cat([fea1, fea2], dim=0))
        out4 = self.last_conv16(x40)
        out4 = out4.replace_feature(torch.sigmoid(out4.features))
        score0 = out4.features[mask0]
        score1 = out4.features[mask1]

        # im0 = torch.zeros(x40.spatial_shape).float().to(fea1.device)
        # indi0 = x40.indices[mask0].long()
        # im0[indi0[:,1], indi0[:,2]] = score0.reshape(-1)
        # plt.subplot(1, 2, 1)
        # plt.imshow(im0.detach().cpu().numpy(), cmap = "Reds")
        # # plt.show()

        # im1 = torch.zeros(x40.spatial_shape).float().to(fea1.device)
        # indi1 = x40.indices[mask1].long()
        # im1[indi1[:,1], indi1[:,2]] = score1.reshape(-1)
        # plt.subplot(1, 2, 2)
        # plt.imshow(im1.detach().cpu().numpy(), cmap = "Reds")
        # plt.show()

        score_sum0 = torch.sum(score0) / len(score0)
        score_sum1 = torch.sum(score1) / len(score1)
        return (score_sum0 + score_sum1) / 2., out4


if __name__ == "__main__":
    pass
