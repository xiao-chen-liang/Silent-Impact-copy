import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ClassificationModel(nn.Module):
    def __init__(self, input_channels, num_classes, seq_len):
        super(ClassificationModel, self).__init__()

        self.att_1 = nn.Sequential(
            nn.Conv1d(input_channels, 1, kernel_size=11, padding=5),
            nn.Sigmoid()
        )
        self.att_2 = nn.Sequential(
            nn.Conv1d(input_channels, 1, kernel_size=11, padding=5),
            nn.Sigmoid()
        )
        self.att_3 = nn.Sequential(
            nn.Conv1d(input_channels, 1, kernel_size=11, padding=5),
            nn.Sigmoid()
        )

        self.att_classifier_1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.ReLU(),
            Flatten(),
            nn.Linear(seq_len * 16, num_classes),
            nn.Softmax(dim=1)
        )
        self.att_classifier_2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.ReLU(),
            Flatten(),
            nn.Linear(seq_len * 16, num_classes),
            nn.Softmax(dim=1)
        )
        self.att_classifier_3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, padding=5),
            nn.ReLU(),
            Flatten(),
            nn.Linear(seq_len * 16, num_classes),
            nn.Softmax(dim=1)
        )
        
        # First convolution block
        self.conv1_0 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.conv1_1 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.conv1_3 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.bn1_0 = nn.BatchNorm1d(32)
        self.bn1_1 = nn.BatchNorm1d(32)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.bn1_3 = nn.BatchNorm1d(32)
        self.mish1 = Mish()
        
        # Second convolution block
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.mish2 = Mish()
        
        # Third convolution block
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.mish3 = Mish()
        
        # Global Average Pooling (GAP) layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, xfft):
        # torch.Size([1, 112262, 6]) torch.Size([1, 3, 112262, 6])
        x = x.transpose(1, 2)
        xfft = xfft.transpose(2, 3)

        a1 = self.att_1(xfft[:, 0])
        a2 = self.att_2(xfft[:, 1])
        a3 = self.att_3(xfft[:, 2])

        x0_hid = self.conv1_0(x)
        x0 = self.mish1(self.bn1_0(x0_hid))

        x1_hid = self.conv1_1(x0 * a1)
        x1 = self.mish1(self.bn1_1(x1_hid))

        x2_hid = self.conv1_2(x1 * a2)
        x2 = self.mish1(self.bn1_2(x2_hid))

        x3_hid = self.conv1_3(x2 * a3)
        x3 = self.mish1(self.bn1_3(x3_hid))

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.mish2(self.bn2(self.conv2(x)))
        x = self.mish3(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.softmax(x)
        
        att_out_1 = self.att_classifier_1(a1)
        att_out_2 = self.att_classifier_2(a2)
        att_out_3 = self.att_classifier_3(a3)
        
        outputs = torch.stack([att_out_1, att_out_2, att_out_3, out], dim=0)
        return outputs



class DetectionModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(DetectionModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        outputs = outputs.transpose(2, 3)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size=3):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation*(kernel_size//2), dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)
