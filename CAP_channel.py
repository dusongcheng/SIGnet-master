import torch
import torch.nn as nn


class GMLayer(nn.Module):
    def __init__(self, channel=256, reduction=16):
        super(GMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gm1 = nn.Sequential(
                nn.Linear(channel, reduction),
                nn.ReLU(inplace=True),
                nn.Linear(reduction, channel),
                nn.Sigmoid()
        )
        self.gm2 = nn.Sequential(
                nn.Linear(channel, reduction),
                nn.ReLU(inplace=True),
                nn.Linear(reduction, channel),
                nn.Sigmoid()
        )
        self.gm3 = nn.Sequential(
                nn.Linear(channel, reduction),
                nn.ReLU(inplace=True),
                nn.Linear(reduction, channel),
                nn.Sigmoid()
        )
        self.gm4 = nn.Sequential(
                nn.Linear(channel, reduction),
                nn.ReLU(inplace=True),
                nn.Linear(reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        g1 = self.gm1(y).view(b, c, 1)
        g2 = self.gm2(y).view(b, c, 1)
        g3 = self.gm3(y).view(b, c, 1)
        g4 = self.gm4(y).view(b, c, 1)
        channel_guidance = torch.cat((g1, g2, g3, g4), -1)
        return channel_guidance


class Affinity_Propagate_Channel(nn.Module):
    def __init__(self, prop_time=4, prop_kernel=5, norm_type='4sum'):
        super(Affinity_Propagate_Channel, self).__init__()
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        assert prop_kernel == 5, 'this version only support 4 (5 - 1) neighborhood'

        self.norm_type = norm_type
        assert norm_type == '4sum'

        self.in_feature = 1
        self.out_feature = 1

    def forward(self, guidance, blur_depth):
        gate_wb, gate_sum = self.affinity_normalization(guidance)  # b*c*4*1*1, b*c*1*1

        # pad input and convert to 8 channel 3D features
        raw_depth_input = blur_depth
        result_depth = blur_depth

        for i in range(self.prop_time):
            # one propagation
            # spn_kernel = self.prop_kernel
            result_depth = self.pad_blur_depth(result_depth)  # b*c*2*h*w
            neigbor_weighted_sum = torch.sum(gate_wb * result_depth, 2, keepdim=False)
            result_depth = neigbor_weighted_sum

            if '4sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            else:
                raise ValueError('unknown norm %s' % self.norm_type)

        return result_depth

    def affinity_normalization(self, guidance):
        gate1_wb_cmb = guidance.narrow(-1, 0                   , self.out_feature).unsqueeze(1)  # b*c*1
        gate2_wb_cmb = guidance.narrow(-1, 1 * self.out_feature, self.out_feature).unsqueeze(1)
        gate3_wb_cmb = guidance.narrow(-1, 2 * self.out_feature, self.out_feature).unsqueeze(1)
        gate4_wb_cmb = guidance.narrow(-1, 3 * self.out_feature, self.out_feature).unsqueeze(1)

        # gate1:before_top, gate2:before_second, gate3:behind_second, gate4:behind_top
        before_top_pad = nn.ZeroPad2d((0, 0, 0, 2))
        gate1_wb_cmb = before_top_pad(gate1_wb_cmb).squeeze(1)
        gate1_wb_cmb = gate1_wb_cmb[:, 2:, :]
        before_second_pad = nn.ZeroPad2d((0, 0, 0, 1))
        gate2_wb_cmb = before_second_pad(gate2_wb_cmb).squeeze(1)
        gate2_wb_cmb = gate2_wb_cmb[:, 1:, :]
        behind_second_pad = nn.ZeroPad2d((0, 0, 1, 0))
        gate3_wb_cmb = behind_second_pad(gate3_wb_cmb).squeeze(1)
        gate3_wb_cmb = gate3_wb_cmb[:, :-1, :]
        behind_top_pad = nn.ZeroPad2d((0, 0, 2, 0))
        gate4_wb_cmb = behind_top_pad(gate4_wb_cmb).squeeze(1)
        gate4_wb_cmb = gate4_wb_cmb[:, :-2, :]

        gate_wb = torch.cat((gate1_wb_cmb, gate2_wb_cmb, gate3_wb_cmb, gate4_wb_cmb), -1)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = torch.sum(gate_wb_abs, -1, keepdim=True)

        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = torch.sum(gate_wb, -1, keepdim=True)

        gate_wb = gate_wb.unsqueeze(-1).unsqueeze(-1)
        gate_sum = gate_sum.unsqueeze(-1)

        return gate_wb, gate_sum

    def pad_blur_depth(self, blur_depth):
        b, c, h, w = blur_depth.size()
        blur_depth = blur_depth.view(b, c, h*w).unsqueeze(1)

        before_top_pad = nn.ZeroPad2d((0, 0, 0, 2))
        blur_depth_1 = before_top_pad(blur_depth).squeeze(1)
        blur_depth_1 = blur_depth_1[:, 2:, :].view(b, c, 1, h, w)
        before_second_pad = nn.ZeroPad2d((0, 0, 0, 1))
        blur_depth_2 = before_second_pad(blur_depth).squeeze(1)
        blur_depth_2 = blur_depth_2[:, 1:, :].view(b, c, 1, h, w)
        behind_second_pad = nn.ZeroPad2d((0, 0, 1, 0))
        blur_depth_3 = behind_second_pad(blur_depth).squeeze(1)
        blur_depth_3 = blur_depth_3[:, :-1, :].view(b, c, 1, h, w)
        behind_top_pad = nn.ZeroPad2d((0, 0, 2, 0))
        blur_depth_4 = behind_top_pad(blur_depth).squeeze(1)
        blur_depth_4 = blur_depth_4[:, :-2, :].view(b, c, 1, h, w)

        result_depth = torch.cat((blur_depth_1, blur_depth_2, blur_depth_3, blur_depth_4), 2)
        return result_depth
