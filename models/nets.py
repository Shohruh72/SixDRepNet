import torch
import torch.nn as nn
from utils import util


def conv_bn(inp, oup, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(oup))
    return result


class RepVGGBlock(nn.Module):
    def __init__(self, inp, oup, k, s=1, p=0, d=1, gr=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.inp = inp
        self.groups = gr
        self.deploy = deploy
        self.nonlinearity = nn.ReLU()
        self.se = nn.Identity()

        assert k == 3
        assert p == 1

        padding = p - k // 2

        if deploy:
            self.rbr_reparam = nn.Conv2d(inp, oup, k, s, p, d, gr, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(inp) if oup == inp and s == 1 else None
            self.rbr_dense = conv_bn(inp, oup, k, s, p, groups=gr)
            self.rbr_1x1 = conv_bn(inp, oup, 1, s, padding, groups=gr)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(x)))

        if self.rbr_identity is None:
            out = 0
        else:
            out = self.rbr_identity(x)

        return self.nonlinearity(self.se(self.rbr_dense(x) + self.rbr_1x1(x) + out))


class RepVGG(nn.Module):
    def __init__(self, layers, width=None, num_cls=1000, gr_map=None, deploy=False):
        super(RepVGG, self).__init__()
        self.deploy = deploy
        self.cur_layer_idx = 1
        self.gr_map = gr_map or dict()

        assert len(width) == 4
        assert 0 not in self.gr_map

        self.inp = min(64, int(64 * width[0]))

        self.stage0 = RepVGGBlock(3, self.inp, 3, 2, 1, deploy=self.deploy)
        self.stage1 = self._make_stage(int(64 * width[0]), layers[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width[1]), layers[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width[2]), layers[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width[3]), layers[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width[3]), num_cls)

    def _make_stage(self, oup, layer, stride):
        strides = [stride] + [1] * (layer - 1)
        layers = []
        for stride in strides:
            cur_groups = self.gr_map.get(self.cur_layer_idx, 1)
            layers.append(RepVGGBlock(self.inp, oup, 3, stride, p=1, gr=cur_groups, deploy=self.deploy, ))
            self.inp = oup
            self.cur_layer_idx += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def create_model(backbone_name, num_cls=1000):
    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g2_map = {l: 2 for l in optional_groupwise_layers}
    g4_map = {l: 4 for l in optional_groupwise_layers}
    default_group_map = None
    net_configs = {
        'RepVGG-A0': ([2, 4, 14, 1], [0.75, 0.75, 0.75, 2.5], default_group_map),
        'RepVGG-A1': ([2, 4, 14, 1], [1, 1, 1, 2.5], default_group_map),
        'RepVGG-A2': ([2, 4, 14, 1], [1.5, 1.5, 1.5, 2.75], default_group_map),
        'RepVGG-B0': ([4, 6, 16, 1], [1, 1, 1, 2.5], default_group_map),
        'RepVGG-B1': ([4, 6, 16, 1], [2, 2, 2, 4], default_group_map),
        'RepVGG-B1g2': ([4, 6, 16, 1], [2, 2, 2, 4], g2_map),
        'RepVGG-B1g4': ([4, 6, 16, 1], [2, 2, 2, 4], g4_map),
        'RepVGG-B2': ([4, 6, 16, 1], [2.5, 2.5, 2.5, 5], default_group_map),
        'RepVGG-B2g2': ([4, 6, 16, 1], [2.5, 2.5, 2.5, 5], g2_map),
        'RepVGG-B2g4': ([4, 6, 16, 1], [2.5, 2.5, 2.5, 5], g4_map),
        'RepVGG-B3': ([4, 6, 16, 1], [3, 3, 3, 5], default_group_map),
        'RepVGG-B3g2': ([4, 6, 16, 1], [3, 3, 3, 5], g2_map),
        'RepVGG-B3g4': ([4, 6, 16, 1], [3, 3, 3, 5], g4_map),
    }

    def model_constructor(deploy):
        configs = net_configs.get(backbone_name)
        if configs is None:
            raise ValueError(f"Network {backbone_name} is not supported.")
        layers, width, gr_map = configs[:3]
        return RepVGG(layers, width, num_cls, gr_map, deploy=deploy)

    return model_constructor


class HPE(nn.Module):
    def __init__(self, model_name, weight, deploy, pretrained=True):
        super(HPE, self).__init__()
        repvgg = create_model(model_name)
        backbone = repvgg(deploy)
        if pretrained:
            checkpoint = torch.load(weight)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
            v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)

        return util.compute_rotation(x)
