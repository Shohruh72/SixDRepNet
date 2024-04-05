# https://github.com/Shohruh72
import math
import copy
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
        # return RepVGG([2, 4, 14, 1], [1.5, 1.5, 1.5, 2.75], deploy=deploy)
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


def ensure_weights(filepath, url):
    import os, gdown
    if not os.path.exists(filepath):
        gdown.download(url, filepath, quiet=False)
        print(f"Downloaded weights to {filepath}")
    else:
        print(f"Pretrained Weights already present at {filepath}")


def rep_net_a0(deploy=False):
    url = 'https://drive.google.com/uc?id=13Gn8rq1PztoMEgK7rCOPMUYHjGzk-w11'
    ensure_weights('./weights/a0.pth', url)
    return RepVGG([2, 4, 14, 1], [0.75, 0.75, 0.75, 2.5], deploy=deploy)


def rep_net_a1(deploy=False):
    url = 'https://drive.google.com/uc?id=19lX6lNKSwiO5STCvvu2xRTKcPsSfWAO1'
    ensure_weights('./weights/a1.pth', url)
    return RepVGG([2, 4, 14, 1], [1, 1, 1, 2.5], deploy=deploy)


def rep_net_a2(deploy=False):
    url = 'https://drive.google.com/uc?id=1PvtYTOX4gd-1VHX8LoT7s6KIyfTKOf8G'
    ensure_weights('./weights/a2.pth', url)
    return RepVGG([2, 4, 14, 1], [1.5, 1.5, 1.5, 2.75], deploy=deploy)


def rep_net_b0(deploy=False):
    url = 'https://drive.google.com/uc?id=18g7YziprUky7cX6L6vMJ_874PP8tbtKx'
    ensure_weights('./weights/b0.pth', url)
    return RepVGG([4, 6, 16, 1], [1, 1, 1, 2.5], deploy=deploy)


def rep_net_b1(deploy=False):
    url = 'https://drive.google.com/uc?id=1VlCfXXiaJjNjzQBy3q7C3H2JcxoL0fms'
    ensure_weights('./weights/b1.pth', url)
    return RepVGG([4, 6, 16, 1], [2, 2, 2, 4], deploy=deploy)


def rep_net_b1g2(deploy=False):
    url = 'https://drive.google.com/uc?id=1PL-m9n3g0CEPrSpf3KwWEOf9_ZG-Ux1Z'
    ensure_weights('./weights/b1g2.pth', url)
    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g2_map = {l: 2 for l in optional_groupwise_layers}
    return RepVGG([4, 6, 16, 1], [2, 2, 2, 4], gr_map=g2_map, deploy=deploy)


def rep_net_b1g4(deploy=False):
    url = 'https://drive.google.com/uc?id=1WXxhyRDTgUjgkofRV1bLnwzTsFWRwZ0k'
    ensure_weights('./weights/b1g4.pth', url)
    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g4_map = {l: 4 for l in optional_groupwise_layers}
    return RepVGG([4, 6, 16, 1], [2, 2, 2, 4], gr_map=g4_map, deploy=deploy)


def rep_net_b2(deploy=False):
    url = 'https://drive.google.com/uc?id=1cFgWJkmf9U1L1UmJsA8UT__kyd3xuY_y'
    ensure_weights('./weights/b2.pth', url)
    return RepVGG([4, 6, 16, 1], [2.5, 2.5, 2.5, 5], deploy=deploy)


def rep_net_b2g4(deploy=False):
    url = 'https://drive.google.com/uc?id=1LZ61o5XH6u1n3_tXIgKII7XqKoqqracI'
    ensure_weights('./weights/b2g4.pth', url)
    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g4_map = {l: 4 for l in optional_groupwise_layers}
    return RepVGG([4, 6, 16, 1], [2.5, 2.5, 2.5, 5], gr_map=g4_map, deploy=deploy)


def rep_net_b3(deploy=False):
    url = 'https://drive.google.com/uc?id=1wBpq5317iPKk3-qblBHnx35bY_WumAlU'
    ensure_weights('./weights/b3.pth', url)
    return RepVGG([4, 6, 16, 1], [3, 3, 3, 5], deploy=deploy)


def rep_net_b3g4(deploy=False):
    url = 'https://drive.google.com/uc?id=1s7PxIP-oYB1a94_qzHyzfXAbbI24GYQ8'
    ensure_weights('./weights/b3g4.pth', url)
    optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
    g4_map = {l: 4 for l in optional_groupwise_layers}
    return RepVGG([4, 6, 16, 1], [3, 3, 3, 5], gr_map=g4_map, deploy=deploy)


class HPE(nn.Module):
    def __init__(self, model_name, weight, deploy, pretrained=True):
        super(HPE, self).__init__()
        model_selector = {
            'a0': rep_net_a0,
            'a1': rep_net_a1,
            'a2': rep_net_a2,
            'b0': rep_net_b0,
            'b1': rep_net_b1,
            'b1g2': rep_net_b1g2,
            'b1g4': rep_net_b1g4,
            'b2': rep_net_b2,
            'b2g4': rep_net_b2g4,
            'b3': rep_net_b3,
            'b3g4': rep_net_b3g4,
        }
        backbone = model_selector[model_name](deploy)

        if pretrained:
            checkpoint = torch.load(weight)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
            v in checkpoint.items()}
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


class EMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
