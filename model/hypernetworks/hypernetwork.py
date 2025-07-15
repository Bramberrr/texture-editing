from torch import nn
from torch.nn import BatchNorm2d, PReLU, Sequential, Module
from torchvision.models import resnet34

# from training.hypernetworks.refinement_blocks import HyperRefinementBlock, RefinementBlock, RefinementBlockSeparable
from model.hypernetworks.refinement_blocks import HyperRefinementBlockG3, RefinementBlockG3
from model.hypernetworks.shared_weights_hypernet import SharedWeightsHypernet

class SharedWeightsHyperNetResNetG3(nn.Module):
    def __init__(self, opts):
        super(SharedWeightsHyperNetResNetG3, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.PReLU(64)

        # Load pretrained ResNet backbone
        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        self.body = nn.Sequential(*[b for block in blocks for b in block])

        # Decide layers to tune
        if len(opts.layers_to_tune) == 0:
            self.layers_to_tune = list(range(opts.n_hypernet_outputs))
        else:
            self.layers_to_tune = [int(l) for l in opts.layers_to_tune.split(',')]

        self.shared_layers = [2,3,4,5,6,7,8]
        self.shared_weight_hypernet = SharedWeightsHypernet(
            f_size=1,
            in_size=1024,
            out_size=1024,
            mode=None
        )

        self.refinement_blocks = nn.ModuleList()
        self.n_outputs = opts.n_hypernet_outputs

        for layer_idx in range(self.n_outputs):
            if layer_idx in self.layers_to_tune:
                if layer_idx in self.shared_layers:
                    refinement_block = HyperRefinementBlockG3(
                        shared_hypernet=self.shared_weight_hypernet,
                        spatial=16,  # or set per-layer spatial resolution if needed
                        in_feat=512,
                        mid_feat=128
                    )
                else:
                    refinement_block = RefinementBlockG3(
                        layer_idx=layer_idx,
                        spatial=16,
                        in_feat=512,
                        mid_feat=256
                    )
            else:
                refinement_block = None
            self.refinement_blocks.append(refinement_block)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)

        weight_deltas = []
        for j in range(self.n_outputs):
            if self.refinement_blocks[j] is not None:
                delta = self.refinement_blocks[j](x)
            else:
                delta = None
            weight_deltas.append(delta)
        return weight_deltas


# class SharedWeightsHyperNetResNet(Module):

#     def __init__(self, opts):
#         super(SharedWeightsHyperNetResNet, self).__init__()

#         self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = BatchNorm2d(64)
#         self.relu = PReLU(64)

#         resnet_basenet = resnet34(pretrained=True)
#         blocks = [
#             resnet_basenet.layer1,
#             resnet_basenet.layer2,
#             resnet_basenet.layer3,
#             resnet_basenet.layer4
#         ]
#         modules = []
#         for block in blocks:
#             for bottleneck in block:
#                 modules.append(bottleneck)
#         self.body = Sequential(*modules)

#         if len(opts.layers_to_tune) == 0:
#             self.layers_to_tune = list(range(opts.n_hypernet_outputs))
#         else:
#             self.layers_to_tune = [int(l) for l in opts.layers_to_tune.split(',')]

#         self.shared_layers = [0, 2, 3, 5, 6, 8, 9, 11, 12]
#         self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=None)

#         self.refinement_blocks = nn.ModuleList()
#         self.n_outputs = opts.n_hypernet_outputs
#         for layer_idx in range(self.n_outputs):
#             if layer_idx in self.layers_to_tune:
#                 if layer_idx in self.shared_layers:
#                     refinement_block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512, inner_c=128)
#                 else:
#                     refinement_block = RefinementBlock(layer_idx, opts, n_channels=512, inner_c=256)
#             else:
#                 refinement_block = None
#             self.refinement_blocks.append(refinement_block)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.body(x)
#         weight_deltas = []
#         for j in range(self.n_outputs):
#             if self.refinement_blocks[j] is not None:
#                 delta = self.refinement_blocks[j](x)
#             else:
#                 delta = None
#             weight_deltas.append(delta)
#         return weight_deltas


# class SharedWeightsHyperNetResNetSeparable(Module):

#     def __init__(self, opts):
#         super(SharedWeightsHyperNetResNetSeparable, self).__init__()

#         self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = BatchNorm2d(64)
#         self.relu = PReLU(64)

#         resnet_basenet = resnet34(pretrained=True)
#         blocks = [
#             resnet_basenet.layer1,
#             resnet_basenet.layer2,
#             resnet_basenet.layer3,
#             resnet_basenet.layer4
#         ]
#         modules = []
#         for block in blocks:
#             for bottleneck in block:
#                 modules.append(bottleneck)
#         self.body = Sequential(*modules)

#         if len(opts.layers_to_tune) == 0:
#             self.layers_to_tune = list(range(opts.n_hypernet_outputs))
#         else:
#             self.layers_to_tune = [int(l) for l in opts.layers_to_tune.split(',')]

#         self.shared_layers = [0, 2, 3, 5, 6, 8, 9, 11, 12]
#         self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=None)

#         self.refinement_blocks = nn.ModuleList()
#         self.n_outputs = opts.n_hypernet_outputs
#         for layer_idx in range(self.n_outputs):
#             if layer_idx in self.layers_to_tune:
#                 if layer_idx in self.shared_layers:
#                     refinement_block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512, inner_c=128)
#                 else:
#                     refinement_block = RefinementBlockSeparable(layer_idx, opts, n_channels=512, inner_c=256)
#             else:
#                 refinement_block = None
#             self.refinement_blocks.append(refinement_block)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.body(x)
#         weight_deltas = []
#         for j in range(self.n_outputs):
#             if self.refinement_blocks[j] is not None:
#                 delta = self.refinement_blocks[j](x)
#             else:
#                 delta = None
#             weight_deltas.append(delta)
#         return weight_deltas
