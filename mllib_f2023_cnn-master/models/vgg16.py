import torch
import torch.nn as nn

from models.blocks.vgg16_blocks import conv_block, classifier_block


class VGG16(nn.Module):
    def __init__(self, cfg, nrof_classes):
        """https://arxiv.org/pdf/1409.1556.pdf"""
        super(VGG16, self).__init__()

        self.cfg = cfg
        self.nrof_classes = nrof_classes

        # инициализируйте сверточные слои модели, используя функцию conv_block
        self.conv1 = conv_block(in_channels=[3, 64], out_channels=[64, 64],
                                conv_params=dict(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                maxpool_params=dict(kernel_size=2, stride=2, padding=0))
        self.conv2 = conv_block(in_channels=[64, 128], out_channels=[128, 128],
                                conv_params=dict(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                maxpool_params=dict(kernel_size=2, stride=2, padding=0))
        self.conv3 = conv_block(in_channels=[128, 256, 256], out_channels=[256, 256, 256],
                                conv_params=dict(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                maxpool_params=dict(kernel_size=2, stride=2, padding=0))
        self.conv4 = conv_block(in_channels=[256, 512, 512], out_channels=[512, 512, 512],
                                conv_params=dict(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                maxpool_params=dict(kernel_size=2, stride=2, padding=0))
        self.conv5 = conv_block(in_channels=[512, 512, 512], out_channels=[512, 512, 512],
                                conv_params=dict(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                maxpool_params=dict(kernel_size=2, stride=2, padding=0))

        #  инициализируйте полносвязные слои модели, используя функцию classifier_block
        #  (последний слой инициализируется отдельно)
        self.linears = classifier_block(in_features=[25088, 4096, 4096], out_features=[4096, 4096, nrof_classes])

        #  инициализируйте последний полносвязный слой для классификации с помощью
        #  nn.Linear(in_features=4096, out_features=nrof_classes)
        self.classifier = nn.Linear(in_features=4096, out_features=nrof_classes)

    def forward(self, inputs):
        """
           Forward pass нейронной сети, все вычисления производятся для батча
           :param inputs: torch.Tensor(batch_size, channels, height, weight)
           :return output of the model: torch.Tensor(batch_size, nrof_classes)

           # реализуйте forward pass
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        x = self.linears(x)
        output = self.classifier(x)

        return output
