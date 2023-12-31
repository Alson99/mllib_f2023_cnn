import torch
import torch.nn as nn


class InputStem(nn.Module):
    def __init__(self):
        """
            Входной блок нейронной сети ResNet, содержит свертку 7x7 c количеством фильтров 64 и шагом 2, затем
            следует max-pooling 3x3 с шагом 2.
            
            TODO: инициализируйте слои входного блока
        """
        super(InputStem, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        # реализуйте forward pass
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1, down_sampling=False):
        """
            Остаточный блок, состоящий из 3 сверточных слоев (path A) и shortcut connection (path B).
            Может быть двух видов:
                1. Down sampling (только первый Bottleneck блок в Stage)
                2. Residual (последующие Bottleneck блоки в Stage)

            Path A:
                Cостоит из 3-x сверточных слоев (1x1, 3x3, 1x1), после каждого слоя применяется BatchNorm,
                после первого и второго слоев - ReLU. Количество фильтров для первого слоя - out_channels,
                для второго слоя - out_channels, для третьего слоя - out_channels * expansion.

            Path B:
                1. Down sampling: path B = Conv (1x1, stride) и  BatchNorm
                2. Residual: path B = nn.Identity

            Выход Bottleneck блока - path_A(inputs) + path_B(inputs)

            :param in_channels: int - количество фильтров во входном тензоре
            :param out_channels: int - количество фильтров в промежуточных слоях
            :param expansion: int = 4 - множитель на количество фильтров в выходном слое
            :param stride: int
            :param down_sampling: bool
            TODO: инициализируйте слои Bottleneck
        """
        super(Bottleneck, self).__init__()

        self.downsampling = down_sampling
        self.stride = stride

        self.path_A = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * expansion),
        )

        if down_sampling:
            self.path_B = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * expansion),
            )
        else:
            self.path_B = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        # реализуйте forward pass
        return self.relu(self.path_A(inputs) + self.path_B(inputs))


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, nrof_blocks: int):
        """
            Последовательность Bottleneck блоков, первый блок Down sampling, остальные - Residual

            :param nrof_blocks: int - количество Bottleneck блоков
            TODO: инициализируйте слои, используя класс Bottleneck
        """
        super(Stage, self).__init__()

        self.blocks = nn.Sequential(
            Bottleneck(in_channels, out_channels, down_sampling=True),
            *[Bottleneck(out_channels * 4, out_channels) for _ in range(1, nrof_blocks)]
        )

    def forward(self, inputs):
        # TODO: реализуйте forward pass
        return self.blocks(inputs)


# Пример использования классов
input_stem = InputStem()
bottleneck = Bottleneck(in_channels=64, out_channels=128)
stage = Stage(in_channels=128 * 4, out_channels=256, nrof_blocks=3)

# Forward pass example
x = torch.randn(1, 3, 224, 224)
x = input_stem(x)
x = bottleneck(x)
x = stage(x)
print(x.shape)