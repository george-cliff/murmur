# stdlib

# third-party
import torch
import torch.nn as nn
import torch.nn.functional as F

# local
from tools.paths import MODELS_DIR

# constants for easy tuning
NEG_SLOPE = 0.01
CONV1_OUT = 32
CONV2_OUT = 64
CONV3_OUT = 128
CONV4_OUT = 128
GAP_H, GAP_W = 2, 2
FC1_IN = CONV4_OUT * GAP_H * GAP_W
FC1_OUT = 128
CONV_KERNEL = 5
PADDING = (CONV_KERNEL - 1) // 2
DROPOUT = 0.5


class Sparrowhawk(nn.Module):
    def __init__(self):
        super().__init__()

        # bias=False on every conv as BatchNorm re-centers the output right after
        self.conv1a = nn.Conv2d(
            in_channels=1,
            out_channels=CONV1_OUT,
            kernel_size=CONV_KERNEL,
            padding=PADDING,
            bias=False,
        )
        self.conv1b = nn.Conv2d(
            in_channels=CONV1_OUT,
            out_channels=CONV1_OUT,
            kernel_size=CONV_KERNEL,
            padding=PADDING,
            bias=False,
        )

        # NEXT DEPTH TEST: split conv2 into conv2a/conv2b too, same pattern as conv1a/conv1b
        self.conv2a = nn.Conv2d(
            in_channels=CONV1_OUT,
            out_channels=CONV2_OUT,
            kernel_size=CONV_KERNEL,
            padding=PADDING,
            bias=False,
        )

        self.conv2b = nn.Conv2d(
            in_channels=CONV2_OUT,
            out_channels=CONV2_OUT,
            kernel_size=CONV_KERNEL,
            padding=PADDING,
            bias=False,
        )

        # self.conv2 = nn.Conv2d(
        #    in_channels=CONV1_OUT,
        #    out_channels=CONV2_OUT,
        #    kernel_size=CONV_KERNEL,
        #    padding=PADDING,
        #    bias=False,
        # )

        self.conv3 = nn.Conv2d(
            in_channels=CONV2_OUT,
            out_channels=CONV3_OUT,
            kernel_size=CONV_KERNEL,
            padding=PADDING,
            bias=False,
        )

        self.conv4 = nn.Conv2d(
            in_channels=CONV3_OUT,
            out_channels=CONV4_OUT,
            kernel_size=CONV_KERNEL,
            padding=PADDING,
            bias=False,
        )

        self.fc1 = nn.Linear(FC1_IN, FC1_OUT)
        self.fc2 = nn.Linear(FC1_OUT, 1)

        self.bn1a = nn.BatchNorm2d(num_features=CONV1_OUT)
        self.bn1b = nn.BatchNorm2d(num_features=CONV1_OUT)
        # NEXT DEPTH TEST: bn2a/bn2b to match conv2a/conv2b above
        self.bn2a = nn.BatchNorm2d(num_features=CONV2_OUT)
        self.bn2b = nn.BatchNorm2d(num_features=CONV2_OUT)
        # self.bn2 = nn.BatchNorm2d(num_features=CONV2_OUT)
        self.bn3 = nn.BatchNorm2d(num_features=CONV3_OUT)
        self.bn4 = nn.BatchNorm2d(num_features=CONV4_OUT)

        self.gap = nn.AdaptiveAvgPool2d((GAP_H, GAP_W))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=DROPOUT)

    def forward(self, x):
        # conv block
        x = self.conv1a(x)
        x = self.bn1a(x)
        x = F.leaky_relu(input=x, negative_slope=NEG_SLOPE)
        x = self.conv1b(x)
        x = self.bn1b(x)
        x = F.leaky_relu(input=x, negative_slope=NEG_SLOPE)
        x = F.max_pool2d(x, kernel_size=2)
        # NEXT DEPTH TEST: swap this block for conv2a/conv2b, same pattern as conv1a/conv1b above
        x = self.conv2a(x)
        x = self.bn2a(x)
        x = F.leaky_relu(input=x, negative_slope=NEG_SLOPE)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = F.leaky_relu(input=x, negative_slope=NEG_SLOPE)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = F.leaky_relu(input=x, negative_slope=NEG_SLOPE)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(input=x, negative_slope=NEG_SLOPE)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(input=x, negative_slope=NEG_SLOPE)
        x = F.max_pool2d(x, kernel_size=2)

        # gap block
        x = self.gap(x)

        # classifier block
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.leaky_relu(input=x, negative_slope=NEG_SLOPE)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = Sparrowhawk()
    print(model)
    dummy_input = torch.randn(8, 1, 128, 1000)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs: {output[:4].detach().squeeze()}")
    torch.save(obj=model.state_dict(), f=f"{MODELS_DIR}/sample_model.pt")
