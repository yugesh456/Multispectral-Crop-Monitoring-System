import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution block, a fundamental building block of Xception.
    Combines a spatial-only depthwise convolution with a channel-only pointwise convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        # Depthwise conv: each channel is convolved independently
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        # Pointwise conv: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    """
    Residual Xception block containing multiple separable convolutions with residual connections.
    """
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True, grow_first=True):
        super(XceptionBlock, self).__init__()

        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        self.rep = nn.ModuleList()
        current_channels = in_channels

        for i in range(reps):
            # If grow_first is True, expand channels in the first conv; otherwise, expand in the last conv
            if grow_first and i == 0:
                conv_out_channels = out_channels
            elif not grow_first and i == reps - 1:
                conv_out_channels = out_channels
            else:
                conv_out_channels = current_channels

            layers = []
            if start_with_relu or i > 0:
                layers.append(nn.ReLU(inplace=False))
            
            # Use stride on the final separable convolution if it's the downsampling conv
            layer_stride = stride if i == reps - 1 else 1
            
            layers.append(SeparableConv2d(current_channels, conv_out_channels, stride=layer_stride))
            layers.append(nn.BatchNorm2d(conv_out_channels))
            
            self.rep.append(nn.Sequential(*layers))
            current_channels = conv_out_channels

    def forward(self, x):
        residual = x
        
        # Apply skip connection if dimensions changed
        if self.skip is not None:
            residual = self.skip(residual)
            residual = self.skipbn(residual)

        out = x
        for layer in self.rep:
            out = layer(out)

        out = out + residual
        return out

class AgriScanXception(nn.Module):
    """
    Custom 5-Channel Xception model for multispectral image disease classification.
    Inputs: 5 bands [Blue, Green, Red, Red-Edge, NIR]
    Outputs: 5 severity levels [0: Healthy, 1: Mild/Early, 2: Moderate, 3: Severe, 4: Terminal]
    """
    def __init__(self, num_classes=5):
        super(AgriScanXception, self).__init__()
        self.num_classes = num_classes

        # Entry Flow: Custom initial layers supporting 5-channel input
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Entry Flow Blocks
        self.block1 = XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False, grow_first=True)
        self.block2 = XceptionBlock(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 512, reps=2, stride=2, start_with_relu=True, grow_first=True)

        # Middle Flow: 4 repeated residual blocks (designed for rapid GPU/CPU training)
        self.middle_blocks = nn.ModuleList([
            XceptionBlock(512, 512, reps=3, stride=1, start_with_relu=True, grow_first=True)
            for _ in range(4)
        ])

        # Exit Flow
        self.block4 = XceptionBlock(512, 1024, reps=2, stride=2, start_with_relu=True, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048)
        self.bn4 = nn.BatchNorm2d(2048)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Entry Flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle Flow
        for mblock in self.middle_blocks:
            x = mblock(x)

        # Exit Flow
        x = self.block4(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def interpret_bands(self, x):
        """
        Computes the relative saliency (gradient attribution) of each spectral band.
        Indicates model sensitivity to: [Blue, Green, Red, Red-Edge, NIR]
        """
        self.eval()
        # Enable gradient tracking on input
        x = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        logits = self(x)
        
        # Get target class (highest logit)
        score, target_idx = torch.max(logits, dim=1)
        
        # Backward pass
        self.zero_grad()
        score.backward()
        
        # Saliency is the absolute gradient of the input
        saliency, _ = torch.max(torch.abs(x.grad.data), dim=1) # Max over spatial dimensions
        
        # Average across spatial dimensions per batch sample to get channel-wise importance
        channel_importance = torch.mean(torch.abs(x.grad.data), dim=(2, 3)) # (Batch, Channels)
        
        # Normalize per sample so importances sum to 100%
        normalized_importance = []
        for sample_grads in channel_importance:
            total = torch.sum(sample_grads)
            if total > 0:
                normalized_importance.append((sample_grads / total * 100.0).tolist())
            else:
                normalized_importance.append([20.0, 20.0, 20.0, 20.0, 20.0]) # Equal default
                
        return normalized_importance
