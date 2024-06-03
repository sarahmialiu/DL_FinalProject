import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class conv_layer(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class block_1(nn.Module):
    # 16, 32
    def __init__(self, ch):
        super(block_1, self).__init__()
        #self.red_conv = nn.Conv2d(ch//2, ch_in//2, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch//2, ch//4, kernel_size=1, stride=1, padding=0, bias=True),
            #nn.GroupNorm(8, ch_out),
            nn.BatchNorm2d(ch//4),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch//4, ch//4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch//4),
            nn.ELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch//4, ch//4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch//4),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch//2, ch//4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch//4),
            nn.ELU()
        )
        self.activation = nn.ELU()

    def forward(self, x):
        #red_x = self.red_conv(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        res = self.conv4(x)
        out = self.activation(out + res)
        return out
    
class block_2(nn.Module):
    def __init__(self, ch):
        super(block_2, self).__init__()
        #self.up_conv = nn.Conv2d(ch_in, ch_in*2, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch*2, ch//4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch//4),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch//4, ch//4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch//4),
            nn.ELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch//4, ch//4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch//4),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch*2, ch//4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch//4),
            nn.ELU()
        )
        self.activation = nn.ELU()

    def forward(self, x):
        #up_x = self.up_conv(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        res = self.conv4(x)
        out = self.activation(out + res)
        return out
    
class block_3(nn.Module):
    def __init__(self, ch):
        super(block_3, self).__init__()
        #self.red_conv = nn.Conv2d(ch_in, ch_in//4, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch//4, ch//4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch//4),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch//4, ch//4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch//4),
            nn.ELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch//4, ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch//4, ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch),
            nn.ELU()
        )
        self.activation = nn.ELU()

    def forward(self, x):
        #red_x = self.red_conv(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        res = self.conv4(x)
        out = self.activation(out + res)
        return out

class DRUNet(nn.Module):
    def __init__(self, input_shape=(1, 1, 256, 256), dims=32):
        super(DRUNet, self).__init__()
                
        self.first_conv = conv_layer(1, 16)
        self.block_c32 = block_1(dims)
        self.block_c32_3 = block_3(dims)

        self.block_c64 = block_1(dims * 2)
        self.block2_c64_3 = block_3(dims * 2)

        self.block_c128 = block_1(dims * 4)
        self.block_c128_3 = block_3(dims * 4)

        self.block_c256 = block_1(dims * 8)
        self.block_c256_3 = block_3(dims * 8)

        self.block_c512 = block_1(dims * 16)
        self.block_c512_3 = block_3(dims * 16)

        self.block_c1024 = block_1(dims * 32)
        self.block_c1024_3 = block_3(dims * 32)

        self.up_conv_c1024 = nn.ConvTranspose2d(dims * 32, dims * 16, kernel_size=2, stride=2)
        self.up_c512 = block_2(dims * 16)
        self.up_c512_3 = block_3(dims * 16)

        self.up_conv_c512 = nn.ConvTranspose2d(dims * 16, dims * 8, kernel_size=2, stride=2)
        self.up_c256 = block_2(dims * 8)
        self.up_c256_3 = block_3(dims * 8)

        self.up_conv_c256 = nn.ConvTranspose2d(dims * 8, dims * 4, kernel_size=2, stride=2)
        self.up_c128 = block_2(dims * 4)
        self.up_c128_3 = block_3(dims * 4)

        self.up_conv_c128 = nn.ConvTranspose2d(dims * 4, dims * 2, kernel_size=2, stride=2)
        self.up_c64 = block_2(dims * 2)
        self.up_c64_3 = block_3(dims * 2)

        self.up_conv_c64 = nn.ConvTranspose2d(dims * 2, dims * 1, kernel_size=2, stride=2)
        self.up_c32 = block_2(dims)
        self.up_c32_3 = block_3(dims)

        self.final_conv = nn.Conv2d(dims, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.first_conv(x) # (B,1,256,256) -> (B,16,256,256)
        out1 = self.block_c32(out) # (B,16,256,256) -> (B, 8, 256, 256)
        out1_3 = self.block_c32_3(out1) # (B,8,256,256) -> (B, 32, 256, 256)
        out = F.max_pool2d(out1_3, 2, 2) # (B, 32, 256, 256) -> (B, 32, 128, 128)
        
        out2 = self.block_c64(out) # (B,32,256,256) -> (B, 64, 128, 128)
        out2_3 = self.block2_c64_3(out2)
        out = F.max_pool2d(out2_3, 2, 2)

        out3 = self.block_c128(out)
        out3_3 = self.block_c128_3(out3)
        out = F.max_pool2d(out3_3, 2, 2)

        out4 = self.block_c256(out)
        out4_3 = self.block_c256_3(out4)
        out = F.max_pool2d(out4_3, 2, 2)

        out5 = self.block_c512(out)
        out5_3 = self.block_c512_3(out5)
        out = F.max_pool2d(out5_3, 2, 2)

        out = self.block_c1024(out)
        out = self.block_c1024_3(out)

        out = self.up_conv_c1024(out)
        out = torch.cat([out, out5_3], dim=1) # are these the right outs to be concatenating
        out = self.up_c512(out)
        out = self.up_c512_3(out)

        out = self.up_conv_c512(out)
        out = torch.cat([out, out4_3], dim=1)
        out = self.up_c256(out)
        out = self.up_c256_3(out)

        out = self.up_conv_c256(out)
        out = torch.cat([out, out3_3], dim=1)
        out = self.up_c128(out)
        out = self.up_c128_3(out)

        out = self.up_conv_c128(out)
        out = torch.cat([out, out2_3], dim=1)
        out = self.up_c64(out)
        out = self.up_c64_3(out)

        out = self.up_conv_c64(out)
        out = torch.cat([out, out1_3], dim=1)
        out = self.up_c32(out)
        out = self.up_c32_3(out)

        out = self.final_conv(out)
        out = torch.sigmoid(out)

        return out