import torch
import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self, init_type = "normal", gain = 0.02):

        def init_func(module):
            classname = module.__class__.__name__
                
            if hasattr(module, "weight") and (classname.find("Linear") != -1 or classname.find("Conv") != -1):
                if init_type == "normal":
                    nn.init.normal_(module.weight.data, mean = 0.0, std = gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(module.weight.data, gain = gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(module.weight.data, a = 0, mode = "fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(module.weight.data, gain = gain)
                
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)

            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(module.weight.data, 1.0, gain = gain)
                nn.init.constant_(module.bias.data, 0.0)

        self.apply(init_func)

class SRGenerator(BaseNet):
    def __init__(self, scale = 4, residual_blocks = 8):
        super().__init__()

        self.grad_encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, padding = 0)),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True)
            )

        self.grad_encoder2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True)
            )

        self.grad_encoder3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(256, track_running_stats = False),
            nn.ReLU(True)
            )

        grad_blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            grad_blocks.append(block)

        self.grad_middle = nn.Sequential(*grad_blocks)

        self.grad_decoder1 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True)
            )

        self.grad_decoder2 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 256, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True)
            )

        self.grad_decoder3 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)),
            nn.InstanceNorm2d(32, track_running_stats = False),
            nn.ReLU(True)
            )

        self.grad_decoder4 = nn.Conv2d(in_channels = 32, out_channels = 3, kernel_size = 1, padding = 0)
            

        #==========================================

        self.sr_encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, padding = 0),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True)
            )

        self.sr_encoder2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True)
            )

        self.sr_encoder3 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(256, track_running_stats = False),
            nn.ReLU(True)
            )

        sr_blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            sr_blocks.append(block)

        self.sr_middle = nn.Sequential(*sr_blocks)

        self.sr_decoder1 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(128, track_running_stats = False),
            nn.ReLU(True)
            )

        self.sr_decoder2 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(64, track_running_stats = False),
            nn.ReLU(True)
            )

        self.sr_decoder3 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 4, stride = 2, padding = 1)),
            nn.InstanceNorm2d(32, track_running_stats = False),
            nn.ReLU(True)
            )

        self.sr_decoder4 = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, padding = 1)

        self.init_weights()
    
    def forward(self, lr_images, lr_grads):
        feature1 = self.sr_encoder1(lr_images) #size:256, channel:64
        feature2 = self.sr_encoder2(feature1) #size:128, channel:128
        ouput = self.sr_encoder3(feature2) #size:64, channel:256
        
        ouput = self.sr_middle(output) #size:64, channel:256

        feature3 = self.sr_decoder1(output) #size:128, channel:128
        feature4 = self.sr_decoder2(feature3) #size:256, channel:64
        output = self.sr_decoder3(feature4) #size:256, channel:32

        grad = self.grad_encoder1(lr_grads) #size:256, channel:64
        ori_grad = grad
        grad = torch.cat((grad, feature1), dim=1) #channel:128
        grad = self.grad_encoder2(grad) #size:128, channel:128
        grad = torch.cat((grad, feature2), dim=1) #channel:256
        grad = self.grad_encoder3(grad) #size:64, channel:256

        grad = self.grad_middle(grad) #size:64, channel:256

        grad = self.grad_decoder1(grad) #size:128, channel:128
        grad = torch.cat((grad, feature3), dim=1) #channel:256
        grad = self.grad_decoder2(grad) #size:256, channel:64
        grad = ori_grad + grad
        grad = torch.cat((grad, feature4), dim=1) #channel:128
        grad_to_sr = self.grad_decoder3(grad) #size:256, channel:32
        grad = self.grad_decoder4(grad_to_sr) #size:256, channel:3
        final_grad = torch.sigmoid(grad) #channel = 3
        
        output = torch.cat((output, grad_to_sr), dim=1) #channel = 64
        output = self.sr_decoder4(output) #channel = 3
        output = torch.sigmoid(output)
        
        return output, final_grad

class GradDiscriminator(BaseNet):
    def __init__(self, in_channels, use_sigmoid = True):
        super().__init__()

        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 1, padding = 1, bias = False)),
            )

        self.conv5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.init_weights()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        output = conv5
        if self.use_sigmoid:
            output = nn.sigmoid(conv5)

        return output

class SRDiscriminator(BaseNet):
    def __init__(self, in_channels, use_sigmoid = True):
        super().__init__()

        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 1, padding = 1, bias = False)),
            )

        self.conv5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, True)
            )

        self.init_weights()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        output = conv5
        if self.use_sigmoid:
            output = nn.sigmoid(conv5)

        return output

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation = 1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.utils.spectral_norm(nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 3, padding = 0, dilation = dilation, bias = False)),
            nn.InstanceNorm2d(dim, track_running_stats = False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 3, padding = 0, dilation = 1, bias = False)),
            nn.InstanceNorm2d(dim, track_running_stats = False)
            )

    def forward(self, input):
        output = input + self.conv_block(input)

        return output


