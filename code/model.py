from import_library import *

def cal_tsm(m): 
    sim = torch.cdist(m, m, p=2, compute_mode="use_mm_for_euclid_dist_if_necessary") ** 2
    
    sim_min, sim_max = sim.min(), sim.max()
    sim_range = sim_max - sim_min
    
    sim = (sim - sim_min)/(sim_range + 1e-8)
    sim = 1 - sim
    
    return sim

## Residual CNN + Dropout ## Reference: wide-dropout (https://cumulu-s.tistory.com/35)
class ConvBlock(nn.Module):
    def __init__(self, activation, in_channel, out_channel, dimension, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='zeros'):
        super(ConvBlock, self).__init__()
        
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Mish':
            self.activation = nn.Mish()
        elif activation == 'SELU':
            self.activation = nn.SELU()
        elif activation == 'SiLU': # = swish
            self.activation = nn.SiLU()
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU()

        if dimension == 1:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, dilation, padding_mode=padding_mode),
                nn.BatchNorm1d(out_channel),
                self.activation
            )
        elif dimension == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, padding_mode=padding_mode),
                nn.BatchNorm2d(out_channel),
                self.activation
            )

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, activation, in_channels, out_channels, dimension=1, res_droprate=0.3):
        super(ResBlock, self).__init__()
        
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Mish':
            self.activation = nn.Mish()
        elif activation == 'SELU':
            self.activation = nn.SELU()
        elif activation == 'SiLU': 
            self.activation = nn.SiLU()
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU()

        if dimension == 1:
            self.res = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
                self.activation,
                nn.Dropout(res_droprate),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
            )
            self.bn = nn.BatchNorm1d(out_channels)
    
            self.conv1x1 = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                self.activation,
                nn.Dropout(res_droprate),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
            )
            self.bn = nn.BatchNorm2d(out_channels)
    
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        x = self.res(x)
        
        if x.shape[1] == identity.shape[1]:
            x += identity
        else:
            x += self.conv1x1(identity)

        x = self.activation(self.bn(x))
        return x
    
class ReSEBlock(nn.Module): ## Reference: https://github.com/tae-jun/resemul/blob/master/model.py
    def __init__(self, activation, in_channels, out_channels, dimension=1, res_droprate=0.3):
        super(ReSEBlock, self).__init__()
        
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Mish':
            self.activation = nn.Mish()
        elif activation == 'SELU':
            self.activation = nn.SELU()
        elif activation == 'SiLU':
            self.activation = nn.SiLU()
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ELU':
            self.activation = nn.ELU()

        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.dimension == 1:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            
            self.res = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
                self.activation,
                nn.Dropout(res_droprate),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
            )
            self.res2 = nn.Sequential(
                nn.Linear(out_channels, int(out_channels//16)),
                self.activation,
                nn.Linear(int(out_channels//16), out_channels),
            )
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            
            self.res = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                self.activation,
                nn.Dropout(res_droprate),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
            )
            self.res2 = nn.Sequential(
                nn.Linear(out_channels, int(out_channels//16)),
                self.activation,
                nn.Linear(int(out_channels//16), out_channels),
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()
                

    def forward(self, x):
        identity = x # (batch, channel, length)

        if self.in_channels != self.out_channels:
            identity = self.shortcut(identity)

        x_res = self.res(x)
        identity2 = x_res # (batch, channel, length)

        if self.dimension == 1:
            x_res = self.avgpool(x_res).permute(0, 2, 1) # (batch, 1, channel)
            x_res = self.sigmoid(self.res2(x_res)).permute(0, 2, 1) # (batch, channel, 1)
        else:
            x_res = self.avgpool(x_res).flatten(2).permute(0, 2, 1)
            x_res = self.sigmoid(self.res2(x_res)).permute(0, 2, 1).unsqueeze(-1) # (batch, channel, 1, 1)

        out = identity2 * x_res
        out = identity + out
        out = self.activation(out)

        return out

class RPAM(nn.Module):
    def __init__(self, activation, max_frame_num, res_droprate=0.3):
        super(RPAM, self).__init__()

        self.rpam = nn.Sequential(
            ReSEBlock(activation, 1, 64, dimension=2, res_droprate=res_droprate),
            ReSEBlock(activation, 64, 64, dimension=2, res_droprate=res_droprate),
            nn.AdaptiveMaxPool2d((1, max_frame_num))
        )
    
    def forward(self, x):
        output = self.rpam(x)
        return output

class Resnet(nn.Module):
    def __init__(self, model_name, activation, max_frame_num, S, res_droprate):
        super(Resnet, self).__init__()
        
        self.model_name = model_name
        self.activation = activation
        self.max_frame_num = max_frame_num
        self.res_droprate = res_droprate
        self.S = S

        if self.model_name == 'resnet18':
            self.model = models.resnet18(weights=None)
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(weights=None)
        self.model = self.convert_2d21d(self.model) # Convert 2D to 1D

        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        
        ## Change first layer's in_channels 3 to 1
        if isinstance(self.conv1, nn.Conv1d):
            old_conv = self.conv1
            new_conv = nn.Conv1d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size[0],
                stride=old_conv.stride[0],
                padding=old_conv.padding[0],
                bias=old_conv.bias is not None
            )
            self.conv1 = new_conv

        ### Using Temporal Similarity Matrix (TSM)
        if self.max_frame_num:
            first_block = self.layer4[0]  # First BasicBlock

            # Add 64 channels
            new_conv1 = nn.Conv1d(
                in_channels=first_block.conv1.in_channels + 64, 
                out_channels=first_block.conv1.out_channels,
                kernel_size=first_block.conv1.kernel_size[0],
                stride=first_block.conv1.stride[0],
                padding=first_block.conv1.padding[0],
                bias=False
            )

            # Add 64 channels
            new_downsample = nn.Sequential(
                nn.Conv1d(first_block.downsample[0].in_channels + 64, first_block.downsample[0].out_channels, kernel_size=first_block.downsample[0].kernel_size[0], stride=first_block.downsample[0].stride[0], padding=first_block.downsample[0].padding[0], bias=False),
                first_block.downsample[1]
            )

            self.layer4[0].conv1 = new_conv1
            self.layer4[0].downsample = new_downsample

            self.adaptivepool_tsm = nn.AdaptiveMaxPool1d(self.max_frame_num)
            self.rpam = RPAM(self.activation, self.max_frame_num, self.res_droprate)

        self.adaptivepool = nn.AdaptiveMaxPool1d(self.S)

    def convert_2d21d(self, model):
        new_model = copy.deepcopy(model)
        torch.cuda.empty_cache()

        for name, module in new_model.named_children():
            # Convert Conv2d to Conv1d
            if isinstance(module, nn.Conv2d):
                new_conv = nn.Conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride[0],
                    padding=module.padding[0],
                    dilation=module.dilation[0],
                    groups=module.groups,
                    bias=module.bias is not None
                )
                setattr(new_model, name, new_conv)

            # Convert BatchNorm2d to BatchNorm1d
            elif isinstance(module, nn.BatchNorm2d):
                new_bn = nn.BatchNorm1d(
                    num_features=module.num_features,
                    eps=module.eps,
                    momentum=module.momentum,
                    affine=module.affine,
                    track_running_stats=module.track_running_stats
                )
                setattr(new_model, name, new_bn)

            # Convert MaxPool2d to MaxPool1d
            elif isinstance(module, nn.MaxPool2d):
                new_maxpool = nn.MaxPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_maxpool)

            # Convert AvgPool2d to AvgPool1d
            elif isinstance(module, nn.AvgPool2d):
                new_avgpool = nn.AvgPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_avgpool)

            # Convert AdaptiveAvgPool2d to AdaptiveAvgPool1d
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                new_adaptive_avgpool = nn.AdaptiveAvgPool1d(
                    output_size=module.output_size[0] if isinstance(module.output_size, tuple) else module.output_size
                )
                setattr(new_model, name, new_adaptive_avgpool)

            # Convert subsequential modules recursively
            elif len(list(module.children())) > 0:
                setattr(new_model, name, self.convert_2d21d(module))
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return new_model
    
    def forward(self, x):
        embeddings = self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x))))))) # (batch, 128, 750)
        
        if self.max_frame_num:
            embeddings = self.adaptivepool_tsm(embeddings) # (batch, 128, max_frame_num)

            ## Concatenate the output from 5th layer with the TSM representations
            tsm_ = cal_tsm(embeddings.permute(0, 2, 1)) # tsm (30, 30)
            tsm = self.rpam(tsm_.unsqueeze(1)).flatten(2)  # (64, 30)
            embeddings_tsm = torch.concat([tsm, embeddings], dim=1) # (batch, 192, max_frame_num)
            output = self.layer4(embeddings_tsm)
            output = self.adaptivepool(output)

            return output, embeddings, tsm_
        else:
            output = self.layer4(embeddings)
            output = self.adaptivepool(output)

            return output, embeddings, None

class M34_res(nn.Module):
    def __init__(self, activation, max_frame_num, S, res_droprate):
        super(M34_res, self).__init__()

        self.max_frame_num = max_frame_num
        self.activation = activation
        self.S = S
        self.res_droprate = 0.3


        self.layer1 = nn.Sequential(
            ConvBlock(self.activation, 1, 48, dimension=1, kernel_size=80, stride=4, padding=40),
            nn.MaxPool1d(kernel_size=4)
        ) 

        self.layer2 = nn.Sequential(
            ResBlock(self.activation, 48, 48, res_droprate=self.res_droprate),
            ResBlock(self.activation, 48, 48, res_droprate=self.res_droprate),
            ResBlock(self.activation, 48, 48, res_droprate=self.res_droprate),
            nn.MaxPool1d(kernel_size=4)
        ) 

        self.layer3 = nn.Sequential(
            ResBlock(self.activation, 48, 96, res_droprate=self.res_droprate),
            ResBlock(self.activation, 96, 96, res_droprate=self.res_droprate),
            ResBlock(self.activation, 96, 96, res_droprate=self.res_droprate),
            ResBlock(self.activation, 96, 96, res_droprate=self.res_droprate),
            nn.MaxPool1d(kernel_size=4)
        )

        if self.max_frame_num:
            self.layer4 = nn.Sequential(
                ResBlock(self.activation, 96, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
            )
            self.adaptivepool_tsm = nn.AdaptiveMaxPool1d(self.max_frame_num)
            self.rpam = RPAM(self.activation, self.max_frame_num, self.res_droprate)

            self.layer5 = nn.Sequential(
                ResBlock(self.activation, 192+64, 384, res_droprate=self.res_droprate),
                ResBlock(self.activation, 384, 384, res_droprate=self.res_droprate),
                ResBlock(self.activation, 384, 384, res_droprate=self.res_droprate),
            )

        else:
            self.layer4 = nn.Sequential(
                ResBlock(self.activation, 96, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
                ResBlock(self.activation, 192, 192, res_droprate=self.res_droprate),
                nn.MaxPool1d(kernel_size=4)
            )
            self.layer5 = nn.Sequential(
                ResBlock(self.activation, 192, 384, res_droprate=self.res_droprate),
                ResBlock(self.activation, 384, 384, res_droprate=self.res_droprate),
                ResBlock(self.activation, 384, 384, res_droprate=self.res_droprate),
            )

        self.adaptivepool = nn.AdaptiveMaxPool1d(self.S)

    def forward(self, x):
        # print(self.layer1(x).shape) # 3750
        embeddings = self.layer4(self.layer3(self.layer2(self.layer1(x)))) # 3750 -> 937 -> 234
        
        if self.max_frame_num:
            embeddings = self.adaptivepool_tsm(embeddings) # (batch, 96, max_frame_num)

            ## Concatenate the output from 3rd layer with the TSM representations
            tsm_ = cal_tsm(embeddings.permute(0, 2, 1)) # tsm (30, 30)
            tsm = self.rpam(tsm_.unsqueeze(1)).flatten(2)  # (64, 30)
            embeddings_tsm = torch.concat([tsm, embeddings], dim=1) # (batch, 160, max_frame_num)
            output = self.layer5(embeddings_tsm)
            output =self.adaptivepool(output)

            return output, embeddings, tsm_
        else:
            output = self.layer5(embeddings)
            output = self.adaptivepool(output)

            return output, embeddings, None

class VGG16(nn.Module):
    def __init__(self, activation, max_frame_num, S, res_droprate):
        super(VGG16, self).__init__()
        
        self.activation = activation
        self.max_frame_num = max_frame_num
        self.S = S
        self.res_droprate = res_droprate

        self.model = models.vgg16(weights=None)
        self.new_model = self.model.features # Remove avgpool and classifier layer
        self.new_model = self.convert_2d21d(self.new_model) # Convert 2d to 1d
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        ## Change first layer's in_channels 3 to 1
        if isinstance(self.new_model[0], nn.Conv1d):
            old_conv = self.new_model[0]
            new_conv = nn.Conv1d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size[0],
                stride=old_conv.stride[0],
                padding=old_conv.padding[0],
                bias=old_conv.bias is not None
            )
            self.new_model[0] = new_conv

        ### Using Temporal Similarity Matrix (TSM)
        if self.max_frame_num:
            new_conv1 = nn.Conv1d(
                in_channels=self.new_model[24].in_channels + 64, 
                out_channels=self.new_model[24].out_channels,
                kernel_size=self.new_model[24].kernel_size[0],
                stride=self.new_model[24].stride[0],
                padding=self.new_model[24].padding[0],
                bias=False
            )

            self.new_model[24] = new_conv1

            self.adaptivepool_tsm = nn.AdaptiveMaxPool1d(self.max_frame_num)
            self.rpam = RPAM(self.activation, self.max_frame_num, self.res_droprate)

        self.adaptivepool = nn.AdaptiveMaxPool1d(self.S)

    def convert_2d21d(self, model):
        new_model = copy.deepcopy(model)
        torch.cuda.empty_cache() 

        for name, module in new_model.named_children():
            # Convert Conv2d to Conv1d
            if isinstance(module, nn.Conv2d):
                new_conv = nn.Conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride[0],
                    padding=module.padding[0],
                    dilation=module.dilation[0],
                    groups=module.groups,
                    bias=module.bias is not None
                )
                setattr(new_model, name, new_conv)

            # Convert BatchNorm2d to BatchNorm1d
            elif isinstance(module, nn.BatchNorm2d):
                new_bn = nn.BatchNorm1d(
                    num_features=module.num_features,
                    eps=module.eps,
                    momentum=module.momentum,
                    affine=module.affine,
                    track_running_stats=module.track_running_stats
                )
                setattr(new_model, name, new_bn)

            # Convert MaxPool2d to MaxPool1d
            elif isinstance(module, nn.MaxPool2d):
                new_maxpool = nn.MaxPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_maxpool)

            # Convert AvgPool2d to AvgPool1d
            elif isinstance(module, nn.AvgPool2d):
                new_avgpool = nn.AvgPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_avgpool)

            # Convert AdaptiveAvgPool2d to AdaptiveAvgPool1d
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                new_adaptive_avgpool = nn.AdaptiveAvgPool1d(
                    output_size=module.output_size[0] if isinstance(module.output_size, tuple) else module.output_size
                )
                setattr(new_model, name, new_adaptive_avgpool)

            # Convert subsequential modules recursively
            elif len(list(module.children())) > 0:
                setattr(new_model, name, self.convert_2d21d(module))
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return new_model
    
    def forward(self, x):
        embeddings = self.new_model[:23](x) # (batch, 128, 750)
        
        if self.max_frame_num:
            embeddings = self.adaptivepool_tsm(embeddings) # (batch, 128, max_frame_num)

            ## Concatenate the output from the 5th layer with the TSM representations
            tsm_ = cal_tsm(embeddings.permute(0, 2, 1)) # tsm (30, 30)
            tsm = self.rpam(tsm_.unsqueeze(1)).flatten(2)  # (64, 30)
            embeddings_tsm = torch.concat([tsm, embeddings], dim=1) # (batch, 192, max_frame_num)
            output = self.new_model[24:](embeddings_tsm)
            output =self.adaptivepool(output)

            return output, embeddings, tsm_
        else:
            output = self.new_model[23:](embeddings)
            output = self.adaptivepool(output)

            return output, embeddings, None

class Dense121(nn.Module):
    def __init__(self, activation, max_frame_num, S, res_droprate):
        super(Dense121, self).__init__()
        
        self.activation = activation
        self.max_frame_num = max_frame_num
        self.S = S
        self.res_droprate = res_droprate

        self.model = models.densenet121(weights=None)
        self.new_model = self.model.features # Remove avgpool and classifier layer
        self.new_model = self.convert_2d21d(self.new_model) # Convert 2d to 1d
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        ## Change first layer's in_channels 3 to 1
        if isinstance(self.new_model[0], nn.Conv1d):
            old_conv = self.new_model[0]
            new_conv = nn.Conv1d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size[0],
                stride=old_conv.stride[0],
                padding=old_conv.padding[0],
                bias=old_conv.bias is not None
            )
            self.new_model[0] = new_conv

        ### Using Temporal Similarity Matrix (TSM)
        if self.max_frame_num:
            new_bn1 = nn.BatchNorm1d(
                num_features=self.new_model.denseblock4.denselayer1[2].in_channels + 64
            )
            new_conv1 = nn.Conv1d(
                in_channels=self.new_model.denseblock4.denselayer1[2].in_channels + 64, 
                out_channels=self.new_model.denseblock4.denselayer1[2].out_channels,
                kernel_size=self.new_model.denseblock4.denselayer1[2].kernel_size[0],
                stride=self.new_model.denseblock4.denselayer1[2].stride[0],
                padding=self.new_model.denseblock4.denselayer1[2].padding[0],
                bias=False
            )

            self.new_model.denseblock4.denselayer1[0] = new_bn1
            self.new_model.denseblock4.denselayer1[2] = new_conv1

            self.adaptivepool_tsm = nn.AdaptiveMaxPool1d(self.max_frame_num)
            self.rpam = RPAM(self.activation, self.max_frame_num, self.res_droprate)

        self.adaptivepool = nn.AdaptiveMaxPool1d(self.S)

    def convert_2d21d(self, model):
        new_model = copy.deepcopy(model)
        torch.cuda.empty_cache() 

        for name, module in new_model.named_children():
            # Convert Conv2d to Conv1d
            if isinstance(module, nn.Conv2d):
                new_conv = nn.Conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride[0],
                    padding=module.padding[0],
                    dilation=module.dilation[0],
                    groups=module.groups,
                    bias=module.bias is not None
                )
                setattr(new_model, name, new_conv)

            # Convert BatchNorm2d to BatchNorm1d
            elif isinstance(module, nn.BatchNorm2d):
                new_bn = nn.BatchNorm1d(
                    num_features=module.num_features,
                    eps=module.eps,
                    momentum=module.momentum,
                    affine=module.affine,
                    track_running_stats=module.track_running_stats
                )
                setattr(new_model, name, new_bn)

            # Convert MaxPool2d to MaxPool1d
            elif isinstance(module, nn.MaxPool2d):
                new_maxpool = nn.MaxPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_maxpool)

            # Convert AvgPool2d to AvgPool1d
            elif isinstance(module, nn.AvgPool2d):
                new_avgpool = nn.AvgPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_avgpool)

            # Convert AdaptiveAvgPool2d to AdaptiveAvgPool1d
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                new_adaptive_avgpool = nn.AdaptiveAvgPool1d(
                    output_size=module.output_size[0] if isinstance(module.output_size, tuple) else module.output_size
                )
                setattr(new_model, name, new_adaptive_avgpool)

            # Convert subsequential modules recursively
            elif len(list(module.children())) > 0:
                setattr(new_model, name, self.convert_2d21d(module))
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return new_model
    
    def forward(self, x):
        embeddings = self.new_model[:7](x) # (batch, 128, 750)
        
        if self.max_frame_num:
            embeddings = self.adaptivepool_tsm(self.new_model[7][:-1](embeddings)) # (batch, 512, max_frame_num)

            tsm_ = cal_tsm(embeddings.permute(0, 2, 1)) # tsm (30, 30)
            tsm = self.rpam(tsm_.unsqueeze(1)).flatten(2)  # (64, 30)
            embeddings_tsm = torch.concat([tsm, embeddings], dim=1) # (batch, 192, max_frame_num)
            output = self.new_model[8:](embeddings_tsm)
            output =self.adaptivepool(output)

            return output, embeddings, tsm_
        else:
            output = self.new_model[7:](embeddings)
            output = self.adaptivepool(output)

            return output, embeddings, None

class MobileNetV2(nn.Module):
    def __init__(self, activation, max_frame_num, S, res_droprate):
        super(MobileNetV2, self).__init__()
        
        self.activation = activation
        self.max_frame_num = max_frame_num
        self.S = S
        self.res_droprate = res_droprate

        self.model = models.mobilenet_v2(weights=None)
        self.new_model = self.model.features # Remove avgpool and classifier layer
        self.new_model = self.convert_2d21d(self.new_model) # Convert 2d to 1d
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        ## Change first layer's in_channels 3 to 1
        if isinstance(self.new_model[0][0], nn.Conv1d):
            old_conv = self.new_model[0][0]
            new_conv = nn.Conv1d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size[0],
                stride=old_conv.stride[0],
                padding=old_conv.padding[0],
                bias=old_conv.bias is not None
            )
            self.new_model[0][0] = new_conv

        ### Using Temporal Similarity Matrix (TSM)
        if self.max_frame_num:
            new_conv1 = nn.Conv1d(
                in_channels=self.new_model[18].conv[0][0].in_channels + 64, 
                out_channels=self.new_model[18].conv[0][0].out_channels,
                kernel_size=self.new_model[18].conv[0][0].kernel_size[0],
                stride=self.new_model[18].conv[0][0].stride[0],
                padding=self.new_model[18].conv[0][0].padding[0],
                bias=False
            )
            self.new_model[18].conv[0][0] = new_conv1

            self.adaptivepool_tsm = nn.AdaptiveMaxPool1d(self.max_frame_num)
            self.rpam = RPAM(self.activation, self.max_frame_num, self.res_droprate)

        self.adaptivepool = nn.AdaptiveMaxPool1d(self.S)

    def convert_2d21d(self, model):
        new_model = copy.deepcopy(model)
        torch.cuda.empty_cache() 

        for name, module in new_model.named_children():
            # Convert Conv2d to Conv1d
            if isinstance(module, nn.Conv2d):
                new_conv = nn.Conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride[0],
                    padding=module.padding[0],
                    dilation=module.dilation[0],
                    groups=module.groups,
                    bias=module.bias is not None
                )
                setattr(new_model, name, new_conv)

            # Convert BatchNorm2d to BatchNorm1d
            elif isinstance(module, nn.BatchNorm2d):
                new_bn = nn.BatchNorm1d(
                    num_features=module.num_features,
                    eps=module.eps,
                    momentum=module.momentum,
                    affine=module.affine,
                    track_running_stats=module.track_running_stats
                )
                setattr(new_model, name, new_bn)

            # Convert MaxPool2d to MaxPool1d
            elif isinstance(module, nn.MaxPool2d):
                new_maxpool = nn.MaxPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_maxpool)

            # Convert AvgPool2d to AvgPool1d
            elif isinstance(module, nn.AvgPool2d):
                new_avgpool = nn.AvgPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_avgpool)

            # Convert AdaptiveAvgPool2d to AdaptiveAvgPool1d
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                new_adaptive_avgpool = nn.AdaptiveAvgPool1d(
                    output_size=module.output_size[0] if isinstance(module.output_size, tuple) else module.output_size
                )
                setattr(new_model, name, new_adaptive_avgpool)

            # Convert subsequential modules recursively
            elif len(list(module.children())) > 0:
                setattr(new_model, name, self.convert_2d21d(module))
        
        del model
        gc.collect()
        torch.cuda.empty_cache()  

        return new_model
    
    def forward(self, x):
        embeddings = self.new_model[:18](x) # (batch, 128, 750)
        
        if self.max_frame_num:
            embeddings = self.adaptivepool_tsm(embeddings) # (batch, 512, max_frame_num)

            tsm_ = cal_tsm(embeddings.permute(0, 2, 1)) # tsm (30, 30)
            tsm = self.rpam(tsm_.unsqueeze(1)).flatten(2)  # (64, 30)
            embeddings_tsm = torch.concat([tsm, embeddings], dim=1) # (batch, 192, max_frame_num)
            output = self.new_model[18:](embeddings_tsm)
            output =self.adaptivepool(output)

            return output, embeddings, tsm_
        else:
            output = self.new_model[18:](embeddings)
            output = self.adaptivepool(output)

            return output, embeddings, None

class Efficientb0(nn.Module):
    def __init__(self, activation, max_frame_num, S, res_droprate):
        super(Efficientb0, self).__init__()
        
        self.activation = activation
        self.max_frame_num = max_frame_num
        self.S = S
        self.res_droprate = res_droprate

        self.model = models.efficientnet_b0(weights=None)
        self.new_model = self.model.features # Remove avgpool and classifier
        self.new_model = self.convert_2d21d(self.new_model) # Convert 2d to 1d
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        ## Change first layer's in_channels 3 to 1
        if isinstance(self.new_model[0][0], nn.Conv1d):
            old_conv = self.new_model[0][0]
            new_conv = nn.Conv1d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size[0],
                stride=old_conv.stride[0],
                padding=old_conv.padding[0],
                bias=old_conv.bias is not None
            )
            self.new_model[0][0] = new_conv

        ### Using Temporal Similarity Matrix (TSM)
        if self.max_frame_num:
            new_conv1 = nn.Conv1d(
                in_channels=self.new_model[8][0].in_channels + 64, 
                out_channels=self.new_model[8][0].out_channels,
                kernel_size=self.new_model[8][0].kernel_size[0],
                stride=self.new_model[8][0].stride[0],
                padding=self.new_model[8][0].padding[0],
                bias=False
            )
            self.new_model[8][0] = new_conv1

            self.adaptivepool_tsm = nn.AdaptiveMaxPool1d(self.max_frame_num)
            self.rpam = RPAM(self.activation, self.max_frame_num, self.res_droprate)

        self.adaptivepool = nn.AdaptiveMaxPool1d(self.S)

    def convert_2d21d(self, model):
        new_model = copy.deepcopy(model)
        torch.cuda.empty_cache() 

        for name, module in new_model.named_children():
            # Convert Conv2d to Conv1d
            if isinstance(module, nn.Conv2d):
                new_conv = nn.Conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size[0],
                    stride=module.stride[0],
                    padding=module.padding[0],
                    dilation=module.dilation[0],
                    groups=module.groups,
                    bias=module.bias is not None
                )
                setattr(new_model, name, new_conv)

            # Convert BatchNorm2d to BatchNorm1d
            elif isinstance(module, nn.BatchNorm2d):
                new_bn = nn.BatchNorm1d(
                    num_features=module.num_features,
                    eps=module.eps,
                    momentum=module.momentum,
                    affine=module.affine,
                    track_running_stats=module.track_running_stats
                )
                setattr(new_model, name, new_bn)

            # Convert MaxPool2d to MaxPool1d
            elif isinstance(module, nn.MaxPool2d):
                new_maxpool = nn.MaxPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_maxpool)

            # Convert AvgPool2d to AvgPool1d
            elif isinstance(module, nn.AvgPool2d):
                new_avgpool = nn.AvgPool1d(
                    kernel_size=module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                    stride=module.stride if isinstance(module.stride, int) else module.stride[0],
                    padding=module.padding if isinstance(module.padding, int) else module.padding[0],
                    ceil_mode=module.ceil_mode
                )
                setattr(new_model, name, new_avgpool)

            # Convert AdaptiveAvgPool2d to AdaptiveAvgPool1d
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                new_adaptive_avgpool = nn.AdaptiveAvgPool1d(
                    output_size=module.output_size[0] if isinstance(module.output_size, tuple) else module.output_size
                )
                setattr(new_model, name, new_adaptive_avgpool)

            # Convert subsequential modules recursively
            elif len(list(module.children())) > 0:
                setattr(new_model, name, self.convert_2d21d(module))
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return new_model
    
    def forward(self, x):
        embeddings = self.new_model[:8](x) # (batch, 128, 750)
        
        if self.max_frame_num:
            embeddings = self.adaptivepool_tsm(embeddings) # (batch, 512, max_frame_num)

            tsm_ = cal_tsm(embeddings.permute(0, 2, 1)) # tsm (30, 30)
            tsm = self.rpam(tsm_.unsqueeze(1)).flatten(2)  # (64, 30)
            embeddings_tsm = torch.concat([tsm, embeddings], dim=1) # (batch, 192, max_frame_num)
            output = self.new_model[8:](embeddings_tsm)
            output =self.adaptivepool(output)

            return output, embeddings, tsm_
        else:
            output = self.new_model[8:](embeddings)
            output = self.adaptivepool(output)

            return output, embeddings, None

class RespiDet(nn.Module):
    def __init__(self, activation, max_frame_num, S, res_droprate):
        super(RespiDet, self).__init__()

        self.max_frame_num = max_frame_num
        self.activation = activation
        self.S = S
        self.res_droprate = res_droprate

        self.layer1 = nn.Sequential(
            ConvBlock(activation, 1, 48, dimension=1, kernel_size=40, stride=2, padding=15),   # If 8kHz sampling rate, k-80 s-4 ==> 10ms mid frequency
            nn.MaxPool1d(kernel_size=8, stride=4)
        ) 

        self.layer2 = nn.Sequential(
            ReSEBlock(activation, 48, 48, res_droprate=res_droprate),
            ReSEBlock(activation, 48, 48, res_droprate=res_droprate),
            nn.MaxPool1d(kernel_size=4)
        ) 

        self.layer3 = nn.Sequential(
            ReSEBlock(activation, 48, 96, res_droprate=res_droprate),
            ReSEBlock(activation, 96, 96, res_droprate=res_droprate),
            nn.MaxPool1d(kernel_size=4)
        )

        if self.max_frame_num:
            self.layer4 = nn.Sequential(
                ReSEBlock(activation, 96, 192, res_droprate=res_droprate),
                ReSEBlock(activation, 192, 192, res_droprate=res_droprate),
            )
            self.adaptivepool_tsm = nn.AdaptiveMaxPool1d(self.max_frame_num)
            self.rpam = RPAM(self.activation, self.max_frame_num, self.res_droprate)

            self.layer5 = nn.Sequential(
                ReSEBlock(activation, 192+64, 384, res_droprate=res_droprate),
                ReSEBlock(activation, 384, 384, res_droprate=res_droprate),
                nn.AdaptiveMaxPool1d(self.S)
            )
        else:
            self.layer4 = nn.Sequential(
                ReSEBlock(activation, 96, 192, res_droprate=res_droprate),
                ReSEBlock(activation, 192, 192, res_droprate=res_droprate),
                nn.MaxPool1d(kernel_size=4)
            )
            self.layer5 = nn.Sequential(
                ReSEBlock(activation, 192, 384, res_droprate=res_droprate),
                ReSEBlock(activation, 384, 384, res_droprate=res_droprate),
                nn.AdaptiveMaxPool1d(self.S)
            )

        self.layer6 = nn.Sequential(
            ReSEBlock(activation, 384, 384, res_droprate=res_droprate),
            ReSEBlock(activation, 384, 384, res_droprate=res_droprate),
        )

    def forward(self, x):
        embeddings = self.layer4(self.layer3(self.layer2(self.layer1(x)))) # (48, 7498) -> (48, 1874) -> (96, 468) -> (192, 30) ->

        if self.max_frame_num:
            embeddings = self.adaptivepool_tsm(embeddings) # (batch, 192, max_frame_num)

            ## Concatenate the output from the 4th layer with the TSM representations
            tsm_ = cal_tsm(embeddings.permute(0, 2, 1)) # tsm (30, 30)
            tsm = self.rpam(tsm_.unsqueeze(1)).flatten(2)  # (64, 30)
            embeddings_tsm = torch.concat([tsm, embeddings], dim=1) # (batch, 160, max_frame_num)
            output = self.layer6(self.layer5(embeddings_tsm))

            return output, embeddings, tsm_
        else:
            output = self.layer6(self.layer5(embeddings))

            return output, embeddings, None

class LightRespiDet(nn.Module):
    def __init__(self, activation, max_frame_num, S, res_droprate):
        super(LightRespiDet, self).__init__()

        self.max_frame_num = max_frame_num
        self.activation = activation
        self.S = S
        self.res_droprate = res_droprate

        self.layer1 = nn.Sequential(
            ConvBlock(activation, 1, 32, dimension=1, kernel_size=40, stride=2, padding=20),   # If 8kHz sampling arte, k-80 s-4 ==> 10ms mid frequency
            nn.MaxPool1d(kernel_size=4)
        ) 

        self.layer2 = nn.Sequential(
            ReSEBlock(activation, 32, 32, res_droprate=res_droprate),
            ReSEBlock(activation, 32, 32, res_droprate=res_droprate),
            nn.MaxPool1d(kernel_size=4)
        ) 

        self.layer3 = nn.Sequential(
            ReSEBlock(activation, 32, 64, res_droprate=res_droprate),
            ReSEBlock(activation, 64, 64, res_droprate=res_droprate),
            nn.MaxPool1d(kernel_size=4)
        )

        self.layer4 = nn.Sequential(
            ReSEBlock(activation, 64, 128, res_droprate=res_droprate),
            ReSEBlock(activation, 128, 128, res_droprate=res_droprate),
            nn.MaxPool1d(kernel_size=4)
        )

        if self.max_frame_num:
            self.layer5 = nn.Sequential(
                ReSEBlock(activation, 128, 256, res_droprate=res_droprate),
                ReSEBlock(activation, 256, 256, res_droprate=res_droprate),
            )

            self.layer6 = nn.Sequential(
                ReSEBlock(activation, 256+64, 256, res_droprate=res_droprate),
                ReSEBlock(activation, 256, 256, res_droprate=res_droprate),
                nn.AdaptiveMaxPool1d(self.S)
            )

            self.adaptivepool_tsm = nn.AdaptiveMaxPool1d(self.max_frame_num)
            self.rpam = RPAM(self.activation, self.max_frame_num, self.res_droprate)
        else:
            self.layer5 = nn.Sequential(
                ReSEBlock(activation, 128, 256, res_droprate=res_droprate),
                ReSEBlock(activation, 256, 256, res_droprate=res_droprate),
                nn.MaxPool1d(kernel_size=4)
            )

            self.layer6 = nn.Sequential(
                ReSEBlock(activation, 256, 256, res_droprate=res_droprate),
                ReSEBlock(activation, 256, 256, res_droprate=res_droprate),
                nn.AdaptiveMaxPool1d(self.S)
            )

    def forward(self, x):
        embeddings = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))) # (32, 7499) -> (32, 3749) -> (64, 1874) -> (128, 937) -> (256, 467)

        if self.max_frame_num:
            embeddings = self.adaptivepool_tsm(embeddings) # (batch, 256, max_frame_num)

            ## Concatenate the output from the 4th layer with the TSM representations.
            tsm_ = cal_tsm(embeddings.permute(0, 2, 1)) # tsm (30, 30)
            tsm = self.rpam(tsm_.unsqueeze(1)).flatten(2)  # (64, 30)
            embeddings_tsm = torch.concat([tsm, embeddings], dim=1) # (batch, 320, max_frame_num)
            output = self.layer6(embeddings_tsm)

            return output, embeddings, tsm_
        else:
            output = self.layer6(embeddings)

            return output, embeddings, None

class DetectionHead(nn.Module):
    def __init__(self, in_channels, C):
        super(DetectionHead, self).__init__()

        self.in_channels = in_channels
        self.C = C

        self.class_predict = nn.Sequential(
            nn.Conv1d(self.in_channels, self.C, kernel_size=1, stride=1, padding=0), # 512 -> 4
        )
        self.conf_predict = nn.Sequential(
            nn.Conv1d(self.in_channels, 1, kernel_size=1, stride=1, padding=0), # 512 -> 4
        )
        self.coord_predict = nn.Sequential(
            nn.Conv1d(self.in_channels, 2, kernel_size=1, stride=1, padding=0), # 512 -> 4
        )
    
    def forward(self, x):
        class_predictions = self.class_predict(x) # (batch, self.C, self.S)
        conf_predictions = self.conf_predict(x) # (batch, 1, self.S)
        coord_predictions = self.coord_predict(x) # (batch, 2, self.S)

        object_predictions = torch.concat([class_predictions, conf_predictions, coord_predictions], dim=1).permute(0, 2, 1)
        return object_predictions

class Detector(nn.Module):
    def __init__(self, model_name, activation='LeakyReLU', max_frame_num=False, S=15, C=1, res_droprate=0.3, device="cuda:0"): # If max_frame_num == False, without RPAM
        super(Detector, self).__init__()

        if model_name == 'resnet18':
            head_channel = 512
            self.model = Resnet(model_name, activation, max_frame_num, S, res_droprate)
        elif model_name == 'resnet50':
            head_channel = 2048
            self.model = Resnet(model_name, activation, max_frame_num, S, res_droprate)
        elif model_name == 'm34_res':
            head_channel = 384
            self.model = M34_res(activation, max_frame_num, S, res_droprate)
        elif model_name == 'respidet':
            head_channel = 384
            self.model = RespiDet(activation, max_frame_num, S, res_droprate)
        elif model_name == 'lightrespidet':
            head_channel = 256
            self.model = LightRespiDet(activation, max_frame_num, S, res_droprate)
        elif model_name == 'vgg16':
            head_channel = 512
            self.model = VGG16(activation, max_frame_num, S, res_droprate)
        elif model_name == 'dense121':
            head_channel = 1024
            self.model = Dense121(activation, max_frame_num, S, res_droprate)
        elif model_name == 'mobilev2':
            head_channel = 1280
            self.model = MobileNetV2(activation, max_frame_num, S, res_droprate)
        elif model_name == 'efficientb0':
            head_channel = 1280
            self.model = Efficientb0(activation, max_frame_num, S, res_droprate)


        self.header = DetectionHead(head_channel, C)

        self.apply(self.init_weight)
        
        self.to(device)

    def forward(self, x):
        output, embeddings, emb_sim = self.model(x)

        object_predictions = self.header(output)

        return object_predictions, embeddings, emb_sim

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            # m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
            # m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
            # m.bias.data.fill_(0.0)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.0)