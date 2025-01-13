import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
t1 = "Bright objects, High-contrast objects, Main foreground objects, Distinctive regions, Objects with sharp edges, Areas with strong color contrast, Visually dominant areas"
t2 = "Blended objects, Background-matching patterns, Objects with camouflage, Low-contrast objects, Regions with subtle texture changes, Hidden shapes in cluttered environments, Objects mimicking surroundings"
t3 = "Marine animals, Fish with distinctive patterns, Underwater creatures, Sea organisms with unique shapes, Coral and aquatic plants, Ocean wildlife, Fluorescent sea creatures"
t4 = "Shiny surfaces, Reflective regions, Bright mirrored areas, Objects with glossy finishes, High-gloss materials, Light-bouncing surfaces, Mirror-like reflections"
t5 = "Polyps in medical images, Irregular growths on tissue, Smooth or lobulated lesions, Abnormal structures in scans, Rounded or oval shapes, Bright or dark regions against normal tissue, Polyp outlines with soft edges, Contrast-enhanced abnormal regions"
t6 = "Outdoor shadows under sunlight, Shadows created by artificial lights, Urban shadows cast by buildings or trees,Faint shadows blending into textured surfaces, Shadows overlapping with dark regions, Complex interactions between shadow edges and object boundaries"

task_text=[t1,t2,t3,t4,t5]
ttt=None
index=None

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        """
        Basic Convolutional Block with optional BatchNorm and ReLU.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            kernel_size (int or tuple): Convolution kernel size.
            stride (int): Stride of the convolution.
            padding (int): Padding added to both sides of the input.
            dilation (int): Dilation rate for the convolution.
            groups (int): Number of blocked connections from input channels to output channels.
            relu (bool): If True, apply ReLU activation.
            bn (bool): If True, apply Batch Normalization.
            bias (bool): If True, adds a learnable bias to the output.
        """
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        """
        Feature Enhancement Module (FEM) with multi-branch structure.

        Args:
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
            stride (int): Stride of the down-sampling.
            scale (float): Scale factor for residual connection.
            map_reduce (int): Factor to reduce the number of intermediate channels.
        """
        super(FEM, self).__init__()
        self.scale = scale
        inter_planes = in_planes // map_reduce

        # Branch 0
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )

        # Branch 1
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        # Branch 2
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        # Fusion Layer
        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        # Shortcut Layer
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        # ReLU Activation
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """
        Forward pass of FEM.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Enhanced feature map.
        """
        # Compute branches
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # Concatenate branch outputs
        out = torch.cat((x0, x1, x2), 1)
        # Fuse features
        out = self.ConvLinear(out)
        # Add residual connection
        short = self.shortcut(x)
        out = out * self.scale + short
        # Apply activation
        out = self.relu(out)

        return out


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,input_resolution):
        super().__init__()
        self.size=(input_resolution,input_resolution)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    def forward(self, x1, x2,sx):
        sx=F.interpolate(sx,self.size,mode='bilinear')
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x1=x1*sx
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class MoE_Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(MoE_Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features  # 输入的维度
        self.num_experts = 5

        # Experts: Each expert is a separate feedforward network (you can change this to any task-specific sub-network)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()

        ) for _ in range(self.num_experts)])

        # Final layer to combine the output of the selected experts
        self.final_layer = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        self.text_to_gate = nn.Sequential(
            nn.Linear(512, 32),  # 映射到更低维特征空间
            nn.GELU(),
            nn.Linear(32, self.num_experts),    # 再映射到专家权重
        )
    def forward(self, x):
        # Compute gate values based on text features
        B,H,W,C=x.shape

        # gates = F.one_hot(index, num_classes=self.num_experts).float()  # Shape: [B, num_experts]

        # task_features = torch.randn(B, self.num_experts).to("cuda")  # 模拟任务特征
        gates = torch.softmax(self.text_to_gate(ttt)/0.9, dim=-1)  # Shape: [B, num_experts]
        # gates = torch.softmax(task_features / 0.5, dim=-1)

        # 混合任务索引生成的权重
        index_gates = F.one_hot(index, num_classes=self.num_experts).float()
        alpha = 0.3  # 控制动态 gates 的比例
        gates = alpha * gates + (1 - alpha) * index_gates

        # 归一化
        gates = gates / gates.sum(dim=-1, keepdim=True)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, num_experts, H, W, C]

        # 聚合专家输出
        output = torch.sum(expert_outputs * gates.view(B, self.num_experts, 1, 1, 1), dim=1)  # [B, H, W, C]

        # 最终融合
        output = self.final_layer(output) + x
        output = self.block(output)

        return output


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class CLIPTextAdapter(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', hidden_dim=512, adapter_dim=256):

        super(CLIPTextAdapter, self).__init__()

        # Load CLIP Text Encoder and Tokenizer
        self.clip_text_model = CLIPTextModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        # Adapter Layer
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )

    def forward(self, text_list):

        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to("cuda")
        # Extract original CLIP text embeddings
        with torch.no_grad():
            clip_text_embeddings = self.clip_text_model(**inputs).pooler_output  # [B, hidden_dim]

        # Pass through Adapter
        adapted_text_features = self.adapter(clip_text_embeddings) + clip_text_embeddings # [B, hidden_dim]
        return adapted_text_features

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out

        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out
class SAM2UNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2UNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
            # text_model, preprocess = clip.load("ViT-B/32")
        else:
            model = build_sam2(model_cfg)
            # text_model, preprocess = clip.load("ViT-B/32")
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck

        self.encoder = model.image_encoder.trunk
        self.text_encoder = CLIPTextAdapter()

        for param in self.encoder.parameters():
            param.requires_grad = False

        blocks = []
        for idx, block in enumerate(self.encoder.blocks):
            if idx % 2 == 0:  # 偶数索引
                blocks.append(Adapter(block))
            else:  # 奇数索引
                blocks.append(MoE_Adapter(block))

        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        self.rfb1 = FEM(144, 64)
        self.rfb2 = FEM(288, 64)
        self.rfb3 = FEM(576, 64)
        self.rfb4 = FEM(1152, 64)
        self.up1 = (Up(128, 64,input_resolution=22))
        self.up2 = (Up(128, 64,input_resolution=44))
        self.up3 = (Up(128, 64,input_resolution=88))
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)
        self.ca1=ChannelAttention(64)
        self.ca2=ChannelAttention(64)
        self.ca3=ChannelAttention(64)
        self.sa=SpatialAttention()

    def forward(self, x, text_prompts):
        B, C, H, W = x.shape
        text_features = self.text_encoder(text_prompts).to(torch.float32)
        task_text_features = self.text_encoder(task_text).to(torch.float32)
        global index
        global ttt
        ttt,index=self.replace_with_task_features(text_features,task_text_features)

        x1, x2, x3, x4 = self.encoder(x)

        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        # sa=self.sa(x1)
        sx=self.sa(x1)
        cx3=self.ca3(x4)
        cx2=self.ca2(x4)
        cx1=self.ca1(x4)
        # torch.Size([4, 1, 88, 88])
        # torch.Size([4, 64, 1, 1])

        # text_loss,p_loss,n_loss=self.text_loss(text_features, x1, x2, x3, x4,task_text_features)

        x = self.up1(x4, x3,sx)

        x=x*cx3
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2,sx)

        x=x*cx2
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')

        x = self.up3(x, x1,sx)
        x=x*cx1
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2
    def replace_with_task_features(self,text_features, task_text_features):
        """
        替换用户输入的文本特征为对应的任务定义特征。

        Args:
        - text_features (torch.Tensor): 用户输入的文本特征 [B, 512]。
        - task_text_features (torch.Tensor): 任务定义的文本特征 [T, 512]。

        Returns:
        - replaced_features (torch.Tensor): 替换后的文本特征 [B, 512]。
        - task_indices (torch.Tensor): 每个输入文本对应的任务索引 [B]。
        """
        # 1. 归一化特征以计算余弦相似度
        text_features_normalized = F.normalize(text_features, dim=1)  # [B, 512]
        task_text_features_normalized = F.normalize(task_text_features, dim=1)  # [T, 512]

        # 2. 计算相似度矩阵 [B, T]
        similarity_matrix = torch.matmul(text_features_normalized, task_text_features_normalized.T)  # [B, T]

        # 3. 获取每个 B 对应的 T 的索引
        task_indices = similarity_matrix.argmax(dim=1)  # [B]

        # 4. 根据索引替换特征
        replaced_features = task_text_features[task_indices]  # [B, 512]

        return replaced_features, task_indices


from thop import profile

if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet().cuda()
        x = torch.randn(4, 3, 352, 352).cuda()
        text_prompts = [t1, t2, t3, t1]

        # Filter parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

        # Compute FLOPs and parameters
        flops, params = profile(model, (x, text_prompts))

        # Count only trainable parameters
        trainable_params_count = sum(p.numel() for p in trainable_params)

        print("FLOPs (in GFLOPs):", flops / 1e9)
        print("Total Params (in M):", params / 1e6)
        print("Trainable Params (in M):", trainable_params_count / 1e6)