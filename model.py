# model.py
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from timm.models.swin_transformer_v2 import WindowAttention

def create_custom_swin_model():
    # Load the pretrained Swin Transformer model
    swin_model = timm.create_model('swinv2_tiny_window8_256', pretrained=True)
    swin_model.patch_embed.img_size = (224, 224)
    swin_model.patch_embed.strict_img_size = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    swin_model = swin_model.to(device)

    # Modify WindowAttention modules to include uncertainty branch
    for name, module in swin_model.named_modules():
        if isinstance(module, WindowAttention):
            # Add uncertainty branch
            module.uncertainty_branch = nn.Sequential(
                nn.LayerNorm(module.dim),
                nn.Linear(module.dim, module.num_heads),
                nn.Sigmoid()
            ).to(device)

            # Define the new forward method
            def new_forward(self, x, mask=None):
                B_, N, C = x.shape
                qkv_bias = None
                if self.q_bias is not None:
                    qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
                qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
                qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Cosine attention
                attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
                max_value = torch.log(torch.tensor(1. / 0.01, device=x.device))
                logit_scale = torch.clamp(self.logit_scale, max=max_value).exp()
                attn = attn * logit_scale

                relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
                relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
                attn = attn + relative_position_bias.unsqueeze(0)

                if mask is not None:
                    nW = mask.shape[0]
                    attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, self.num_heads, N, N)
                    attn = self.softmax(attn)
                else:
                    attn = self.softmax(attn)

                # Compute uncertainty scores and modulate attention
                uncertainty_scores = self.uncertainty_branch(x)  # (B_, N, num_heads)
                uncertainty_scores = uncertainty_scores.permute(0, 2, 1)  # (B_, num_heads, N)
                uncertainty_factors = uncertainty_scores * 1.0 + 0.5
                uncertainty_factors = uncertainty_factors.unsqueeze(-1)  # (B_, num_heads, N, 1)
                attn = attn * uncertainty_factors

                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x

            # Replace the forward method
            module.forward = types.MethodType(new_forward, module)

    print(f"SRA enhancement applied to {swin_model.__class__.__name__} on {device}")


    swin_model.load_state_dict(torch.load('swin_sra_final_model.pth',map_location=torch.device('cpu')))
    return swin_model
