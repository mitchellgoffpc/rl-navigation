import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange

torch.manual_seed(42)

class Downsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

  def forward(self, x):
    return self.conv(F.pad(x, (0,1,0,1), mode="constant", value=0))  # no asymmetric padding in torch conv, must do it ourselves

class Upsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))

class ResnetBlock(nn.Module):
  def __init__(self, in_channels, out_channels=None, dropout=0.0):
    super().__init__()
    out_channels = out_channels or in_channels
    self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
    self.dropout = nn.Dropout(dropout)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    if in_channels != out_channels:
      self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    else:
      self.nin_shortcut = nn.Identity()

  def forward(self, x):
    h = F.silu(self.norm1(x))
    h = self.conv1(h)
    h = F.silu(self.norm2(h))
    h = self.dropout(h)
    h = self.conv2(h)
    return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    h = self.norm(x)
    q, k, v = self.q(h), self.k(h), self.v(h)

    # compute attention
    B, C, H, W = q.shape
    q = q.reshape(B, C, H*W)
    q = q.permute(0, 2, 1)   # B,HW,C
    k = k.reshape(B, C, H*W) # B,C,HW
    w = torch.bmm(q, k)     # B,HW,HW
    w = w * (C ** -0.5)
    w = F.softmax(w, dim=2)

    # attend to values
    v = v.reshape(B, C, H*W)
    w = w.permute(0, 2, 1)   # B,HW,HW (first hw of k, second of q)
    h = torch.bmm(v, w)     # B,C,HW (hw of q)
    h = h.reshape(B, C, H, W)

    return x + self.proj_out(h)


class Encoder(nn.Module):
  BLOCKS = [
    (128, 128, False, True),
    (128, 128, False, True),
    (128, 256, False, True),
    (256, 256, False, True),
    (256, 512, True, False)]

  def __init__(self, in_channels=3, z_channels=256, num_res_blocks=2, dropout=0.0):
    super().__init__()

    # downsample
    self.conv_in = nn.Conv2d(in_channels, Encoder.BLOCKS[0][0], kernel_size=3, stride=1, padding=1)
    self.layers = nn.ModuleList()
    for in_channels, out_channels, use_attn, use_downsample in Encoder.BLOCKS:
      layer = nn.Module()
      layer.blocks = nn.ModuleList()
      layer.attn = nn.ModuleList()
      for _ in range(num_res_blocks):
        layer.blocks.append(ResnetBlock(in_channels=in_channels, out_channels=out_channels, dropout=dropout))
        layer.attn.append(AttnBlock(out_channels) if use_attn else nn.Identity())
        in_channels = out_channels
      layer.downsample = Downsample(out_channels) if use_downsample else nn.Identity()
      self.layers.append(layer)

    # middle
    self.mid_block_1 = ResnetBlock(in_channels=in_channels, out_channels=in_channels, dropout=dropout)
    self.mid_attn_1 = AttnBlock(in_channels)
    self.mid_block_2 = ResnetBlock(in_channels=in_channels, out_channels=in_channels, dropout=dropout)

     # end
    self.norm_out = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    self.conv_out = nn.Conv2d(in_channels, z_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    x = x.permute(0,3,1,2)

    # downsampling
    h = self.conv_in(x)
    for layer in self.layers:
      for block, attn in zip(layer.blocks, layer.attn):
        h = attn(block(h))
      h = layer.downsample(h)

    # middle
    h = self.mid_block_1(h)
    h = self.mid_attn_1(h)
    h = self.mid_block_2(h)

    # end
    h = F.silu(self.norm_out(h))
    h = self.conv_out(h)
    return h


class Decoder(nn.Module):
  BLOCKS = [
    (512, 512, True, True),
    (512, 256, False, True),
    (256, 256, False, True),
    (256, 128, False, True),
    (128, 128, False, False)]

  def __init__(self, ch=128, out_ch=3, ch_mult=(1,1,2,2,4), num_res_blocks=2,
                     attn_resolutions=[16], dropout=0.0, in_channels=3,
                     resolution=256, z_channels=256):
    super().__init__()
    block_in = Decoder.BLOCKS[0][0]
    self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

    # middle
    self.mid_block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
    self.mid_attn_1 = AttnBlock(block_in)
    self.mid_block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

    # upsampling
    self.layers = nn.ModuleList()
    for block_in, block_out, use_attn, use_upsample in Decoder.BLOCKS:
      layer = nn.Module()
      layer.blocks = nn.ModuleList()
      layer.attn = nn.ModuleList()
      for _ in range(num_res_blocks + 1):
        layer.blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
        layer.attn.append(AttnBlock(block_out) if use_attn else nn.Identity())
        block_in = block_out
      layer.upsample = Upsample(block_out) if use_upsample else nn.Identity()
      self.layers.append(layer)

    # end
    self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
    self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

  def forward(self, z):
    h = self.conv_in(z)

    # middle
    h = self.mid_block_1(h)
    h = self.mid_attn_1(h)
    h = self.mid_block_2(h)

    # upsampling
    for layer in self.layers:
      for block, attn in zip(layer.blocks, layer.attn):
        h = attn(block(h))
      h = layer.upsample(h)

    h = F.silu(self.norm_out(h))
    h = self.conv_out(h)
    return h


class VectorQuantizer(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, commitment_cost):
    super().__init__()
    self.commitment_cost = commitment_cost
    self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

  def forward(self, inputs):
    B, C, H, W = inputs.shape
    flat_input = rearrange(inputs, 'B C H W -> (B H W) C')

    # Calculate distances
    distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

    # Encoding
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    quantized = self.embed(encoding_indices)
    quantized = rearrange(quantized, '(B H W) C -> B C H W', B=B, C=C, H=H, W=W).contiguous()
    encoding_indices = rearrange(encoding_indices, '(B H W) 1 -> B H W', B=B, H=H, W=W)

    # Loss
    e_latent_loss = (quantized.detach() - inputs).square().mean(dim=(1,2,3))
    q_latent_loss = (quantized - inputs.detach()).square().mean(dim=(1,2,3))
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    quantized = inputs + (quantized - inputs).detach()
    return quantized, encoding_indices, loss

  def embed(self, encoding_indices):
    encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=encoding_indices.device)
    encodings.scatter_(1, encoding_indices, 1)
    quantized = torch.matmul(encodings, self.embedding.weight)
    return quantized


class VQModel(nn.Module):
  def __init__(self, z_channels=256, vocab_size=1024):
    super().__init__()
    self.encoder = Encoder(z_channels=z_channels)
    self.quant_conv = nn.Conv2d(z_channels, z_channels, 1)
    self.quantize = VectorQuantizer(vocab_size, z_channels, commitment_cost=4.0)
    self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
    self.decoder = Decoder(z_channels=z_channels)

  def forward(self, x):
    x = self.encoder(x)
    x = self.quant_conv(x)
    x, encoding_indices, emb_loss = self.quantize(x)
    x = self.post_quant_conv(x)
    x = self.decoder(x)
    return x
