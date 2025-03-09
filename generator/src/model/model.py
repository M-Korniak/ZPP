# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
# This file has been modified from the original Llama 3 source code.

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn

import src.transformations.transformations as transformations


@dataclass
class ModelArgs:
    dim: int = 32  # Play to determine the best value
    n_layers: int = 8
    n_heads: int = 8
    multiple_of: int = 64  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    rope_theta: float = 1000

    max_batch_size: int = 32
    max_seq_len: int = 258

    out_channel_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    paddings: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    scaling_factor: int = 2


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)

    Args:
    - dim (int): The input dimension
    - eps (float): A small value to prevent division by zero

    Attributes:
    - eps (float): A small value to prevent division by zero
    - weight (torch.nn.Parameter): The learnable weight parameter
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequencies for the rotary embeddings.

    Args:
    - dim (int): The input dimension
    - end (int): The end value
    - theta (float): The theta value

    Returns:
    - torch.Tensor: The precomputed frequencies of shape (end, dim)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape the frequencies for broadcasting.

    Args:
    - freqs_cis (torch.Tensor): The frequencies
    - x (torch.Tensor): The input tensor

    Returns:
    - torch.Tensor: The reshaped frequencies
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the rotary embeddings to the input tensors.

    Args:
    - xq (torch.Tensor): The query tensor
    - xk (torch.Tensor): The key tensor
    - freqs_cis (torch.Tensor): The frequencies

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: The query and key tensors with the rotary embeddings applied
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    Multi-head attention layer.

    Args:
    - args (ModelArgs): The model arguments

    Attributes:
    - n_heads (int): The number of heads
    - head_dim (int): The dimension of each head
    - wq (nn.Linear): The query weight matrix
    - wk (nn.Linear): The key weight matrix
    - wv (nn.Linear): The value weight matrix
    - wo (nn.Linear): The output weight matrix
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Feed-forward layer with Swish-Gated Linear Unit (SwiGLU) activation.

    Args:
    - dim (int): The input dimension
    - hidden_dim (int): The hidden dimension
    - multiple_of (int): The multiple of the hidden dimension

    Attributes:
    - w1 (nn.Linear): The first weight matrix
    - w2 (nn.Linear): The second weight matrix
    - w3 (nn.Linear): The third weight matrix
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.

    Args:
    - layer_id (int): The layer ID
    - args (ModelArgs): The model arguments

    Attributes:
    - n_heads (int): The number of heads
    - dim (int): The input dimension
    - head_dim (int): The dimension of each head
    - attention (Attention): The attention layer
    - feed_forward (FeedForward): The feed-forward layer
    - layer_id (int): The layer ID
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """
    Transformer model with multiple model blocks.

    Args:
    - params (ModelArgs): The model arguments

    Attributes:
    - params (ModelArgs): The model arguments
    - n_layers (int): The number of layers
    - layers (torch.nn.ModuleList): The list of model blocks
    - norm (RMSNorm): The RMSNorm layer
    - freqs_cis (torch.Tensor): The precomputed frequencies
    """
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def forward(self, input_tensor: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen, _ = input_tensor.shape
        h = input_tensor
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=input_tensor.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=input_tensor.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = h.float()
        return output


class Encoder(nn.Module):
    """
    Encoder model with convolutional and fully connected layers. Takes input of shape (B, S, C, H, W).

    Args:
    - args (ModelArgs): The model arguments

    Attributes:
    - latent_dim (int): The latent dimension
    - n_conv_layers (int): The number of convolutional layers
    - conv_layers (nn.ModuleList): The list of convolutional layers
    - batch_norm_layers (nn.ModuleList): The list of batch normalization layers
    - fc (nn.Linear): The fully connected layer
    - after_conv_shape (torch.Size): The shape after the convolutional layers
    - scaling_factor (int): The scaling factor for the convolutional layers
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.latent_dim = args.dim
        self.n_conv_layers = len(args.out_channel_sizes)

        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.scaling_factor = args.scaling_factor
        for i in range(self.n_conv_layers):
            in_channels = 3 if i == 0 else args.out_channel_sizes[i - 1]

            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=args.out_channel_sizes[i],
                    kernel_size=args.kernel_sizes[i],
                    stride=args.strides[i],
                    padding=args.paddings[i],
                )
            )

            self.batch_norm_layers.append(
                nn.BatchNorm2d(args.out_channel_sizes[i])
            )

        self.fc = None  # Will be initialized with the first forward pass
        self.after_conv_shape = None

    def get_after_conv_shape(self) -> torch.Size:
        return self.after_conv_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, S, C, H, W) -> (B * S, C, H, W)
        B, S = x.shape[:2]
        x = x.reshape(B * S, *x.shape[2:])
        for conv_layer, batch_layer in zip(self.conv_layers, self.batch_norm_layers):
            x = conv_layer(x)
            x = batch_layer(F.relu(x))
            x = F.max_pool2d(x, self.scaling_factor, self.scaling_factor)

        if self.after_conv_shape is None:
            self.after_conv_shape = x.shape[1:]

        x = torch.flatten(x, 1)
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], self.latent_dim).to(x.device)

        x = self.fc(x)

        return x.view(B, S, -1)


class Decoder(nn.Module):
    """
    Decoder model with deconvolutional layers. Takes input of shape (B, S, C, H, W).

    Args:
    - args (ModelArgs): The model arguments

    Attributes:
    - latent_dim (int): The latent dimension
    - n_conv_layers (int): The number of convolutional layers
    - deconv_layers (nn.ModuleList): The list of deconvolutional layers
    - batch_norm_layers (nn.ModuleList): The list of batch normalization layers
    - after_conv_shape (torch.Size): The shape after the convolutional layers
    - fc (nn.Linear): The fully connected layer
    - scaling_factor (int): The scaling factor for the deconvolutional layers
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.latent_dim = args.dim
        self.n_conv_layers = len(args.out_channel_sizes)

        self.after_conv_shape = None  # To be set based on the encoder
        self.fc = None  # To be set based on the encoder

        self.deconv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.scaling_factor = args.scaling_factor

        for i in range(self.n_conv_layers - 1, 0, -1):
            in_channels = args.out_channel_sizes[i]
            out_channels = args.out_channel_sizes[i - 1]

            self.deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=args.kernel_sizes[i] + 1,
                    stride=self.scaling_factor,
                    padding=args.paddings[i],
                )
            )

            self.batch_norm_layers.append(
                nn.BatchNorm2d(out_channels)
            )

        self.last_conv_layer = nn.ConvTranspose2d(
            in_channels=args.out_channel_sizes[0],
            out_channels=3,
            kernel_size=args.kernel_sizes[0] + 1,
            stride=self.scaling_factor,
            padding=args.paddings[0],
        )

    def set_after_conv_shape(self, after_conv_shape: torch.Size):
        self.after_conv_shape = after_conv_shape
        self.fc = (nn.Linear(self.latent_dim, int(torch.prod(torch.Tensor(list(after_conv_shape)))))
                   .to(next(self.parameters()).device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, S, H) -> (B * S, H)
        B, S = x.shape[:2]
        x = x.view(B * S, *x.shape[2:])
        x = self.fc(x)
        x = x.view(-1, *self.after_conv_shape)

        for deconv_layer, batch_layer in zip(self.deconv_layers, self.batch_norm_layers):
            x = deconv_layer(x)
            x = batch_layer(F.relu(x))

        x = self.last_conv_layer(x)

        return x.view(B, S, *x.shape[1:])


class AutoEncoder(nn.Module):
    """
    Autoencoder model.

    Args:
    - args (ModelArgs): The model arguments

    Attributes:
    - encoder (Encoder): The encoder model
    - decoder (Decoder): The decoder model
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

        self.decoder_init = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        if not self.decoder_init:
            self.decoder.set_after_conv_shape(self.encoder.get_after_conv_shape())
            self.decoder_init = True

        x = self.decoder(x)

        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Spatio-temporal transformer model.

    Args:
    - args (ModelArgs): The model arguments

    Attributes:
    - encoder (Encoder): The encoder model
    - transformer (Transformer): The transformer model
    - decoder (Decoder): The decoder model
    - decoder_init (bool): Whether the decoder has been initialized
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.encoder = Encoder(args)
        self.transformer = Transformer(args)
        self.decoder = Decoder(args)

        self.decoder_init = False

    def set_decoder_init(self, decoder_init: bool):
        self.decoder_init = decoder_init

    def get_encoder_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_encoder_transformer_space(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.transformer(x)
        if not self.decoder_init:
            self.decoder.set_after_conv_shape(self.encoder.get_after_conv_shape())
            self.decoder_init = True

        x = self.decoder(x)

        return x


if __name__ == "__main__":
    args = ModelArgs()
    model = SpatioTemporalTransformer(args)
    frames = transformations.transform_gif_to_tensor()
    frames = transformations.transform_image_to_trainable_form(frames)
    print(frames.shape)
    output = model(frames)
    print(output.shape)



