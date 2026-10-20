import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import unfold, fold
import math

# ==============================================================================
#  1. The New Meta-Architecture Wrapper
# ==============================================================================

class MetaBlock(nn.Module):
    """
    Wraps a pure operation with LayerNorm, residual, and FFN.
    Automatically adapts input channels if needed.
    """
    def __init__(self, operation, num_features):
        super(MetaBlock, self).__init__()
        self.operation = operation
        self.num_features = num_features
        
        # Get expected input/output channels from operation
        # Check multiple possible attribute names
        if hasattr(operation, 'in_channels'):
            self.expected_in_ch = operation.in_channels
        elif hasattr(operation, 'in_ch'):
            self.expected_in_ch = operation.in_ch
        elif hasattr(operation, 'conv') and hasattr(operation.conv, 'in_channels'):
            self.expected_in_ch = operation.conv.in_channels
        else:
            self.expected_in_ch = num_features
            
        if hasattr(operation, 'out_channels'):
            self.out_channels = operation.out_channels
        elif hasattr(operation, 'out_ch'):
            self.out_channels = operation.out_ch
        elif hasattr(operation, 'conv') and hasattr(operation.conv, 'out_channels'):
            self.out_channels = operation.conv.out_channels
        else:
            self.out_channels = num_features
        
        self.norm1 = nn.LayerNorm(num_features)
        self.batch_norm = nn.BatchNorm2d(self.out_channels)
        self.activation = nn.ReLU(inplace=True)

    def _get_input_adapter(self, in_ch, target_ch, device):
        """Get or create a 1x1 conv adapter for input channel mismatch."""
        if in_ch == target_ch:
            return None
        key = f"adapter_{in_ch}_to_{target_ch}"
        
        # Check if adapter exists as a registered submodule
        if not hasattr(self, key):
            adapter = nn.Conv2d(in_ch, target_ch, kernel_size=1, bias=True).to(device)
            # Register it as a submodule so it moves with the model
            setattr(self, key, adapter)
        
        return getattr(self, key)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Adapt input channels if needed
        adapter = self._get_input_adapter(C, self.expected_in_ch, x.device)
        if adapter is not None:
            x_adapted = adapter(x)
        else:
            x_adapted = x
        
        # Apply operation
        try:
            op_output = self.operation(x_adapted)
        except RuntimeError as e:
            # If operation fails, provide detailed error info
            raise RuntimeError(
                f"MetaBlock operation failed: {e}\n"
                f"Input shape: {x_adapted.shape}, Expected input channels: {self.expected_in_ch}\n"
                f"Operation type: {type(self.operation).__name__}"
            )
        
        B_out, C_out, H_out, W_out = op_output.shape
        
        # Flatten for LayerNorm
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        op_seq = op_output.flatten(2).transpose(1, 2)  # (B, H_out*W_out, C_out)
        
        # Apply norm1 and residual (only if original input channels match output)
        if C_out == C and H_out == H and W_out == W:
            # Only apply residual if spatial dims AND channels match
            normed_seq = self.norm1(x_seq + op_seq)
        else:
            # Skip residual if channels changed
            normed_seq = op_seq

        # Reshape back to spatial dims (use output spatial dimensions, not input)
        normed_tensor = normed_seq.transpose(1, 2).reshape(B_out, C_out, H_out, W_out)
        output = self.activation(self.batch_norm(normed_tensor))
        return output

# ==============================================================================
#  2. Simplified, "Pure" Operations
# ==============================================================================

class FFTBlock(nn.Module):
    """
    Frequency-domain filtering with optional channel projection.
    """
    def __init__(self, in_channels, out_channels=None):
        super(FFTBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.in_channels,
            kernel_size=3, padding=1, groups=self.in_channels, bias=True
        )
        self.pointwise = (
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
            if self.out_channels != self.in_channels else nn.Identity()
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        x_dw = self.depthwise_conv(x)
        fft_x = torch.fft.rfft2(x_dw, norm='backward')
        fft_mag = fft_x.abs()
        gate = torch.cos(2 * torch.pi * torch.sigmoid(fft_mag))
        filtered_fft_x = fft_x * gate
        output = torch.fft.irfft2(filtered_fft_x, s=(H, W), norm='backward')
        return self.pointwise(output)


import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    Standard Residual block with optional channel adjustment on the shortcut.
    """
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        
        # Main convolution path
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        # Shortcut projection if channel dimensions change
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection addition
        out += residual
        
        # Final non-linearity applied AFTER addition
        out = self.relu(out)
        return out


class AttentionBlock(nn.Module):
    """
    Spatial Windowed Multi-Head Self-Attention Block with Residual Connections.
    """
    def __init__(self, in_channels, num_heads=4, window_size=64, out_channels=None):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Projection for channel mismatch on shortcut/output
        if self.out_channels != self.in_channels:
            self.channel_proj = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.channel_proj = nn.Identity()

    def _apply_attention(self, x_seq):
        """Helper to apply LayerNorm + MultiheadAttention."""
        norm_x = self.norm(x_seq)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        return x_seq + attn_out  # Residual connection

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        # If spatial dimensions fit inside one window, run standard attention directly
        if H <= ws and W <= ws:
            x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            out_seq = self._apply_attention(x_seq)
            out = out_seq.transpose(1, 2).reshape(B, C, H, W)
            return self.channel_proj(out)

        # Pad to be divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_padded = x

        _, _, Hp, Wp = x_padded.shape

        # Window Partitioning: (B, C, Hp, Wp) -> (B * num_windows, ws*ws, C)
        # Reshape into non-overlapping patches cleanly without unfold/fold
        x_windows = x_padded.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1).contiguous()
        x_windows = x_windows.view(-1, ws * ws, C)

        # Apply Attention
        attn_windows = self._apply_attention(x_windows)

        # Reverse Window Partitioning: (B * num_windows, ws*ws, C) -> (B, C, Hp, Wp)
        num_windows_h = Hp // ws
        num_windows_w = Wp // ws
        out = attn_windows.view(B, num_windows_h, num_windows_w, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, Hp, Wp)

        # Crop back to original dimensions (if padded)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return self.channel_proj(out)


class MambaBlock(nn.Module):
    """
    2D Spatial Mamba (Selective SSM) Block.
    Uses padding='same' in Conv1d to prevent sequence length mismatches.
    """
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2, dt_rank="auto", out_channels=None):
        super(MambaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.d_state = d_state
        self.d_inner = int(expand * self.in_channels)
        self.dt_rank = int(torch.ceil(torch.tensor(self.in_channels / 16)).item()) if dt_rank == "auto" else dt_rank

        self.norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(self.in_channels, 2 * self.d_inner, bias=False)

        # Fix: padding='same' ensures output length strictly equals input length L
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding='same'
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.out_channels, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.shortcut = (
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
            if self.in_channels != self.out_channels else nn.Identity()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        residual = self.shortcut(x)

        # Reshape to sequence: (B, C, H, W) -> (B, L, C) where L = H * W
        x_seq = self.norm(x.flatten(2).transpose(1, 2))
        
        # Split into signal path (x_conv) and gate path (z)
        xz = self.in_proj(x_seq)
        x_conv, z = xz.chunk(2, dim=-1)

        # 1D Convolution with 'same' padding — sequence length remains L
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.conv1d(x_conv).transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective SSM
        y = self.ssm(x_conv)
        
        # Gating — both y and z now strictly match length L
        y = y * F.silu(z)

        # Project to output and reshape to 2D
        output_seq = self.out_proj(y)
        out = output_seq.transpose(1, 2).reshape(B, self.out_channels, H, W)

        return out + residual

    def ssm(self, x):
        B, L, d_inner = x.shape
        A = -torch.exp(self.A_log.float())

        # Dynamic parameter projections
        x_dbl = self.x_proj(x)
        dt_val, B_val, C_val = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_val))

        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for i in range(L):
            delta_i = delta[:, i, :]            # (B, d_inner)
            B_i = B_val[:, i, :]                # (B, d_state)
            C_i = C_val[:, i, :]                # (B, d_state)
            u_i = x[:, i, :]                    # (B, d_inner)

            # Discretization
            dA = torch.exp(delta_i.unsqueeze(-1) * A)             # (B, d_inner, d_state)
            dB_u = torch.einsum('bd,bs,bd->bds', delta_i, B_i, u_i)  # (B, d_inner, d_state)

            # Recurrent update
            h = dA * h + dB_u
            
            # Compute output
            y_i = torch.einsum('bds,bs->bd', h, C_i)              # (B, d_inner)
            ys.append(y_i)

        y = torch.stack(ys, dim=1)
        y = y + x * self.D
        return y
class GNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.linear = nn.Linear(
            self.in_channels,
            self.out_channels
        )

        self.local_proj = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                2 * self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        B, C, H, W = x.shape

        nodes = x.flatten(2).transpose(1, 2)
        transformed = self.linear(nodes)

        global_context = transformed.mean(dim=1)
        global_context = global_context.view(
            B, self.out_channels, 1, 1
        )
        global_context = global_context.expand(
            -1, -1, H, W
        )

        local_features = self.local_proj(x)

        fused = torch.cat(
            [local_features, global_context],
            dim=1
        )

        output = self.fusion(fused)

        return output + local_features



class ConvLSTMCell(nn.Module):
    """
    Processes a SINGLE time step (x_t) and propagates recurrent states (h, c).
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Single 2D conv computes all 4 gates (i, f, o, g) at once
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=True
        )

    def forward(self, x, state=None):
        """
        x: (B, C, H, W)
        state: Tuple of (h_prev, c_prev)
        """
        B, _, H, W = x.shape

        # Initialize zero state dynamically only when state is missing
        if state is None:
            h_prev = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Recurrent cell and hidden state updates
        c_next = (f * c_prev) + (i * g)
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class ConvLSTMBlock(nn.Module):
    """
    ConvLSTM block that seamlessly accepts both 4D spatial tensors (B, C, H, W)
    and 5D sequence tensors (B, T, C, H, W).
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super(ConvLSTMBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.cell = ConvLSTMCell(self.in_channels, self.out_channels, kernel_size)

    def forward(self, x, state=None):
        # 1. Detect if input is 4D (B, C, H, W) and adapt to 5D (B, 1, C, H, W)
        is_4d = (x.dim() == 4)
        if is_4d:
            x = x.unsqueeze(1)  # Adds dummy time dimension T=1

        # 2. Unpack safely (now guaranteed to be 5D)
        B, T, C, H, W = x.shape
        outputs = []

        for t in range(T):
            x_t = x[:, t, ...]
            h_next, state = self.cell(x_t, state)
            outputs.append(h_next)

        out = torch.stack(outputs, dim=1)  # Shape: (B, T, out_channels, H, W)

        # 3. If input was originally 4D, squeeze the dummy time dimension back out
        if is_4d:
            out = out.squeeze(1)  # Shape: (B, out_channels, H, W)

        return out


class downsampling_block(nn.Module):
    """
    A downsampling block using strided convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(downsampling_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Strided convolution updates channels AND reduces spatial dimension
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down(x)


class upsampling_block(nn.Module):
    """
    A pure upsampling block using bicubic interpolation followed by a smoothing convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(upsampling_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.upsample = nn.Upsample(scale_factor=stride, mode='bicubic', align_corners=False)
        # 3x3 Conv smooths bicubic interpolation artifacts and maps to out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class bottleneck_block(nn.Module):
    """
    A bottleneck block with residual connection.
    """
    def __init__(self, in_channels, out_channels):
        super(bottleneck_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1x1 reduce -> 3x3 spatial -> 1x1 expand
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

        # Residual connection adapter if channels change
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        res = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + res)
    
    