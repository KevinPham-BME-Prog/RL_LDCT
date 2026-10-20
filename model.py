import torch
import torch.nn as nn
from blocks_2 import MetaBlock , ResBlock, AttentionBlock, MambaBlock, GNNBlock, ConvLSTMBlock, FFTBlock, downsampling_block, upsampling_block

class DQNController(nn.Module):
    """
    The DQN Controller (Agent) which decides which block to use at each layer.
    """
    def __init__(self, feature_dim, hidden_dim, num_actions):
        super(DQNController, self).__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        
        # Use the maximum possible feature dimension for U-Net
        self.lstm = nn.LSTMCell(feature_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x_features, hidden_state=None):
        # Ensure input is always feature_dim size by padding if needed
        if x_features.size(-1) < self.feature_dim:
            x_features = torch.nn.functional.pad(
                x_features, 
                (0, self.feature_dim - x_features.size(-1))
            )
        
        if hidden_state is None:
            h_t = torch.zeros(x_features.size(0), self.hidden_dim, device=x_features.device)
            c_t = torch.zeros(x_features.size(0), self.hidden_dim, device=x_features.device)
        else:
            h_t, c_t = hidden_state
            
        h_t, c_t = self.lstm(x_features, (h_t, c_t))
        q_values = self.q_head(h_t)
        return q_values, (h_t, c_t)

class NAS_SR_Model(nn.Module):
    """
    The main model combining the worker (super-resolution network) and the controller.
    """
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        self.num_features = args.num_features
        self.UNet_initialization = args.use_UNet
        self.input_skip_only = getattr(args, 'input_skip_only', 0)

        if self.UNet_initialization == 1: #code used to initialize UNet structure
            self.layer_channel = []
            mid = self.num_layers // 2
            for i in range(self.num_layers):
                if i <= mid:
                    ch = self.num_features * (2 ** i)
                else:
                    ch = self.num_features * (2 ** (self.num_layers - 1 - i))
                self.layer_channel.append(ch)

        #need to add four main things
        #technically make a list that takes into account all possible layer locations and their input channel given that the downconv and the transpose convs will be adjusting the 
        #input and output channels respectively
        #need to adjust the code so that the features are saved as a list and thencalled when needed for the skip connections, where it performs concationation
        #need to make a 2d list/array to save all possible  operatioon combinations to ensure they are called properly
        #for this you need to make sure that the order is [operation][layer_num]
        #this then will ensure that the correct operation with the correct input channel is called at the correct layer
        #have this done by saturday/sunday night
        layer_list = []
        unet_options = []
        # Wrap each pure operation in the MetaBlock
        if getattr(args, 'use_resblock', 1):
            layer_list.append(MetaBlock(operation=ResBlock(args.num_features), num_features=args.num_features))
            if self.UNet_initialization == 1:
                unet_options_init = []
                for i in range(self.num_layers):
                    in_ch = (self.layer_channel)[i]
                    unet_options_init.append(MetaBlock(operation=ResBlock(in_ch), num_features=in_ch))
                unet_options.append(unet_options_init)
            
            if getattr(args, 'use_transformer', 1):
                layer_list.append(MetaBlock(operation=AttentionBlock(args.num_features, window_size=args.window_size), num_features=args.num_features))
                if self.UNet_initialization == 1:
                    unet_options_init = []
                    for i in range(self.num_layers):
                        in_ch = (self.layer_channel)[i]
                        # Find the largest power of 2 that divides in_ch and doesn't exceed args.gat_heads
                        num_heads = 1
                        for h in range(min(args.gat_heads, in_ch), 0, -1):
                            if in_ch % h == 0:
                                num_heads = h
                                break
                        unet_options_init.append(MetaBlock(operation=AttentionBlock(in_ch, window_size=args.window_size, num_heads=num_heads), num_features=in_ch))
                    unet_options.append(unet_options_init)
            
        if getattr(args, 'use_mamba', 1):
            layer_list.append(MetaBlock(operation=MambaBlock(args.num_features), num_features=args.num_features))
            if self.UNet_initialization == 1:
                unet_options_init = []
                for i in range(self.num_layers):
                    in_ch = (self.layer_channel)[i]
                    unet_options_init.append(MetaBlock(operation=MambaBlock(in_ch), num_features=in_ch))
                unet_options.append(unet_options_init)
            
        # --- ADD THIS SECTION TO INCLUDE THE GNN BLOCK ---
        if getattr(args, 'use_gnn', 1):
            layer_list.append(MetaBlock(operation=GNNBlock(args.num_features), num_features=args.num_features))
            if self.UNet_initialization == 1:
                unet_options_init = []
                for i in range(self.num_layers):
                    in_ch = (self.layer_channel)[i]
                    unet_options_init.append(MetaBlock(operation=GNNBlock(in_ch), num_features=in_ch))
                unet_options.append(unet_options_init)
            
        if getattr(args, 'use_lstmconv', 1):
            layer_list.append(MetaBlock(operation=ConvLSTMBlock(args.num_features), num_features=args.num_features))
            if self.UNet_initialization == 1:
                unet_options_init = []
                for i in range(self.num_layers):
                    in_ch = self.layer_channel[i]
                    unet_options_init.append(MetaBlock(operation=ConvLSTMBlock(in_ch), num_features=in_ch))
                unet_options.append(unet_options_init)
            
        self.layer_options = nn.ModuleList(layer_list)
        self.action_space_size = len(self.layer_options)
        
        if self.UNet_initialization == 1:
            # Create nested ModuleList: [block_type][layer_num]
            self.unet_layer_options = nn.ModuleList([
                nn.ModuleList(block_layers) if isinstance(block_layers, list) else block_layers
                for block_layers in unet_options
            ])

        # Worker network components
        # previous: nn.Conv2d(3, args.num_features, kernel_size=3, padding=1)
        # expect single-channel grayscale input
        
        # Initial convolution - should match the first layer's channel count
        if self.UNet_initialization == 1:
            self.initial_conv = nn.Sequential(
                nn.Conv2d(1, self.layer_channel[0], kernel_size=5, padding=2, bias=False),
                nn.BatchNorm2d(self.layer_channel[0]),
                nn.ReLU(inplace=True)
            )
            
            # Add channel transition layers between U-Net layers
            self.channel_transitions = nn.ModuleList([
                nn.Conv2d(self.layer_channel[i], self.layer_channel[i+1], kernel_size=1) 
                if self.layer_channel[i] != self.layer_channel[i+1] else nn.Identity()
                for i in range(self.num_layers - 1)
            ])
        else:
            self.initial_conv = nn.Conv2d(1, args.num_features, kernel_size=3, padding=1)
            self.channel_transitions = None
        #make the downsampler, upsampler and bottleneck blocks here for UNet initialization
        if self.UNet_initialization == 1:
            self.downsamplers = nn.ModuleList()
            self.upsamplers = nn.ModuleList()
            for i in range(self.num_layers -1):
                if i < self.num_layers // 2:
                    in_ch = self.layer_channel[i]
                    out_ch = self.layer_channel[i+1]
                    down_block = downsampling_block(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
                    self.downsamplers.append(down_block)
                elif i >= self.num_layers // 2:
                    in_ch = self.layer_channel[i]
                    out_ch = self.layer_channel[i+1]
                    up_block = upsampling_block(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
                    self.upsamplers.append(up_block)

            # Skip combiners fuse encoder skips with decoder activations via concatenation + learned 1x1 conv
            num_skip_levels = self.num_layers // 2
            if not self.input_skip_only:
                self.skip_combiners = nn.ModuleList([nn.ModuleDict() for _ in range(num_skip_levels)])
            else:
                self.skip_combiners = nn.ModuleList()
        else:
            self.downsamplers = nn.ModuleList()
            self.upsamplers = nn.ModuleList()
            self.skip_combiners = nn.ModuleList()
        if self.UNet_initialization == 1:
            # U-Net decoder outputs self.layer_channel[-1] feature maps
            up_in_channels = self.layer_channel[-1]
            self.upsampler = nn.Conv2d(up_in_channels, 1, kernel_size=3, padding=1)
            initial_skip_in = self.layer_channel[0]
        else:
            # Standard path keeps num_features channels
            up_in_channels = args.num_features
            self.upsampler = nn.Conv2d(up_in_channels, 1, kernel_size=3, padding=1)
            initial_skip_in = args.num_features

        # Global skips: project low-dose input and early features to decoder width
        self.input_skip_proj = nn.Conv2d(1, up_in_channels, kernel_size=1, bias=True)
        self.initial_skip_proj = (
            nn.Conv2d(initial_skip_in, up_in_channels, kernel_size=1, bias=True)
            if not self.input_skip_only else None
        )
        self.output_activation = nn.ReLU(inplace=True)
        
        # State pooler that works with variable channel dimensions
        self.state_pooler = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Spatial pooling
            nn.Flatten(1)                   # Flatten to [batch, channels]
        )
        
        # Add separate projectors for each layer if using U-Net
        if self.UNet_initialization == 1:
            unique_channels = sorted(set(self.layer_channel))
        else:
            unique_channels = [args.num_features]

        self.state_projector_map = nn.ModuleDict({
            str(ch): nn.Linear(ch, args.num_features) for ch in unique_channels
        })
        
        # Add the loss function as an attribute of the model
        self.mse_loss_fn = nn.MSELoss()

        # The controller is initialized here but can be replaced in train.py
        self.controller = DQNController(
            feature_dim=args.num_features,
            hidden_dim=args.controller_hidden_dim,
            num_actions=self.action_space_size
        )

        # Track lazily created skip combiner parameters for optimizer registration
        self._new_skip_params = []

    # Add the helper function to get the worker parameters
    def get_worker_params(self):
        """Returns a list of all parameters belonging to the worker network."""
        return list(self.initial_conv.parameters()) + \
               list(self.layer_options.parameters()) + \
               list(self.upsampler.parameters()) + \
               (list(self.downsamplers.parameters()) if self.UNet_initialization == 1 else []) + \
               (list(self.upsamplers.parameters()) if self.UNet_initialization == 1 else []) + \
               (list(self.unet_layer_options.parameters()) if self.UNet_initialization == 1 else []) + \
               (list(self.skip_combiners.parameters()) if self.UNet_initialization == 1 else []) + \
               list(self.state_projector_map.parameters()) + \
               list(self.input_skip_proj.parameters()) + \
               (list(self.initial_skip_proj.parameters()) if self.initial_skip_proj is not None else []) + \
               list(self.output_activation.parameters())

    def project_state(self, pooled_state):
        """Project pooled feature state to controller input dimension."""
        channel_dim = pooled_state.size(1)
        key = str(channel_dim)
        if key not in self.state_projector_map:
            raise ValueError(f"No state projector registered for channel dim {channel_dim}")
        return self.state_projector_map[key](pooled_state)

    def combine_with_global_skips(self, features, x_lr, initial_features):
        """Fuse decoder features with projected global skips before reconstruction."""
        input_skip = self.input_skip_proj(x_lr)
        if input_skip.shape != features.shape:
            raise ValueError(
                f"Input skip shape mismatch: features {features.shape}, projected skip {input_skip.shape}"
            )
        combined = features + input_skip
        if self.initial_skip_proj is not None:
            initial_skip = self.initial_skip_proj(initial_features)
            if initial_skip.shape != features.shape:
                raise ValueError(
                    f"Initial skip shape mismatch: features {features.shape}, projected skip {initial_skip.shape}"
                )
            combined = combined + initial_skip
        return combined #replace this back with self.output_activation to readd relu

    def fuse_local_skip(self, decoder_features, skip_features, level_idx):
        """Concatenate decoder features with encoder skip and project back to decoder width."""
        if level_idx >= len(self.skip_combiners):
            raise ValueError(
                f"Skip combiner index {level_idx} out of range for {len(self.skip_combiners)} levels"
            )

        skip_ch = skip_features.shape[1]
        dec_ch = decoder_features.shape[1]
        combiner_bank = self.skip_combiners[level_idx]
        key = f"{skip_ch}->{dec_ch}"

        if key not in combiner_bank:
            conv = nn.Conv2d(skip_ch + dec_ch, dec_ch, kernel_size=1, bias=True).to(decoder_features.device)
            combiner_bank[key] = conv
            self._new_skip_params.append(list(conv.parameters()))

        merged = torch.cat([decoder_features, skip_features], dim=1)
        return combiner_bank[key](merged)

    def forward(self, x_lr, fixed_arc=None):
        """
        Performs a forward pass for a fixed architecture with U-Net skip connections.
        """
        if fixed_arc is None:
            raise ValueError("forward() should be called with a fixed_arc during validation.")
            
        
        if self.UNet_initialization == 1:
            feat_mat = []  # Stores encoder activations for skip connections (LIFO order)
            features = self.initial_conv(x_lr)
            initial_features = features

            # Process all layers with the architecture
            for i in range(self.num_layers):
                if i >= len(fixed_arc):
                    break
                    
                action_idx = fixed_arc[i]
                if not (0 <= action_idx < len(self.layer_options)):
                    raise IndexError(f"Invalid action index {action_idx} for {len(self.layer_options)} options.")
                
                # Use the U-Net-specific block for this action and layer position
                chosen_layer = self.unet_layer_options[action_idx][i]
                
                # Downsampling phase (first half)
                if i < self.num_layers // 2:
                    if not self.input_skip_only:
                        feat_mat.append(features)  # Capture pre-block features for skip connection
                    features = chosen_layer(features)
                    if i < len(self.downsamplers):
                        features = self.downsamplers[i](features)
                
                # Bottleneck layer (middle)
                elif i == self.num_layers // 2:
                    features = chosen_layer(features)
                
                # Upsampling phase (second half)
                else:
                    # Apply upsampling first
                    upsample_idx = i - (self.num_layers // 2 + 1)
                    skip_connection = None
                    if not self.input_skip_only and feat_mat:
                        skip_connection = feat_mat.pop()
                    if 0 <= upsample_idx < len(self.upsamplers):
                        features = self.upsamplers[upsample_idx](features)
                    
                    # Apply block prior to merging the skip connection
                    features = chosen_layer(features)

                    if skip_connection is not None:
                        if skip_connection.shape[-2:] != features.shape[-2:]:
                            raise ValueError(
                                f"Spatial mismatch for skip connection at layer {i}: "
                                f"features spatial {features.shape[-2:]}, skip spatial {skip_connection.shape[-2:]}"
                            )
                        if len(self.skip_combiners) == 0:
                            raise ValueError("Skip combiners are not initialized but a skip connection is available.")
                        comb_idx = min(upsample_idx, len(self.skip_combiners) - 1)
                        features = self.fuse_local_skip(features, skip_connection, comb_idx)
        else:
            # Standard residual path
            features = self.initial_conv(x_lr)
            initial_features = features
            for layer_idx in fixed_arc:
                if not (0 <= layer_idx < len(self.layer_options)):
                    raise IndexError(f"Invalid layer index {layer_idx} for {len(self.layer_options)} options.")
                chosen_layer = self.layer_options[layer_idx]
                features = features + chosen_layer(features)
        # Fuse global skips before reconstruction (both U-Net and standard paths)
        features = self.combine_with_global_skips(features, x_lr, initial_features)
        y_pred = self.upsampler(features)

        if y_pred.shape != x_lr.shape:
            raise ValueError(
                f"Final output shape mismatch: model output {y_pred.shape}, input image {x_lr.shape}"
            )

        final_output = self.output_activation(y_pred)
        
        return {"y_pred": final_output}
