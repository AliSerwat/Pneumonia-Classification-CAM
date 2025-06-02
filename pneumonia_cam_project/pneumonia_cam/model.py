import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import Optional, Tuple, List, Dict

# ============================ Custom Model Components ============================ #

class MedicalAttentionGate(nn.Module):
    """
    A module that applies a learnable attention mechanism to an input tensor.
    This gate computes attention weights using a small neural network and
    then uses these weights to scale the input tensor element-wise.
    """
    def __init__(self, dimension: int):
        """
        Initializes the MedicalAttentionGate.

        Args:
            dimension (int): The dimensionality of the input features.
        """
        super().__init__()
        bottleneck_dimension = max(1, dimension // 4)
        if dimension < 4:
            print(
                f"⚠️ WARNING (MedicalAttentionGate): Input dimension ({dimension}) is small. "
                f"Bottleneck dimension set to {bottleneck_dimension}."
            )
        self.attention_net = nn.Sequential(
            nn.Linear(dimension, bottleneck_dimension),
            nn.Sigmoid(), # Using Sigmoid for smoother gating, could be GELU/ReLU too
            nn.Linear(bottleneck_dimension, dimension),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the attention mechanism to the input tensor.
        Input tensor `x` is expected to have its feature dimension as its last dimension.
        Example shapes for x: (batch_size, dimension) or (batch_size, ..., dimension).

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor `x` scaled by the learned attention weights.
        """
        attention_weights = self.attention_net(x)
        return x * attention_weights

class EnhancedClassifier(nn.Module):
    """
    An Enhanced Classifier with Attention and Custom Initialization for image features.
    Processes input features through linear layers, batch normalization, GELU,
    dropout, and a MedicalAttentionGate, before producing classification logits.
    Features custom weight initialization tailored for GELU.
    """
    _EMPIRICAL_GELU_GAIN: Optional[float] = None

    @staticmethod
    def _get_empirical_gelu_gain() -> float:
        """
        Calculates (or retrieves from cache) an empirical gain for GELU activation.
        Estimated by comparing std dev of a random tensor before/after GELU.

        Returns:
            float: The calculated or cached empirical gain for GELU.
        """
        if EnhancedClassifier._EMPIRICAL_GELU_GAIN is None:
            with torch.no_grad():
                toy_input = torch.randn(100_000) # CPU tensor for this calc
                activated_output = F.gelu(toy_input)
                std_dev_input = toy_input.std().item()
                std_dev_output = activated_output.std().item()
                if std_dev_output == 0:
                    print("⚠️ WARNING (EnhancedClassifier): GELU output std dev is zero. Defaulting gain to 1.0.")
                    EnhancedClassifier._EMPIRICAL_GELU_GAIN = 1.0
                else:
                    EnhancedClassifier._EMPIRICAL_GELU_GAIN = std_dev_input / std_dev_output
        return EnhancedClassifier._EMPIRICAL_GELU_GAIN

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.3,
        hidden_dim: int = 512,
    ):
        """
        Initializes the EnhancedClassifier.

        Args:
            in_features (int): Number of input features from the backbone.
            out_features (int): Number of output features (e.g., number of classes).
            dropout_rate (float): Dropout probability.
            hidden_dim (int): Dimensionality of the intermediate hidden layers.
        """
        super().__init__()
        hidden_dim_expanded = hidden_dim * 2

        self.feature_map_processor = nn.Sequential(
            nn.Linear(in_features, hidden_dim_expanded),
            nn.BatchNorm1d(hidden_dim_expanded), # Assumes input is (batch, features)
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_expanded, hidden_dim),
            # Option: Add BN/GELU before attention if desired
            # nn.BatchNorm1d(hidden_dim),
            # nn.GELU(),
            MedicalAttentionGate(hidden_dim),
            nn.Dropout(dropout_rate / 2), # Reduced dropout after attention
            nn.Linear(hidden_dim, out_features),
        )
        self._initialize_module_weights()

    def _initialize_module_weights(self) -> None:
        """Initializes weights of linear layers using Kaiming-style normal init scaled by GELU gain."""
        custom_gelu_gain = EnhancedClassifier._get_empirical_gelu_gain()
        for module in self.feature_map_processor.modules():
            if isinstance(module, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                if fan_in == 0:
                    print(f"⚠️ WARNING (EnhancedClassifier): Fan-in is zero for Linear layer. Using default init.")
                    nn.init.normal_(module.weight, mean=0.0, std=0.02) # Fallback
                else:
                    std_deviation = custom_gelu_gain / math.sqrt(fan_in)
                    nn.init.normal_(module.weight, mean=0.0, std=std_deviation)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01) # Small constant bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): Input tensor, expected shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output logits, shape (batch_size, out_features).
        """
        return self.feature_map_processor(x)

# ============================ Main Pneumonia Detection Model ============================ #

class PneumoniaModelCAM(nn.Module):
    """
    Pneumonia detection model with a TIMM backbone and an EnhancedClassifier head.
    This model is designed to output feature maps for Class Activation Mapping (CAM).
    """
    def __init__(
        self,
        pretrained_model_name: str = "resnet50",
        num_classes: int = 1, # Binary classification (pneumonia vs. healthy)
        in_channels: int = 1, # Grayscale medical images
        img_size: int = 256,  # For dummy input shape inference if needed by backbone
        classifier_hidden_dim: int = 512,
        classifier_dropout_rate: float = 0.3,
        # criterion_pos_weight: float = 1.0, # Loss criterion is handled in training script
    ):
        """
        Initializes the PneumoniaModelCAM.

        Args:
            pretrained_model_name (str): Name of the TIMM pretrained model.
            num_classes (int): Number of output classes.
            in_channels (int): Number of input image channels.
            img_size (int): Assumed image size (height/width) for dummy input pass.
            classifier_hidden_dim (int): Hidden dimension for the EnhancedClassifier.
            classifier_dropout_rate (float): Dropout rate for the EnhancedClassifier.
        """
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # Create backbone using TIMM
        # num_classes=0 for feature extraction, global_pool='' for feature maps
        try:
            self.backbone = timm.create_model(
                self.pretrained_model_name,
                pretrained=True,
                num_classes=0,
                global_pool="", # Request feature maps (N, C, H, W)
                in_chans=self.in_channels,
            )
            print(f"✅ Backbone '{self.pretrained_model_name}' loaded with `in_chans={self.in_channels}`.")
        except Exception as e_timm:
            print(f"⚠️ WARNING: Failed to create backbone '{self.pretrained_model_name}' with timm `in_chans={self.in_channels}`. Error: {e_timm}")
            print("     Attempting fallback: create with default 3 input channels, then try manual override.")
            self.backbone = timm.create_model(
                self.pretrained_model_name, pretrained=True, num_classes=0, global_pool=""
            )
            if self.in_channels != 3 and not self._attempt_manual_first_conv_override(self.in_channels):
                print(f"❌ ERROR: Failed to adapt backbone for {self.in_channels} input channels. Model may not work.")
            else:
                 print(f"✅ Backbone '{self.pretrained_model_name}' loaded (fallback) and adapted for `in_chans={self.in_channels}`.")


        # Determine backbone output features using a dummy pass
        self.num_backbone_out_features = self._get_backbone_out_features(self.img_size)

        # Pooling and flattening layers to prepare features for the classifier
        self.global_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # Pools to (N, C_backbone, 1, 1)
        self.flatten_features = nn.Flatten(start_dim=1) # Flattens to (N, C_backbone)

        # Custom classifier head
        self.classifier = EnhancedClassifier(
            in_features=self.num_backbone_out_features,
            out_features=self.num_classes,
            dropout_rate=classifier_dropout_rate,
            hidden_dim=classifier_hidden_dim,
        )

    def _get_backbone_out_features(self, img_size_for_dummy_pass: int) -> int:
        """
        Determines the number of output features from the backbone using a dummy pass.
        """
        self.backbone.eval() # Eval mode for dummy pass
        # Create dummy input on CPU to avoid device issues during init
        dummy_input = torch.randn(1, self.in_channels, img_size_for_dummy_pass, img_size_for_dummy_pass, device='cpu')
        with torch.no_grad():
            dummy_feature_maps = self.backbone(dummy_input)
        self.backbone.train() # Back to train mode
        return dummy_feature_maps.shape[1] # C from (N, C, H, W)


    def _attempt_manual_first_conv_override(self, target_in_channels: int) -> bool:
        """
        Fallback to manually modify the first conv layer if timm's `in_chans` fails.
        This is model-dependent and fragile.
        Returns True if modification was attempted, False otherwise.
        """
        # Try common names for the first convolutional layer
        first_conv_layer_names = ['conv_stem', 'conv1']
        modified = False
        for layer_name in first_conv_layer_names:
            if hasattr(self.backbone, layer_name):
                original_conv = getattr(self.backbone, layer_name)
                if isinstance(original_conv, nn.Conv2d) and original_conv.in_channels == 3: # Only if default was 3
                    print(f"    Attempting to manually modify '{layer_name}' for {target_in_channels} input channels.")
                    new_out_channels = original_conv.out_channels
                    new_conv = nn.Conv2d(
                        target_in_channels,
                        new_out_channels,
                        kernel_size=original_conv.kernel_size,
                        stride=original_conv.stride,
                        padding=original_conv.padding,
                        dilation=original_conv.dilation,
                        groups=original_conv.groups, # Preserve groups if it's a depthwise conv
                        bias=(original_conv.bias is not None),
                    )
                    # Attempt to copy weights if reasonable (e.g., averaging for single channel)
                    if original_conv.weight.data.shape[1] == 3 and target_in_channels == 1:
                        new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
                        print(f"    Copied and averaged weights for '{layer_name}'.")
                    else:
                        print(f"    Weights for '{layer_name}' re-initialized (cannot directly copy from 3 to {target_in_channels} channels easily).")

                    setattr(self.backbone, layer_name, new_conv)
                    print(f"    Manually replaced '{layer_name}'. Output channels kept at {new_out_channels}.")
                    modified = True
                    break # Stop after modifying the first found layer
        if not modified:
            print(f"    ⚠️ Manual override: Could not find a known standard first Conv2d attribute to modify from 3 channels.")
        return modified

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): Input tensor (batch_size, in_channels, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logits (torch.Tensor): Classification logits.
                                         Shape (batch_size,) if num_classes=1,
                                         else (batch_size, num_classes).
                - feature_maps (torch.Tensor): Feature maps from the backbone
                                               (batch_size, C_backbone, H_feat, W_feat).
        """
        feature_maps = self.backbone(x)

        pooled_features = self.global_adaptive_pool(feature_maps)
        flat_features = self.flatten_features(pooled_features)

        logits = self.classifier(flat_features)

        # Squeeze last dim if num_classes is 1 (for BCEWithLogitsLoss compatibility)
        if self.num_classes == 1 and logits.ndim > 1 and logits.shape[1] == 1:
            processed_logits = logits.squeeze(1)
        else:
            processed_logits = logits

        return processed_logits, feature_maps


if __name__ == "__main__":
    print("Testing model.py components...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test MedicalAttentionGate
    print("\n--- Testing MedicalAttentionGate ---")
    attention_gate = MedicalAttentionGate(dimension=128).to(device)
    test_input_attn = torch.randn(4, 128).to(device) # Batch of 4, 128 features
    output_attn = attention_gate(test_input_attn)
    print(f"Attention Gate Input Shape: {test_input_attn.shape}")
    print(f"Attention Gate Output Shape: {output_attn.shape}")
    assert output_attn.shape == test_input_attn.shape

    # Test EnhancedClassifier
    print("\n--- Testing EnhancedClassifier ---")
    classifier = EnhancedClassifier(in_features=256, out_features=10, hidden_dim=128).to(device)
    test_input_clf = torch.randn(4, 256).to(device) # Batch of 4, 256 features
    output_clf = classifier(test_input_clf)
    print(f"Enhanced Classifier Input Shape: {test_input_clf.shape}")
    print(f"Enhanced Classifier Output Shape: {output_clf.shape}")
    assert output_clf.shape == (4, 10)

    # Test PneumoniaModelCAM
    print("\n--- Testing PneumoniaModelCAM ---")
    # Use a small, fast model for testing
    # common_timm_models = ["resnet18", "efficientnet_b0", "mobilenetv3_small_050.lamb_in1k"]
    # test_model_name = common_timm_models[1]
    test_model_name = "resnet18" # resnet18 is generally available and small

    print(f"Attempting to create PneumoniaModelCAM with backbone: {test_model_name}")
    try:
        pneumonia_model = PneumoniaModelCAM(
            pretrained_model_name=test_model_name,
            num_classes=1,
            in_channels=1, # Test with 1 channel
            img_size=64, # Smaller image size for faster test
            classifier_hidden_dim=64,
            classifier_dropout_rate=0.1
        ).to(device)

        test_input_pneumonia = torch.randn(2, 1, 64, 64).to(device) # Batch of 2, 1 channel, 64x64
        logits, features = pneumonia_model(test_input_pneumonia)

        print(f"PneumoniaModelCAM Input Shape: {test_input_pneumonia.shape}")
        print(f"PneumoniaModelCAM Logits Shape: {logits.shape}")
        print(f"PneumoniaModelCAM Feature Maps Shape: {features.shape}")

        assert logits.shape == (2,) or logits.shape == (2,1) # (Batch,) if num_classes=1 and squeezed
        assert features.ndim == 4 # (Batch, Channels, Height, Width)

        # Test with 3 input channels
        print("\n--- Testing PneumoniaModelCAM with 3 input channels ---")
        pneumonia_model_3ch = PneumoniaModelCAM(
            pretrained_model_name=test_model_name,
            num_classes=5, # Multi-class test
            in_channels=3,
            img_size=64,
            classifier_hidden_dim=64,
        ).to(device)
        test_input_3ch = torch.randn(2, 3, 64, 64).to(device)
        logits_3ch, features_3ch = pneumonia_model_3ch(test_input_3ch)
        print(f"PneumoniaModelCAM (3ch) Logits Shape: {logits_3ch.shape}")
        assert logits_3ch.shape == (2, 5)

    except Exception as e:
        print(f"Error during PneumoniaModelCAM test with '{test_model_name}': {e}")
        print("This might be due to the model not being available in timm or issues with dummy pass.")
        print("Ensure you have an internet connection if downloading models for the first time.")

    print("\n--- model.py tests finished ---")
