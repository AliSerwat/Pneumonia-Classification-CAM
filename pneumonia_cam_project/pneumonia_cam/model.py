import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import Optional, Tuple, List, Union, Dict # Added Dict
import numpy as np # For CAM processing
import cv2 # For CAM resizing

class MedicalAttentionGate(nn.Module):
    """
    A simple attention gate for medical imaging contexts.
    It learns to scale input features, effectively focusing on more relevant parts.
    """
    def __init__(self, dimension: int):
        """
        Args:
            dimension (int): The input feature dimension.
        """
        super().__init__()
        # Bottleneck dimension is typically a fraction of the input dimension
        bottleneck_dimension = max(1, dimension // 4) 
        if dimension < 4 and dimension > 0: # Added check for dimension > 0
            # print(f"⚠️ WARNING: Input dimension ({dimension}) is very small for MedicalAttentionGate. Consider if this layer is beneficial.")
            pass # Suppressing print for cleaner use as a library
        elif dimension <= 0:
            raise ValueError(f"Input dimension must be positive, got {dimension}")

        self.attention_net = nn.Sequential(
            nn.Linear(dimension, bottleneck_dimension),
            nn.Sigmoid(), # Using Sigmoid to get weights between 0 and 1
            nn.Linear(bottleneck_dimension, dimension),
            nn.Sigmoid(), # Final attention weights also between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dimension).
        Returns:
            torch.Tensor: Output tensor, element-wise multiplied by attention weights.
        """
        attention_weights = self.attention_net(x)
        return x * attention_weights

class EnhancedClassifier(nn.Module):
    """
    An enhanced classifier head with GELU activation, Batch Normalization,
    Dropout, and a MedicalAttentionGate. It also uses custom weight initialization.
    """
    _EMPIRICAL_GELU_GAIN: Optional[float] = None

    @staticmethod
    def _get_empirical_gelu_gain() -> float:
        """
        Calculates or retrieves the empirical gain for GELU activation.
        This is used for Kaiming-like initialization specific to GELU.
        The gain is cached after first calculation.
        """
        if EnhancedClassifier._EMPIRICAL_GELU_GAIN is None:
            with torch.no_grad(): # Ensure no gradients are computed during this calculation
                toy_input = torch.randn(100_000) # Large sample for stable statistics
                activated_output = nn.functional.gelu(toy_input)
                std_dev_input = toy_input.std().item()
                std_dev_output = activated_output.std().item()
                if std_dev_output == 0: # Avoid division by zero
                    EnhancedClassifier._EMPIRICAL_GELU_GAIN = 1.0
                else:
                    EnhancedClassifier._EMPIRICAL_GELU_GAIN = std_dev_input / std_dev_output
        return EnhancedClassifier._EMPIRICAL_GELU_GAIN

    def __init__(self, in_features: int, out_features: int,
                 dropout_rate: float = 0.3, hidden_dim: int = 512):
        """
        Args:
            in_features (int): Number of input features from the backbone.
            out_features (int): Number of output classes.
            dropout_rate (float): Dropout probability.
            hidden_dim (int): Dimension of the main hidden layer.
        """
        super().__init__()
        if in_features <=0 or hidden_dim <=0:
            raise ValueError("in_features and hidden_dim must be positive.")

        hidden_dim_expanded = hidden_dim * 2 # For the initial expansion layer
        self.feature_map_processor = nn.Sequential(
            nn.Linear(in_features, hidden_dim_expanded),
            nn.BatchNorm1d(hidden_dim_expanded),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_expanded, hidden_dim),
            MedicalAttentionGate(hidden_dim), # Apply attention
            nn.Dropout(dropout_rate / 2), # Slightly less dropout after attention
            nn.Linear(hidden_dim, out_features),
        )
        self._initialize_module_weights()

    def _initialize_module_weights(self) -> None:
        """
        Initializes weights of the linear layers using a normal distribution
        scaled by the empirical GELU gain and fan-in. Biases are initialized to a small constant.
        """
        custom_gelu_gain = EnhancedClassifier._get_empirical_gelu_gain()
        for module in self.feature_map_processor.modules():
            if isinstance(module, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                if fan_in == 0: # Should not happen with proper layer definitions
                    std_deviation = 0.02 # Fallback std
                else:
                    std_deviation = custom_gelu_gain / math.sqrt(fan_in)
                
                nn.init.normal_(module.weight, mean=0.0, std=std_deviation)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1) # Small constant bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor from the backbone's feature extractor.
        Returns:
            torch.Tensor: Logits for each class.
        """
        return self.feature_map_processor(x)


class PneumoniaModelCAM(nn.Module):
    """
    Pneumonia detection model with a TIMM backbone, EnhancedClassifier head,
    and integrated Class Activation Map (CAM) generation capability.
    """
    def __init__(self, pretrained_model_name: str, num_classes: int = 1, in_channels: int = 1,
                 criterion_pos_weight: Optional[float] = None, 
                 classifier_hidden_dim: int = 512,
                 classifier_dropout_rate: float = 0.3, img_size: int = 224):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Note: self.device is not explicitly set here. Layer devices are managed by .to(device) call on the model instance.
        # For the dummy_input pass, we'll handle device contextually.

        try:
            self.backbone = timm.create_model(
                self.pretrained_model_name, pretrained=True, num_classes=0,
                global_pool="", in_chans=self.in_channels
            )
            # print(f"✅ Backbone '{self.pretrained_model_name}' created with `in_chans={self.in_channels}`.")
        except RuntimeError as e_timm:
            # print(f"⚠️ TIMM model creation with in_chans={self.in_channels} failed: {e_timm}. Attempting fallback for 3 channels and manual override.")
            self.backbone = timm.create_model(
                self.pretrained_model_name, pretrained=True, num_classes=0, global_pool=""
            )
            if self.in_channels != 3:
                if not self._attempt_manual_first_conv_override(self.in_channels):
                    raise RuntimeError(f"❌ ERROR: Failed to adapt backbone for {self.in_channels} channels after fallback.") from e_timm
            # else:
                # print(f"⚠️ WARNING: Timm failed for in_chans=3, but succeeded with default (3 chans). This is unusual.")

        # Determine num_backbone_out_features dynamically
        # This needs to run on whatever device the model will be on, or at least the backbone.
        # If model is created then immediately moved to GPU, this needs to be after that.
        # For now, assume it's run on CPU or the device is handled before this point.
        # A common pattern is to call model.to(device) after instantiation.
        # We'll use a temporary device for the dummy pass, assuming parameters are not yet moved.
        temp_device = next(self.backbone.parameters()).device if len(list(self.backbone.parameters())) > 0 else torch.device("cpu")

        self.backbone.eval()
        dummy_tensor_dtype = torch.float32 # Use a consistent dtype
        try:
            # Try with a batch size of 1 for minimal memory, ensure it's on the same device as backbone
            dummy_input = torch.randn(1, self.in_channels, self.img_size, self.img_size, 
                                      dtype=dummy_tensor_dtype).to(temp_device)
            with torch.no_grad():
                dummy_feature_maps = self.backbone(dummy_input)
            self.num_backbone_out_features = dummy_feature_maps.shape[1]
        except Exception as e:
            # Fallback or error if dynamic feature calculation fails
            print(f"Error calculating num_backbone_out_features dynamically: {e}. You may need to set it manually.")
            # As a rough fallback, try to infer from model name, or set a common default
            if "efficientnet_b0" in pretrained_model_name: self.num_backbone_out_features = 1280
            elif "resnet34" in pretrained_model_name: self.num_backbone_out_features = 512
            else: self.num_backbone_out_features = 512 # A generic fallback
            print(f"Using fallback num_backbone_out_features: {self.num_backbone_out_features}")

        self.backbone.train()

        self.global_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_features = nn.Flatten(start_dim=1)
        self.classifier = EnhancedClassifier(
            in_features=self.num_backbone_out_features,
            out_features=self.num_classes,
            dropout_rate=classifier_dropout_rate,
            hidden_dim=classifier_hidden_dim
        )
        
        pos_weight_tensor = torch.tensor(criterion_pos_weight, dtype=torch.float) if criterion_pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    def _attempt_manual_first_conv_override(self, target_in_channels: int) -> bool:
        first_conv_names = ["conv_stem", "conv1", "features.0.0.conv", "features.0.conv", "stem.0"] 
        original_conv = None
        layer_name_found = ""
        for name in first_conv_names:
            try:
                module = self.backbone
                for part in name.split('.'):
                    module = getattr(module, part)
                if isinstance(module, nn.Conv2d):
                    original_conv = module
                    layer_name_found = name
                    break
            except AttributeError: continue
        if not original_conv:
            # print(f"    ⚠️ Manual override: Could not find a standard first Conv2d layer to modify from names: {first_conv_names}.")
            return False
        # print(f"    Attempting to manually modify '{layer_name_found}' for {target_in_channels} input channels.")
        new_conv = nn.Conv2d(
            in_channels=target_in_channels, out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size, stride=original_conv.stride,
            padding=original_conv.padding, dilation=original_conv.dilation,
            groups=original_conv.groups, bias=(original_conv.bias is not None)
        )
        if original_conv.weight.data.shape[1] == 3 and target_in_channels == 1:
            # print(f"    Adapting weights from 3 channels to 1 channel (averaging).")
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            if new_conv.bias is not None and original_conv.bias is not None: new_conv.bias.data = original_conv.bias.data
        elif original_conv.weight.data.shape[1] == target_in_channels:
            # print(f"    Input channels already match target {target_in_channels}. Copying weights.")
            new_conv.weight.data = original_conv.weight.data
            if new_conv.bias is not None and original_conv.bias is not None: new_conv.bias.data = original_conv.bias.data
        # else: print(f"    Cannot directly adapt weights. New conv layer will have random init.")
        module_parent = self.backbone
        name_parts = layer_name_found.split('.')
        for part in name_parts[:-1]: module_parent = getattr(module_parent, part)
        setattr(module_parent, name_parts[-1], new_conv)
        # print(f"    Manually replaced '{layer_name_found}'.")
        self.in_channels = target_in_channels
        return True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features_map = self.backbone(x)
        pooled_features = self.global_adaptive_pool(features_map)
        flat_features = self.flatten_features(pooled_features)
        logits = self.classifier(flat_features)
        if self.num_classes == 1: return logits.squeeze(-1), features_map 
        return logits, features_map

    def generate_cam(self, img_tensor: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor]]:
        self.eval()
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        processed_img_tensor = img_tensor.to(model_device, dtype=model_dtype)
        if processed_img_tensor.ndim == 3: processed_img_tensor = processed_img_tensor.unsqueeze(0)
        elif processed_img_tensor.ndim != 4:
            # print(f"Error: img_tensor has unsupported dimensions: {processed_img_tensor.shape}")
            return None, None
        
        current_target_size: Tuple[int, int] = target_size if target_size else (processed_img_tensor.shape[-2], processed_img_tensor.shape[-1])
        
        logits_out: Optional[torch.Tensor] = None
        cam_np_resized: Optional[np.ndarray] = None

        try:
            with torch.no_grad():
                logits, feature_maps = self(processed_img_tensor)
            logits_out = logits.squeeze().cpu() if logits is not None else None

            final_linear_layer = self.classifier.feature_map_processor[-1]
            if not isinstance(final_linear_layer, nn.Linear):
                # print("Error: Last layer of classifier is not nn.Linear for CAM.")
                return None, logits_out
            
            classifier_weights = final_linear_layer.weight.squeeze()
            if feature_maps.shape[1] != classifier_weights.shape[0]:
                # print(f"Error: Mismatch in feature map channels and classifier weights for CAM.")
                return None, logits_out

            cam_tensor = torch.einsum('bchw,c->bhw', feature_maps, classifier_weights)[0, :, :]
            cam_min, cam_max = cam_tensor.min(), cam_tensor.max()
            cam_normalized = (cam_tensor - cam_min) / (cam_max - cam_min + 1e-6) if (cam_max - cam_min > 1e-6) else torch.zeros_like(cam_tensor)
            cam_np = cam_normalized.cpu().numpy()
            cam_np_resized = cv2.resize(cam_np, (current_target_size[1], current_target_size[0]), interpolation=cv2.INTER_LINEAR)
            return cam_np_resized, logits_out
        except Exception as e:
            # print(f"Error during CAM generation: {e}")
            return cam_np_resized, logits_out # Return what we have
