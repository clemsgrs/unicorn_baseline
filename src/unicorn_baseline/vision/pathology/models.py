from pathlib import Path
import os
import json

import torch
import torch.nn as nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from unicorn_baseline.vision.pathology.model_utils import update_state_dict


class SlideFeatureExtractor(nn.Module):
    def __init__(self, model_dir: Path):
        super(SlideFeatureExtractor, self).__init__()
        self.model_dir = model_dir
        self.build_encoders()
        self.set_device()
        self.load_weights()
        for param in self.parameters():
            param.requires_grad = False

    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def build_encoders(self):
        raise NotImplementedError

    def load_weights(self):
        raise NotImplementedError

    def get_transforms(self):
        return self.tile_encoder.get_transforms()

    def forward(self, x):
        return self.tile_encoder(x)

    def forward_slide(self, **kwargs):
        return self.slide_encoder(**kwargs)

    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.__class__.__name__}\n"
            f"Total Parameters: {total_params / 1e6:.2f}M\n"
            f"Trainable Parameters: {trainable_params / 1e6:.2f}M"
        )


class ProvGigaPathTile(nn.Module):
    def __init__(self, model_dir: Path):
        super(ProvGigaPathTile, self).__init__()
        self.model_dir = model_dir
        self.features_dim = 1536
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_config()
        self.model = self.build_encoder()
        self.load_weights()
        self.transforms = self.get_transforms()

    def load_config(self):
        config_path = self.model_dir / "gigapath-config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def build_encoder(self):
        encoder = timm.create_model(
            "vit_giant_patch14_dinov2",
            pretrained=False,
            **self.config["model_args"]
        )
        return encoder

    def load_weights(self):
        """Load pretrained weights for the tile encoder."""
        checkpoint_path = self.model_dir / "gigapath-tile-encoder.pth"
        print(f"Loading tile encoder weights from {checkpoint_path}...")
        weights = torch.load(checkpoint_path, map_location=self.device)
        updated_sd, msg = update_state_dict(
            model_dict=self.model.state_dict(), state_dict=weights
        )
        print(msg)
        self.model.load_state_dict(updated_sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def get_transforms(self):
        """Retrieve the transformation pipeline for input images."""
        data_config = resolve_data_config(
            self.config["pretrained_cfg"], model=self.model
        )
        return create_transform(**data_config)

    def forward(self, x):
        return self.model(x)


class ProvGigaPath(SlideFeatureExtractor):

    def build_encoders(self):
        # needed to avoid timm error when creating slide encoder model
        # might need to manually add the libary to the pythonpath here with sys
        import gigapath.slide_encoder as sd

        self.tile_encoder = ProvGigaPathTile(model_dir=self.model_dir)
        self.slide_encoder = timm.create_model("gigapath_slide_enc12l768d", pretrained=False, in_chans=1536)

    def load_weights(self):
        """Load pretrained weights for the tile encoder."""
        checkpoint_path = self.model_dir / "gigapath-slide-encoder.pth"
        print(f"Loading slide encoder weights from {checkpoint_path}...")
        weights = torch.load(checkpoint_path, map_location=self.device)["model"]
        updated_sd, msg = update_state_dict(
            model_dict=self.slide_encoder.state_dict(), state_dict=weights
        )
        print(msg)
        self.slide_encoder.load_state_dict(updated_sd, strict=True)
        self.slide_encoder.to(self.device)
        self.slide_encoder.eval()

    def forward_slide(self, tile_features, tile_coordinates, **kwargs):
        tile_features = tile_features.unsqueeze(0)
        output = self.slide_encoder(tile_features, tile_coordinates)
        output = output[0].squeeze()
        return output


class Virchow(nn.Module):
    """
    Tile-level feature extractor.
    """

    def __init__(self, model_dir, mode: str):
        super().__init__()
        self.model_dir = model_dir
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model configuration
        with open(os.path.join(self.model_dir, "virchow-config.json"), "r") as f:
            self.config = json.load(f)

        # Initialize tile encoder
        self.tile_encoder = timm.create_model(
            **self.config, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        )

        self.load_weights()
        self.transforms = self.get_transforms()

    def load_weights(self):
        """Load pretrained weights for the tile encoder."""
        checkpoint_path = os.path.join(self.model_dir, "virchow-tile-encoder.pth")
        print(f"Loading tile encoder weights from {checkpoint_path}...")
        weights = torch.load(checkpoint_path, map_location=self.device)
        updated_sd, msg = update_state_dict(
            model_dict=self.tile_encoder.state_dict(), state_dict=weights
        )
        print(msg)
        self.tile_encoder.load_state_dict(updated_sd, strict=True)
        self.tile_encoder.to(self.device)
        self.tile_encoder.eval()

    def get_transforms(self):
        """Retrieve the transformation pipeline for input images."""
        data_config = resolve_data_config(
            self.config["pretrained_cfg"], model=self.tile_encoder
        )
        return create_transform(**data_config)

    def forward(self, x):
        """Extract tile-level embeddings."""
        x = x.to(self.device)
        with torch.no_grad():
            output = self.tile_encoder(x)

        # Extract class and patch tokens
        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

        if self.mode == "full":
            return embedding

        elif self.mode == "patch_tokens":
            return patch_tokens

        elif self.mode == "class_token":
            return class_token

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from 'full', 'patch_tokens', or 'class_token'.")


class PRISM(SlideFeatureExtractor):
    """
    Slide-level feature extractor (PRISM model).
    """

    def build_encoders(self):
        import sys

        sys.path.insert(0, self.model_dir)
        from unicorn_baseline.vision_language.prism.configuring_prism import (
            PerceiverConfig,
            PrismConfig,
        )
        from unicorn_baseline.vision_language.prism.modeling_prism import Prism
        from transformers.models.biogpt.configuration_biogpt import BioGptConfig

        print(f"Building tile encoder ...")
        self.tile_encoder = Virchow(model_dir=self.model_dir, mode="full")

        cfg = PrismConfig(
            biogpt_config=BioGptConfig(),
            perceiver_config=PerceiverConfig(),
            model_dir=self.model_dir,
        )
        print(f"Building slide encoder ...")
        self.slide_encoder = Prism(cfg)

    def load_weights(self):
        checkpoint_path = self.model_dir / "prism-slide-encoder.pth"
        print(f"Loading slide encoder weights from {checkpoint_path}...")
        self.slide_encoder_weights = torch.load(
            checkpoint_path, map_location=self.device
        )
        updated_sd, msg = update_state_dict(
            model_dict=self.slide_encoder.state_dict(),
            state_dict=self.slide_encoder_weights,
        )
        print(msg)
        self.slide_encoder.load_state_dict(updated_sd, strict=True)
        self.slide_encoder.to(self.device)
        self.slide_encoder.eval()

    def forward_slide(self, tile_features):
        """Generate slide-level captions from tile embeddings."""
        tile_features = tile_features.unsqueeze(0)
        reprs = self.slide_encoder.slide_representations(tile_features)
        output = reprs["image_embedding"]  # [1, 1280]
        return output
