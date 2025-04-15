from pathlib import Path

import torch
import torch.nn as nn

from unicorn_baseline.vision.pathology.model_utils import update_state_dict
from unicorn_baseline.vision.pathology.titan.configuration_titan import TitanConfig
from unicorn_baseline.vision.pathology.titan.modeling_titan import Titan


class SlideFeatureExtractor(nn.Module):
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        super(SlideFeatureExtractor, self).__init__()
        self.build_encoders()
        self.set_device()
        for param in self.parameters():
            param.requires_grad = False

    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def build_encoders(self):
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


class TITAN(SlideFeatureExtractor):
    def __init__(self, model_dir: Path, input_size: int = 512):
        self.model_dir = model_dir
        super(TITAN, self).__init__(input_size)
        self.features_dim = 768

    def build_encoders(self):

        cfg = TitanConfig()
        self.slide_encoder = Titan(cfg)

        checkpoint_path = self.model_dir / "titan-slide-encoder.pth"
        print(f"Loading slide encoder weights from {checkpoint_path} ...")
        self.slide_encoder_weights = torch.load(checkpoint_path)
        updated_sd, msg = update_state_dict(
            model_dict=self.slide_encoder.state_dict(),
            state_dict=self.slide_encoder_weights
        )
        self.slide_encoder.load_state_dict(updated_sd, strict=True)
        print(msg)

        print(f"Building tile encoder ...")
        self.tile_encoder, self.eval_transform = self.slide_encoder.return_conch(
            self.model_dir
        )

    def get_transforms(self):
        return self.eval_transform

    def forward_slide(self, tile_features, tile_coordinates, tile_size_lv0):
        tile_features = tile_features.unsqueeze(0)
        tile_coordinates = tile_coordinates.unsqueeze(0)
        output = self.slide_encoder.encode_slide_from_patch_features(
            tile_features, tile_coordinates, tile_size_lv0
        )
        return output
