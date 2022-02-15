import os
from deepimage.core.style_transfer.net import Net
import cv2
import numpy as np
import torch
from PIL import Image


class StyleLoader:
    def __init__(self, style_folder: str, style_size: int) -> None:
        self.folder = style_folder
        self.style_size = style_size
        self.files = sorted(os.listdir(style_folder), key=str.casefold)
    
    def get(self, i: int) -> torch.Tensor:
        idx = i % len(self.files)
        filepath = os.path.join(self.folder, self.files[idx])
        style = tensor_load_rgbimage(filepath, self.style_size)    
        style = style.unsqueeze(0)
        return preprocess_batch(style)

    def size(self) -> int:
        return len(self.files)


def predict_image(img: np.ndarray,
                  style_loader: StyleLoader,
                  style_model: Net,
                  mirror: bool,
                  style_idx: int) -> np.ndarray: 
    if mirror: 
        img = cv2.flip(img, 1)
    img = np.array(img).transpose(2, 0, 1)
    style = style_loader.get(style_idx)
    style_model.setTarget(style)
    img = torch.from_numpy(img).unsqueeze(0).float()
    img = style_model(img)
    img = img.clamp(0, 255).detach().numpy()
    return img.squeeze().transpose(1, 2, 0).astype("uint8")


def tensor_load_rgbimage(filename: str,
                         size: int = None,
                         scale: int = None,
                         keep_asp: bool = False) -> torch.Tensor:
    img = Image.open(filename).convert("RGB")
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    return torch.from_numpy(img).float()


def preprocess_batch(batch: torch.Tensor) -> torch.Tensor:
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    return batch.transpose(0, 1)
