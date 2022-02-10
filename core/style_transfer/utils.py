import os
import cv2
import numpy as np
import torch
from PIL import Image


def predict_image(img, style_loader, style_model, mirror, style_idx): 
    if mirror: 
        img = cv2.flip(img, 1)
    img = np.array(img).transpose(2, 0, 1)
    style = style_loader.get(style_idx)
    style_model.setTarget(style)
    img = torch.from_numpy(img).unsqueeze(0).float()
    img = style_model(img)
    img = img.clamp(0, 255).detach().numpy()
    return img.squeeze().transpose(1, 2, 0).astype('uint8')


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
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


def tensor_save_rgbimage(tensor, filename):
    img = tensor.clone().clamp(0, 255).detach().numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    return features.bmm(features_t) / (ch * h * w)


def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch - mean


def add_imagenet_mean_batch(batch):
    """Add ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    return batch + mean

def imagenet_clamp_batch(batch, low, high):
    batch[:,0,:,:].data.clamp_(low-103.939, high-103.939)
    batch[:,1,:,:].data.clamp_(low-116.779, high-116.779)
    batch[:,2,:,:].data.clamp_(low-123.680, high-123.680)


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    return batch.transpose(0, 1)


class StyleLoader:
    def __init__(self, style_folder, style_size):
        self.folder = style_folder
        self.style_size = style_size
        self.files = sorted(os.listdir(style_folder), key=str.casefold)
    
    def get(self, i):
        if type(i) == str:
            i = int(i)
        idx = i % len(self.files)
        filepath = os.path.join(self.folder, self.files[idx])
        style = tensor_load_rgbimage(filepath, self.style_size)    
        style = style.unsqueeze(0)
        return preprocess_batch(style)

    def size(self):
        return len(self.files)
