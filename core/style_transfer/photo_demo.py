from configparser import Interpolation
import cv2
import torch
import os
import pygame
import pygame.camera

from .net import Net
from .utils import StyleLoader, predict_image

def predict(args):
    style_model = Net(ngf=args["ngf"])
    model_dict = torch.load(args["model"])
    model_dict_clone = model_dict.copy()
    for key, _ in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model.eval()
    style_loader = StyleLoader(args["style_folder"], args["style_size"])

    content_image = cv2.imread(args["content_image"])
    pred_img = predict_image(content_image,
                                style_loader,
                                style_model,
                                args["mirror"],
                                args["style_idx"])  
    if ("resize" in args):
        if ("new_height" in args and "new_width" in args):
            pred_img = cv2.resize(pred_img, (args["new_width"], args["new_height"]), interpolation = cv2.INTER_AREA)
    cv2.imwrite(args["output_image"], pred_img)
    cv2.destroyAllWindows()


def make_photo(height: int,
               save_path: str) -> None:

    pygame.camera.init()
    pygame.camera.list_cameras() #Camera detected or not
    cam = pygame.camera.Camera("/dev/video0",(800,600))
    cam.start()
    img = cam.get_image()
    pygame.image.save(img, save_path)
    cam.stop()
    