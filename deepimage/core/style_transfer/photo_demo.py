import cv2
import torch
import pygame
import pygame.camera

from .net import Net
from .utils import StyleLoader, predict_image

def predict(style_model: Net,
            style_loader: StyleLoader,
            content_image: str,
            output_image: str,
            mirror: bool,
            style_idx: int,
            new_height: int = 0,
            new_width: int = 0) -> None:
    
    content_image = cv2.imread(content_image)
    pred_img = predict_image(content_image,
                             style_loader,
                             style_model,
                             mirror,
                             style_idx)  
    if new_height and new_width:
        pred_img = cv2.resize(pred_img,
                              (new_width, new_height),
                              interpolation = cv2.INTER_AREA)
    cv2.imwrite(output_image, pred_img)


def make_photo(save_path: str) -> bool:
    pygame.camera.init()
    cameras = pygame.camera.list_cameras() #Camera detected or not
    if cameras:
        cam = pygame.camera.Camera(cameras[0],(800,600)) #/dev/video0
        cam.start()
        img = cam.get_image()
        pygame.image.save(img, save_path)
        cam.stop()
        return True
    else:
        return False