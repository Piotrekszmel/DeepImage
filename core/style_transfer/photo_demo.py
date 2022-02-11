import cv2
import torch
import pygame
import pygame.camera

from .net import Net
from .utils import StyleLoader, predict_image

def predict(Config: dict) -> None:
    style_model = Net(ngf=Config["ngf"])
    model_dict = torch.load(Config["model"])
    model_dict_clone = model_dict.copy()
    for key, _ in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model.eval()
    style_loader = StyleLoader(Config["style_folder"], Config["style_size"])

    content_image = cv2.imread(Config["content_image"])
    pred_img = predict_image(content_image,
                                style_loader,
                                style_model,
                                Config["mirror"],
                                Config["style_idx"])  
    if "resize" in Config:
        if "new_height" in Config and "new_width" in Config:
            pred_img = cv2.resize(pred_img, (Config["new_width"], Config["new_height"]), interpolation = cv2.INTER_AREA)
    cv2.imwrite(Config["output_image"], pred_img)
    cv2.destroyAllWindows()


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