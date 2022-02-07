from configparser import Interpolation
import cv2
import torch

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

    if args["content_image"]:
        content_image = cv2.imread(args["content_image"])
        pred_img = predict_image(content_image,
                                 style_loader,
                                 style_model,
                                 args["mirror"],
                                 args["style_idx"])  
    else:
        height =  args["demo_size"]
        width = int(4.0 / 3 * args["demo_size"])
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("photo")
        cam.set(3, width)
        cam.set(4, height)
        key = 0
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break
            cv2.imshow("photo", img)
            
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                pred_img = predict_image(img,
                                         style_loader,
                                         style_model,
                                         args["mirror"],
                                         args["style_idx"])
                cam.release()
                break
    
    if ("resize" in args):
        if ("new_height" in args and "new_width" in args):
            pred_img = cv2.resize(pred_img, (args["new_height"], args["new_width"]), interpolation = cv2.INTER_AREA)
    cv2.imwrite(args["output_image"], pred_img)
    cv2.destroyAllWindows()
