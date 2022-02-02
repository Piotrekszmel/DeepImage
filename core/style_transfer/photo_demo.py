from ast import arg
import cv2
import torch

from .net import Net
from .option import Options
from .utils import StyleLoader, predict_image

def run_demo(args, mirror=False):
    style_model = Net(ngf=args.ngf)
    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy()
    for key, _ in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    style_model.eval()
    if args.cuda:
        style_loader = StyleLoader(args.style_folder, args.style_size)
        style_model.cuda()
    else:
        style_loader = StyleLoader(args.style_folder, args.style_size, False)

    # Define the codec and create VideoWriter object
    if args.content_image:
        content_image = cv2.imread(args.content_image)
        pred_img = predict_image(content_image,
                            style_loader,
                            style_model,
                            mirror,
                            args.style_idx,
                            args.cuda)  
    else:
        height =  args.demo_size
        width = int(4.0 / 3 * args.demo_size)
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
                                    mirror,
                                    args.style_idx,
                                    args.cuda)
                cam.release()
                break
    img_name = "opencv_img_{}.png".format(args.style_idx)
    cv2.imwrite(img_name, pred_img)
    # cv2.imshow('MSG    Demo', pred_img)
    cv2.destroyAllWindows()

def main():
	# getting things ready
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")

	# run demo
	run_demo(args, mirror=True)

if __name__ == '__main__':
	main()
