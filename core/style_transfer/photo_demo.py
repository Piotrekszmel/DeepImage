import cv2
import numpy as np
import torch
from torch.autograd import Variable

from net import Net
from option import Options
from utils import StyleLoader


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
    height =  args.demo_size
    width = int(4.0 / 3 * args.demo_size)
    swidth = int(width / 4)
    sheight = int(height / 4)
    # if args.record:
    #     fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    #     out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (2*width, height))
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("photo")
    cam.set(3, width)
    cam.set(4, height)
    key = 0
    img_counter = 0
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
            # img_name = "opencv_img_{}.png".format(img_counter)
            # cv2.imwrite(img_name, img)
            # print("{} written!".format(img_name))
            # img_counter += 1
            if mirror: 
                img = cv2.flip(img, 1)
            cimg = img.copy()
            img = np.array(img).transpose(2, 0, 1)

            style = style_loader.get(8)
            style_model.setTarget(style)

            img = torch.from_numpy(img).unsqueeze(0).float()
            if args.cuda:
                img = img.cuda()

            img = style_model(img)

            if args.cuda:
                simg = style.cpu()[0].numpy()
                img = img.cpu().clamp(0, 255)[0].numpy()
            else:
                simg = style.numpy()
                img = img.clamp(0, 255).detach().numpy()
            simg = np.squeeze(simg)
            img = img.squeeze().transpose(1, 2, 0).astype('uint8')
            simg = simg.transpose(1, 2, 0).astype('uint8')

            # display
            simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
            cimg[0:sheight, 0:swidth, :] = simg
            #img = np.concatenate((cimg, img),axis=1)
            img_name = "opencv_img_{}.png".format(img_counter)
            cv2.imwrite(img_name, img)
            cv2.imshow('MSG Demo', img)
            key = cv2.waitKey(1)
            if key % 256 == 27:
                break

    cam.release()
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
