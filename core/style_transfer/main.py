##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys

import torch

import utils
from net import Net

from option import Options

def main():
    # figure out the experiments type
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    elif args.subcommand == 'eval':
        # Test the pre-trained model
        evaluate(args)

    else:
        raise ValueError('Unknow experiment type')


def check_paths(args):
    try:
        if not os.path.exists(args.vgg_model_dir):
            os.makedirs(args.vgg_model_dir)
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def evaluate(args):
    content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style = style.unsqueeze(0)    
    style = utils.preprocess_batch(style)

    style_model = Net(ngf=args.ngf)
    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy()
    for key, _ in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    if args.cuda:
        style_model.cuda()
        content_image = content_image.cuda()
        style = style.cuda()


    content_image = utils.preprocess_batch(content_image)
    style_model.setTarget(style)

    output = style_model(content_image)
    utils.tensor_save_bgrimage(output[0], args.output_image, args.cuda)


def fast_evaluate(args, basedir, contents, idx = 0):
    # basedir to save the data
    style_model = Net(ngf=args.ngf)
    style_model.load_state_dict(torch.load(args.model), False)
    style_model.eval()
    if args.cuda:
        style_model.cuda()
    
    style_loader = utils.StyleLoader(args.style_folder, args.style_size, 
        cuda=args.cuda)

    for content_image in contents:
        idx += 1
        content_image = utils.tensor_load_rgbimage(content_image, size=args.content_size, keep_asp=True).unsqueeze(0)
        if args.cuda:
            content_image = content_image.cuda()
        content_image = utils.preprocess_batch(content_image)

        for isx in range(style_loader.size()):
            style = style_loader.get(isx)
            style_model.setTarget(style)
            output = style_model(content_image)
            filename = os.path.join(basedir, "{}_{}.png".format(idx, isx+1))
            utils.tensor_save_bgrimage(output, filename, args.cuda)
            print(filename)


if __name__ == "__main__":
   main()
