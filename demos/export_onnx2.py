# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca4export import DECA
from decalib.datasets import datasets 
from decalib.utils.config import cfg as deca_cfg


def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config = deca_cfg, device=device)
    deca.eval()
    for param in deca.parameters():
        param.requires_grad = False

    data = testdata[0]
    name = data['imagename']
    images = data['image'].to(device)[None,...]
    with torch.no_grad():
        def fn(images):
            return deca.encode(images)
        print('Encoder input:', 'images', images.shape)
        encode_jit = torch.jit.trace(fn, images, strict=False)
        codedict = encode_jit(images)
        codedict_keys = ['codedict_' + k for k in codedict.keys()]
        codedict_values = list(codedict.values())
        for k, v in zip(codedict_keys, codedict_values):
            print('Encoder output:', 'codedict', k, v.shape)

        class Model(torch.nn.Module):
            def forward(self, images):
                retdict = fn(images)
                return tuple([retdict[k.split('codedict_')[-1]] for k in codedict_keys])
        torch.onnx.export(
            Model(), 
            images, 
            os.path.join(savefolder, 'deca_encoder.onnx'), 
            opset_version=12,
            input_names=['images'],
            output_names=codedict_keys
        )
        print('Encoder export success')
        
        tform =  data['tform'][None, ...]
        tform = torch.inverse(tform).transpose(1,2).to(device)
        original_image = data['original_image'][None, ...].to(device)
        def fn(codedict, original_image, tform):
            return deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)
        for k, v in zip(codedict_keys, codedict_values):
            print('Dncoder input:', 'codedict', k, v.shape)
        print('Decoder input:', 'original_image', original_image.shape)
        print('Decoder input:', 'tform', tform.shape)
        decode_jit = torch.jit.trace(fn, (codedict, original_image, tform), strict=False)
        opdict, visdict = decode_jit(codedict, original_image, tform)
        opdict_keys =  ['opdict_' + k for k in opdict.keys()]
        opdict_values = list(opdict.values())
        visdict_keys = ['visdict_' + k for k in visdict.keys()]
        visdict_values = list(visdict.values())
        for k, v in zip(opdict_keys, opdict_values):
            print('Decoder output:', 'opdict', k, v.shape)
        for k, v in zip(visdict_keys, visdict_values):
            print('Decoder output:', 'visdict', k, v.shape)

        class Model(torch.nn.Module):
            def forward(self, *args):
                codelist, original_image, tform = args[:-2], args[-2], args[-1]
                codedict = { k.split('codedict_')[-1]: v for k, v in zip(codedict_keys, codelist) }
                retdict1, retdict2 = fn(codedict, original_image, tform)
                return tuple(
                    [retdict1[k.split('opdict_')[-1]] for k in opdict_keys] + 
                    [retdict2[k.split('visdict_')[-1]] for k in visdict_keys]
                )
        torch.onnx.export(
            Model(), 
            (*codedict_values, original_image, tform), 
            os.path.join(savefolder, 'deca_decoder.onnx'), 
            opset_version=12,
            input_names=codedict_keys + ['original_image', 'tform'],
            output_names=opdict_keys + visdict_keys
        )
        print('Decoder export success')
        
        deca.render_images(codedict, opdict, visdict)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    main(parser.parse_args())
