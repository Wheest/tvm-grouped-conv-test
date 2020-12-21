import numpy as np
from torch.autograd import Variable
import torch
import onnx
import sys
import os
import argparse
import pickle
import torchvision.models as models
from models import *

parser = argparse.ArgumentParser(description='PyTorch models to ONNX format')
parser.add_argument('--model_set', default='resnet34',
                    help='Set of models to benchmark')
parser.add_argument('--output_dir', default='/tmp', type=str,
                    help='Output dir to save models')
parser.add_argument('--int_inputs', action='store_true', help='Generate input data of sequential integers')


resnet34_grouped = {
    'resnet34': models.__dict__['resnet34'](pretrained=True),
    'Res34-G(2)' : resnet34(conv=DConvA2,block=BasicBlock),
    'Res34-G(4)' : resnet34(conv=DConvA4,block=BasicBlock),
    'Res34-G(8)' : resnet34(conv=DConvA8,block=BasicBlock),
    'Res34-G(16)': resnet34(conv=DConvA16,block=BasicBlock),
    'Res34-G(N)' : resnet34(conv=DConv,block=BasicBlock),
}

wrn_40_2_cifar_grouped = {
    'WRN-40-2': WideResNet(40,2,conv=Conv,block=BasicBlock),
    'G(1)'    : WideResNet(40,2,conv=DConvA1,block=BasicBlock),
    'G(2)'    : WideResNet(40,2,conv=DConvA2,block=BasicBlock),
    'G(4)'    : WideResNet(40,2,conv=DConvA4,block=BasicBlock),
    'G(8)'    : WideResNet(40,2,conv=DConvA8,block=BasicBlock),
    'G(16)'   : WideResNet(40,2,conv=DConvA16,block=BasicBlock),
    'G(N)'    : WideResNet(40,2,conv=DConv,block=BasicBlock),
}

model_sets = {
    'resnet34': resnet34_grouped,
    'wrn-40-2_cifar10': wrn_40_2_cifar_grouped
}

data_shape_dict = {
    'resnet34': {'input': (1, 3, 224, 224), 'output': (1000,)},
    'wrn-40-2_cifar10': {'input': (1, 3, 32, 32), 'output': (10,)}
}

def main():
    # generate dummy data
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(0)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_dict = model_sets[args.model_set]

    for _, key in enumerate(model_dict.keys()):
        name = key.replace("/", "_over_") # > putting a forward slash in a UNIX filename

    shape_dict = data_shape_dict[args.model_set]
    target_outputs = dict()
    input_data = dict()

    print("loading PyTorch models...")
    for name, model in model_dict.items():
        in_shape = shape_dict['input']
        if not args.int_inputs:
            test_input = Variable(torch.rand(in_shape))
        else:
            data = np.zeros(in_shape, dtype=float)
            data = np.arange(data.size).reshape(in_shape)
            test_input = Variable(torch.from_numpy(data)).float()
        x = test_input

        model.eval()
        outs = model(x)
        target_outputs[name] = outs[0].detach().numpy()
        print(name, "outshape:", target_outputs[name].shape)

        np_inputs = x.numpy()
        input_data[name] = np_inputs

    model_fnames = dict()
    onnx_models = dict()
    input_names = dict()
    model_archs = dict()
    print("exporting to ONNX format...")
    for i, name in enumerate(model_dict):
        print(f'{name}, ({i+1}/{len(model_dict.keys())})')
        model = model_dict[name]

        in_shape = shape_dict['input']
        test_input = Variable(torch.rand(in_shape))
        x = test_input
        input_names[name] =  'input_1'
        # export to onnx
        model_fnames[name] = name + '.onnx'
        model_archs[name] = name
        save_name = os.path.join(args.output_dir, name + '.onnx')
        torch.onnx.export(model, x, save_name, input_names=[input_names[name]])

        # load onnx
        onnx_models[name] = onnx.load(save_name)

    print("exported models")
    models_data = (input_data, model_fnames, target_outputs, input_names, model_archs)
    save_name = os.path.join(args.output_dir, 'models_data.pkl')
    with open(save_name, 'wb') as f:
        pickle.dump(models_data, f)
    print("saved data")


if __name__ == '__main__':
    main()
