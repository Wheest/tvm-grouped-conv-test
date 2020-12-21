from __future__ import print_function
import argparse
import pandas as pd
import pickle
import onnx
import torch
import os
import sys
import tvm
import datetime
from src.tvm_utils import Experiment, Device
from src import get_device_info
import tensorflow as tf

def main(args):
    df = pd.DataFrame(columns=['device_name', 'software_library',
                               'library_version', 'model',
                               'configurations', 'threads',
                               'compiler_optimisations',
                               'median_inf_time', 'tag',
                               'date_collected', 'trust'])
    os.environ["TVM_NUM_THREADS"]=str(args.threads)
    device_info = get_device_info(args.device_name)
    device_info['big_little'] = args.big_little
    device = Device(**device_info)
    device.connect()

    load_name = os.path.join(args.model_dir, 'models_data.pkl')
    with open(load_name, 'rb') as handle:
        input_data, model_fnames, target_outputs, \
            input_names, model_arch_names = pickle.load(handle)

    models = dict()
    file_types = dict()
    # print(f'Test, we are running {args.device_name}, our info is: {device_info}')
    # exit(1)
    for name, fname in model_fnames.items():
        #if name != #"WRN-40-2":#"G(16)":
        if name != "G(16)":
            continue
        print("loading", name)
        filepath = os.path.join(args.model_dir, fname)
        _, file_extension = os.path.splitext(fname)
        print(file_extension)
        if file_extension == '.onnx':
            models[name] = onnx.load(filepath)
            file_types[name] = 'onnx'
            # Check that the IR is well formed
            onnx.checker.check_model(models[name])
        elif file_extension == '.h5': # keras
            models[name] = tf.keras.models.load_model(filepath)
            file_types[name] = 'keras'
        elif file_extension == '.t7' or file_extension == '.pt':
            model = torch.load(filepath, map_location=torch.device('cpu'))
            tmp_data = torch.randn(input_data[name].shape)
            models[name] = torch.jit.trace(model, tmp_data).eval()
            file_types[name] = 'pytorch'
        else:
            raise ValueError(f'Unexpected model type `{file_extension}`')


    print("running experiments")
    experiments = []

    exper = Experiment(device, models, input_data, input_names, file_types,
                       valid_output=target_outputs, opt_level=args.opt_level,
                       runs=args.runs, sparse_cnn=args.sparse_cnn)
    exper.run_all()
    experiments.append(exper)

    times = exper.inf_time
    for model_name, _ in times.items():
        mean_time = exper.mean_time[model_name]
        exper_data = {'model': model_name,
                      'model_arch': model_arch_names[model_name],
                      'median_inf_time': exper.med_time[model_name],
                      'mean_inf_time': exper.mean_time[model_name],
                      'std': exper.std_time[model_name],
                      'runs': exper.runs,
                      'experiment_time': exper.exper_time[model_name],
        }
        print(exper)
        df = df.append(exper_data, ignore_index=True, sort=False)

    df.device_name = args.device_name
    df.software_library = 'tvm'
    df.library_version = tvm.__version__
    df.threads = args.threads
    df.date_collected = datetime.datetime.utcnow()
    df.configurations = f"autotune:0;opt:{str(args.opt_level)};thr:{str(args.threads)};{str(args.config)}"
    df.tag = args.tag
    print(df)
    df.to_csv(args.output_file, index=False)
    print('ran all experiments for tvm opt level {}'.format(args.opt_level))

    print('validate is ', not args.no_validate)
    if not args.no_validate:
        for exper in experiments:
            exper.validate(verbose=True)
    print('finished experiments!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TVM Evaluation')
    parser.add_argument('--device_name', default='local', type=str, help='friendly name for host, to use in results')
    parser.add_argument('--tag', default='vanilla',
                        help='Label to give set of experiments')
    parser.add_argument('--model_dir', default='/tmp/',
                        help='Directory where ONNX models and target input/outputs are stored')
    parser.add_argument('--host_type', choices=['local', 'remote'], default='local', help='will tvm run models locally, or on a remote device')
    # parser.add_argument('--host_key', default='', help='key used to access tvm RPC server')
    # parser.add_argument('--host_address', default='example.com', help='IP address, or domain of RPC server')
    # parser.add_argument('--port', default=2023, type=int, help='Port that RPC server is running on')
    # parser.add_argument('--target', default='llvm', type=str, help='Target (e.g., LLVM, OpenCL, etc)')
    # parser.add_argument('--target_string', default='None', type=str, help='llvm target string, if remote should be target triple')
    parser.add_argument('--big_little', action='store_true', help='Use big.LITTLE cores')
    parser.add_argument('--no_validate', action='store_true', help='Skip validating outputs of models')
    parser.add_argument('--sparse_cnn', action='store_true', help='Run the model with sparsity')
    parser.add_argument('--opt_level', default=3, type=int, help='tvm graph level optimisations')
    parser.add_argument('--output_file', default='/tmp/tvm_bench.csv', type=str,
                        help='Output file to save results')
    parser.add_argument('--runs', default=10, type=int, help='Number of repeats for each experiment')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads used (set via ENV variable on device)')
    parser.add_argument('--layer_output_dir', default='/tmp/tvm', type=str,
                            help='Directory to store per-layer information')
    parser.add_argument('--config', default='NoExtra', type=str,
                        help='Extra config string')
    args = parser.parse_args()
    main(args)
