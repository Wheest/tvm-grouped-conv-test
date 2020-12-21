import os
import time
import tvm
import tvm.relay as relay
from tvm.contrib import utils, graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime
from tvm import rpc
import numpy as np
import timeout_decorator
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime
from tvm.relay import data_dep_optimization as ddo

import torch

class Device:
    'Container object for tvm rpc device data'
    def __init__(self, name, host_type, key=None, address=None,
                 port=None, target='llvm', target_string=None,
                 big_little=False):
        self.name = name
        self.host_type = host_type
        self.host_key = key
        self.host_address = address
        self.port = port
        self.target = target
        self.ctx = None
        self.target_host = target_string
        self.host_type = host_type
        self.big_little = big_little

    @timeout_decorator.timeout(10, exception_message="Could not connect "\
                               + "to tvm RPC server. Please check " \
                               + "all parameters (especially the actual " \
                               + "port number) and try again.")
    def connect(self):
        if self.host_type == 'local':
            self.session = rpc.LocalSession()
        elif self.host_type == 'remote':
            self.session = rpc.connect(self.host_address, self.port,
                                       key=self.host_key)
        else:
            raise ValueError('Unexpected host type: "{}"'.format(self.host_type))

        if 'opencl' in self.target:
            self.ctx = self.session.cl(0)
        else:
            self.ctx = self.session.cpu(0)
        if self.target == 'rasp3b':
            self.target = tvm.target.arm_cpu('rasp3b')
        elif self.target == 'llvm':
            self.target = 'llvm'
        elif self.target == 'opencl -device=mali':
            self.target = tvm.target.mali()
        else:
            self.target = tvm.target.create(self.target)




        if self.big_little:
            config_func = self.session.get_function('runtime.config_threadpool')
            # config_func(1, 1) # use 1 big core
            # config_func(1, 2) # use 2 big cores
            #config_func(-1, 1) # use 1 small core
            # config_func(-1, 2) # use 2 small cores
            config_func(4, 4)
        print("connected to host '{}'!".format(self.name))


class Experiment:
    'Object for tvm rpc experiments using mixed model types'
    'Give a dict of models, and use run_all'
    def __init__(self, host, models, input_data, input_names,
                 file_types,
                 opt_level=3, valid_output=None, runs=25,
                 sparse_cnn=False,
                 layer_debug=False, layer_output_dir='/tmp'):
        self.host = host
        self.models = models
        self.input_data = input_data
        self.input_names = input_names
        self.opt_level = opt_level
        self.valid_output = valid_output
        self.exper_time = dict() # time to run experiment
        self.inf_time = dict()  # median infernece time in ms
        self.mean_time = dict()  # mean infernece time in ms
        self.med_time = dict()  # median infernece time in ms
        self.std_time = dict()  # std infernece time in ms
        self.outputs = dict()
        self.runs = runs
        self.file_types = file_types

        self.sparse_cnn = sparse_cnn
        self.layer_debug = layer_debug
        self.layer_output_dir = layer_output_dir

    def run_single_model(self, model_name, model):
        inputs = self.input_data[model_name]
        print(self.input_names)
        shape_dict = {self.input_names[model_name]: inputs.shape}
        print(self.input_names[model_name], inputs.shape)

        print(shape_dict)
        print(type(model))
        if self.file_types[model_name] == 'onnx':
            syms, params = relay.frontend.from_onnx(model, shape_dict)
        elif self.file_types[model_name] == 'keras':
            syms, params = relay.frontend.from_keras(model, shape_dict)
        elif self.file_types[model_name] == 'pytorch':
            shapes_list = list = [(k, v) for k, v in shape_dict.items()]
            syms, params = relay.frontend.from_pytorch(model, shapes_list)
        target = self.host.target
        target_host = self.host.target_host

        if self.sparse_cnn:
            print('converting sparsity')
            syms, params = ddo.simplify_fc_transpose.convert(syms["main"], params)
            syms, params = ddo.csr_conv2d.convert(syms, params, sparsity_threshold=0.0)
        with relay.build_config(opt_level=self.opt_level,
                                # disabled_pass=[
                                #     "FoldConstant",
                                # ],
                                # required_pass=[
                                #     "SimplifyInference",
                                #     "OpFusion",
                                #     # "FoldConstant",
                                #     "FoldScaleAxis",
                                #     "AlterOpLayout",
                                #     "CanonicalizeOps",
                                #     "CanonicalizeCast",
                                #     "EliminateCommonSubexpr",
                                #     "CombineParallelConv2D",
                                #     "CombineParallelDense",
                                #     "CombineParallelBatchMatmul",
                                #     "FastMath"
                                # ]
        ):
            graph, lib, params = relay.build_module.build(
                syms, target, params=params,
                target_host=target_host)

        # After `relay.build`, you will get three return values: graph,
        # library and the new parameter, since we do some optimization that will
        # change the parameters but keep the result of model as the same.

        # Save the library at local temporary directory.
        tmp = utils.tempdir()
        tarname = str(model_name) + '_' + str(self.host.name) + '.tar'
        lib_fname = tmp.relpath(tarname)
        lib.export_library(lib_fname)

        # obtain an RPC session from remote device.
        remote = self.host.session

        # upload the library to remote device and load it
        remote.upload(lib_fname)
        rlib = remote.load_module(tarname)

        # create the remote runtime module
        ctx = self.host.ctx
        if not self.layer_debug:
            module = runtime.create(graph, rlib, ctx)
        else:
            module = debug_runtime.create(graph, rlib, ctx,
                                          dump_root=self.layer_output_dir)

        # set parameter (upload params to the remote device.
        # This may take a while)
        module.set_input(**params)
        # set input data
        module.set_input(key=self.input_names[model_name], value=inputs)

        # run
        print('running bruh')
        module.run()
        # get output
        out = module.get_output(0)
        self.outputs[model_name] = out.asnumpy()

        # get median inference time
        # sample mean is skewed in this setting by potentially unbounded
        # high outliers, thus we get the median
        print('running avg bruh')
        start = time.time()
        f = module.module.time_evaluator('run', ctx, number=10, repeat=self.runs)
        results = f().results
        median = np.median(results) * 1000
        mean = np.mean(results) * 1000
        self.inf_time[model_name] = median
        self.med_time[model_name] = median
        self.mean_time[model_name] = mean
        self.std_time[model_name] = np.std(results) * 1000
        self.exper_time[model_name] = (time.time() - start) * 1000
        print('ran a model bruh')


    def run_all(self):
        for _, model_name in enumerate(self.models):
            model = self.models[model_name]
            self.run_single_model(model_name, model)

    def validate(self, verbose=False):
        'Validate the output of models from some baseline'
        'probably from the original PyTorch model given the same input'
        absolute_tolerance = 0.0001
        relative_tolerance = 0.0001
        for name, tvm_outs in self.outputs.items():
            targets = self.valid_output[name]
            valid = np.allclose(targets, tvm_outs,
                                rtol=relative_tolerance,
                                atol=absolute_tolerance)
            if not valid:
                print("oh no! {} is not valid!".format(name))
                if verbose:
                    print(f'first 100 outputs:\n{tvm_outs.flatten()[0:100]}')
                    print(f'first 100 targets:\n{targets[0:100]}')
