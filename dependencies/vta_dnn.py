"""
 VTA DNN model
"""

from dependencies.dnn import DNNmodel
from dependencies.utilities import Timer

import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import util

import vta
from vta.top import graph_pack

import logging as log
import sys

import Pyro4
import time
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

assert tvm.runtime.enabled("rpc")

class vta_DNN(DNNmodel):

    def __init__(self, env, pool_models, num_models):
        super(vta_DNN, self).__init__(env)

        self.__pool_models = pool_models
        self.__num_models = num_models
        self.__atari_env = env
        self.poolVTA = [    ['192.168.1.248', 12345, False],
                            #['192.168.2.12', 12345, False], 
                            #['192.168.2.13', 12345, False],
                            #['192.168.2.14', 12345, False],
                            #['192.168.2.15', 12345, False],
                        ]

        self.workersSem = Semaphore(len(self.poolVTA))
        self.compileSem = Semaphore(1)

        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.CRITICAL, stream=sys.stdout)
        log.getLogger('autotvm').setLevel(log.CRITICAL)

        self.env = vta.get_env()
        self.target = self.env.target
        
    def update_pool(self, pool_models):
        self.__pool_models = pool_models
        self.__num_models = len(pool_models)
        return

    @Timer(name="VTA-reconfig")
    def __reconfigFPGA(self, pynq_num):

        device_host = os.environ.get("VTA_RPC_HOST", self.poolVTA[pynq_num][0])
        device_port = os.environ.get("VTA_RPC_PORT", 9091)
        self.remote = rpc.connect(device_host, int(device_port))
        self.ctx = self.remote.ext_dev(0)
    
        reconfig_start = time.time()
        
        vta.reconfig_runtime(self.remote)
        vta.program_fpga(self.remote, bitstream="/Users/javier/Documents/Repos/TFM/dependencies/1x16_i8w8a32_15_15_18_17.bit")
        
        reconfig_time = time.time() - reconfig_start
        print("Reconfigured FPGA and RPC runtime in %.2fs for node %d" % (reconfig_time, pynq_num))
    
    @Timer(name="VTA-compile")
    def __compile_VTA(self, kerasModel):

        # Load pre-configured AutoTVM schedules
        with autotvm.tophub.context(self.target):

            # Populate the shape and data type dictionary for ImageNet classifier input
            dtype_dict = {kerasModel.input_names[0]: "float32"}
            shape_dict = {kerasModel.input_names[0]: (self.env.BATCH, 4, 84, 84)}

            # Start front end compilation
            mod, params = relay.frontend.from_keras(kerasModel, shape_dict)
            # Update shape and type dictionary
            shape_dict.update({k: v.shape for k, v in params.items()})
            dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

            # Perform quantization in Relay
            # Note: We set opt_level to 3 in order to fold batch norm
            with tvm.transform.PassContext(opt_level=3):
                with relay.quantize.qconfig(global_scale=8.0):
                    mod = relay.quantize.quantize(mod, params=params)
                # Perform graph packing and constant folding for VTA target
                assert self.env.BLOCK_IN == self.env.BLOCK_OUT
                relay_prog = graph_pack(
                    mod["main"],
                    self.env.BATCH,
                    self.env.BLOCK_OUT,
                    self.env.WGT_WIDTH,
                    start_name="nn.conv2d",
                    stop_name="nn.max_pool2d",
                    start_name_idx=8,
                    stop_name_idx=23
                )
                relay_prog = mod["main"]

            # Compile Relay program with AlterOpLayout disabled
            with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(relay_prog, target=self.target, params=params, target_host=self.env.target_host)

            # Send the inference library over to the remote RPC server
            temp = util.tempdir()
            lib.export_library(temp.relpath("graphlib.tar"))

            with open(temp.temp_dir + "/graphlib.tar","rb") as f:
                graphtar = f.read()

        return graphtar

    @Timer(name="VTA-episode")
    def run_episode(self):
        """ Run episode of pong (one game)
        Each episode run, each of networks in population will play
        the game and get the fitness(final reward when game is finished)
        """

        fitness = [-22 for _ in range(self.__num_models)] # Worst game score in a game is -21
        exec_times = [[0, 0] for _ in range(self.__num_models)] # Worst game score in a game is -21
        models_failed = list()

        executor = ThreadPoolExecutor(max_workers=len(self.poolVTA))
        
        print("Start...")

        future_to_fitness = {executor.submit(self.vtaPYRO, model_num): model_num for model_num in range(self.__num_models)}

        for future in as_completed(future_to_fitness):
            try:
                pyroOutput = future.result()
                model_num = pyroOutput[0]
                fitness[model_num] = pyroOutput[1][0]
                exec_times[model_num][0] = pyroOutput[1][1]
                exec_times[model_num][1] = pyroOutput[1][2]
            except Exception as exc:
                model_num = exc.args[1]
                id_node = exc.args[2]
                print('Model %d generated an exception: %s' % (model_num, exc.args[0]))
                models_failed.append(model_num)
                self.__reconfigFPGA(id_node)
            else:
                print("Game Over for model %d with average score %f. Average times: "
                    "DNN -> %.5fms, Env -> %.5fms. Iteration time: %.5fs. Load time: %.5fs" % 
                    (model_num, fitness[model_num], exec_times[model_num][0], exec_times[model_num][1],Timer().get_time("VTA-iteration" + "#" + str(model_num))[0], Timer().get_time("VTA-load" + "#" + str(model_num))[0]))

        # Segundo intento
        future_to_fitness = {executor.submit(self.vtaPYRO, model_num): model_num for model_num in models_failed}

        for future in as_completed(future_to_fitness):
            try:
                pyroOutput = future.result()
                model_num = pyroOutput[0]
                fitness[model_num] = pyroOutput[1][0]
                exec_times[model_num][0] = pyroOutput[1][1]
                exec_times[model_num][1] = pyroOutput[1][2]
            except Exception as exc:
                model_num = exc.args[1]
                id_node = exc.args[2]
                print('Model %d generated an exception: %s' % (model_num, exc.args[0]))
                fitness[model_num] = -22 # Marcamos el modelo con el peor fitness para no romper la ejecución del programa
            else:
                print("Game Over for model %d with average score %f. Average times: "
                    "DNN -> %.5fms, Env -> %.5fms. Iteration time: %.5fs. Load time: %.5fs" % 
                    (model_num, fitness[model_num], exec_times[model_num][0], exec_times[model_num][1],Timer().get_time("VTA-iteration" + "#" + str(model_num))[0], Timer().get_time("VTA-load" + "#" + str(model_num))[0]))

        return fitness

    def vtaPYRO(self, model_num):
        with self.workersSem:
            for id_node, _ in enumerate(self.poolVTA):
                if self.poolVTA[id_node][2] == False:
                    worker_id = id_node
                    self.poolVTA[id_node][2] = True
                    break

            try:
                dnn = DNNmodel(self.__atari_env)
                dnn.set_weights(self.__pool_models[model_num])

                with self.compileSem:
                    graphtar = self.__compile_VTA(dnn.kerasModel)

                uri = "PYRO:vta.pynq@" + self.poolVTA[worker_id][0] + ":" + str(self.poolVTA[worker_id][1])
                vtaProxy = Pyro4.Proxy(uri)

                with Timer(name="VTA-load" + "#" + str(model_num)):
                    loadOutput = vtaProxy.load_module(graphtar, dnn.kerasModel.input_names[0])
                    if(loadOutput != True):
                        raise Exception('Error loading DNN model!', loadOutput)

                with Timer(name="VTA-iteration" + "#" + str(model_num)):
                    vtaOutput = vtaProxy.run_episode()
                    if not(isinstance(vtaOutput[0], float)): # Si en la posición 0 recibimos un string es que se ha producido un error
                        raise Exception('Error running DNN model!', vtaOutput)

            except Exception as exc:
                self.poolVTA[id_node][2] = False
                raise Exception(exc.args[0], model_num, id_node)
            else:
                self.poolVTA[id_node][2] = False

        return (model_num, vtaOutput)

    def run_test(self):

        import numpy as np
        
        fitness = [-22 for _ in range(self.__num_models)] # Worst game score in a game is -21

        print("Start...")
        for model_num in range(self.__num_models):

            self.set_weights(self.__pool_models[model_num])

            uri = "PYRO:vta.pynq@pynq.micasa.local:12345"
            self.vtaProxy = Pyro4.Proxy(uri)

            graphtar = self.__compile_VTA(self.kerasModel)
            self.vtaProxy.load_module(graphtar, self.kerasModel.input_names[0])

            # Variables de entorno
            state = self.__atari_env.reset()
            total_reward = 0.0
            total_steps = 0
            action = 0            

            while True:

                self.__atari_env.render()

                action = self.vtaProxy.run_test()

                _, reward, done, _ = self.__atari_env.step(action)
                total_reward += reward

                if done:
                    fitness[model_num] = total_reward + (total_steps - 750) / 200
                    break

                total_steps += 1

            print("Game Over for model %d with average score %.3f" % (model_num, fitness[model_num]))

        return fitness