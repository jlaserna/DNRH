import tvm
from tvm.runtime import module
from tvm.contrib import graph_runtime

import numpy as np
import ctypes

import Pyro4
import base64

import logging as log
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dependencies.environment import make_env, Timer
from dependencies.fitness import fitnessGenerator

@Pyro4.expose
class VTA_PYNQ(object):
    def __init__(self, env_name="PongNoFrameskip-v4"):
        self.ctx = tvm.ext_dev(0)
        self.atariEnv = make_env(env_name)

        self.fitness_function = fitnessGenerator(env_name).getFitnessFunction()

        self.clean()

        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.CRITICAL, stream=sys.stdout)
        log.getLogger('autotvm').setLevel(log.CRITICAL)

    def load_module(self, tarmodule, input_layer_name):

        try:
            self.clean()

            self.input_layer_name = input_layer_name

            self.state = self.atariEnv.reset()

            graphtar = base64.b64decode(tarmodule['data'])

            with open('/tmp/VTAPacked.tar', 'wb') as f:
                f.write(graphtar)

            self.lib = module.load_module('/tmp/VTAPacked.tar')
            dll_path = "/home/xilinx/VTA/dependencies/tvm/build/libvta.so"
            ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL)

            self.VTAmodel = graph_runtime.GraphModule(self.lib["default"](self.ctx))
        except Exception as exc:
            return exc.args[0]
        else:
            return True
    
    def clean(self):
        try:
            if(os.path.exists('/tmp/VTAPacked.tar')):
                os.remove('/tmp/VTAPacked.tar')
            if(os.path.exists('/tmp/VTAPacked.tar.so')):
                os.remove('/tmp/VTAPacked.tar.so')
        except Exception:
            return 'Error cleaning environment'
        else:
            return

    def run_episode(self):
        """ Run episode of pong (one game)
        Each episode run, each of networks in population will play 
        the game and get the fitness(final reward when game is finished)
        """

        # Variables de entorno
        state = self.atariEnv.reset()
        rewards_list = list()
        action = 0

        try:
            while True:
            
                input = np.moveaxis(state, 2, 0)
                input = np.expand_dims(input, 0)

                self.VTAmodel.set_input(self.input_layer_name, input)

                with Timer(name="VTA-predict"):
                    self.VTAmodel.run()

                q_vals = self.VTAmodel.get_output(0).asnumpy()[0]

                action = np.argmax(q_vals)

                state, reward, done, _ = self.atariEnv.step(action)

                rewards_list.append(reward)

                if done:
                    # Fitness fuction
                    fitness = self.fitness_function(rewards_list)
                    break

        except Exception as exc:
            self.clean()
            return exc.args[0]
        else:
            self.clean()
            return (fitness, Timer().get_average_ms("VTA-predict"), Timer().get_average_ms("Env-step"))

    def run_test(self):

        input = np.moveaxis(self.state, 2, 0)
        input = np.expand_dims(input, 0)

        self.VTAmodel.set_input(self.input_layer_name, input)
        self.VTAmodel.run()
        q_vals = self.VTAmodel.get_output(0).asnumpy()[0]

        action = np.argmax(q_vals)

        self.state, _, _, _ = self.atariEnv.step(action)


        return int(action)

def main():
    Pyro4.Daemon.serveSimple(
            {
                VTA_PYNQ: "vta.pynq"
            },
            host = '192.168.1.248',
            port = 12345,
            ns = False)

if __name__=="__main__":
    main()
