"""
 GPU DNN model
"""

# Plaidml framework
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from dependencies.cpu_dnn import cpu_DNN

class gpu_DNN(cpu_DNN):

    def __init__(self, env, pool_models, num_models, fitness_funtion, visualize=False, debug=False):
        super(gpu_DNN, self).__init__(env, pool_models, num_models, fitness_funtion, visualize, debug)
