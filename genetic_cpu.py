"""
 CPU genetic algorithm class
"""

from dependencies.cpu_dnn import cpu_DNN
from dependencies.genetic_algorithm import GeneticAlgorithm

class GeneticCPU(GeneticAlgorithm):
    def __init__(   self, num_generations = 1000, population = 1000, mutation_power = 0.005,
                    environment = "PongNoFrameskip-v4", debug = False, visualize = False,
                    output_dir = 'output', start_model_weights = None, re_evaluations = 1,
                    weight_seeds_path = None, record = False
                    ):
        super(GeneticCPU, self).__init__(   num_generations,population,mutation_power,
                                            environment,debug,visualize,output_dir,start_model_weights, 
                                            re_evaluations, weight_seeds_path, 'cpu', record
                                            )

    def run(self):

        DNN_model = cpu_DNN(    self.env, self.currentPool, self.population, 
                                self.fitness_function, self.visualize, self.debug 
                                )

        super().run(DNN_model)
        
        return