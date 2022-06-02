"""
 Genetic algorithm class
"""

import numpy as np
import copy
import skvideo
ffmpeg_path = "C:/Users/Javier/Documents/tfm/dependencies/ffmpeg/bin"
skvideo.setFFmpegPath(ffmpeg_path)
import skvideo.io

from dependencies.environment import make_env
from dependencies.utilities import mkdir, Timer
from dependencies.dnn import DNNmodel
from dependencies.fitness import fitnessGenerator

import datetime

class GeneticAlgorithm():
    def __init__(   self, num_generations = 1000, population = 1000, mutation_power = 0.005,
                    environment = "PongNoFrameskip-v4", debug = False, visualize = False,
                    output_dir = 'output', start_model_weights = None, re_evaluations = 5,
                    weight_seeds_path = None, arch = None, record = False
                ):
                    
        """
         Variables for genetic algorithms
            num_generations     -> Number of times to evole the population.
            population          -> Number of networks in each generation.
            model_to_keep       -> Models to keep between generations
            mutation_power      -> Mutation percentage 
            re_evaluations      -> Number of times to re-evaluate the parents
        """

        self.num_generations = num_generations
        self.population = population
        self.model_to_keep = int(population * 0.2)
        self.mutation_power = mutation_power
        self.re_evaluations = re_evaluations
        self.environment = environment
        self.debug = debug
        self.visualize = visualize
        self.output_dir = output_dir
        self.start_model_weights = start_model_weights
        self.weight_seeds_path = weight_seeds_path
        self.generation = 0
        self.record = record

        self.fitness_function = fitnessGenerator(environment).getFitnessFunction()

        timeStamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.output_dir += str('/' + str(timeStamp) + '#' + str(arch) + '#' + environment)

        mkdir('.', str(self.output_dir))
        mkdir('.', str(self.output_dir) + '/video')
        mkdir('.', str(self.output_dir) + '/best_models')

        with open(str(self.output_dir) + "/Genetic_generation_score.txt", "w") as text_file:
            text_file.write("{}, {}".format("generation","max_fitness"))
            text_file.write("\n")
            text_file.write("="*35)
            text_file.write("\n")

        # Array con los pesos de la población
        self.currentPool = list()

        # Creamos entorno de atariPy-Keras
        self.env = make_env(self.environment)

        if self.weight_seeds_path:
            self.weight_seeds = np.load(self.weight_seeds_path, allow_pickle=True)
            np.random.seed(self.weight_seeds[-1])
        else:
            self.weight_seeds = None

        print("Init First random population")
        # Iniciamos la población de los modelos
        self.currentPool = self.init_population(self.weight_seeds)
        print("Population size: ",len(self.currentPool))

        if self.start_model_weights:
            for idx in range(self.model_to_keep):
                self.currentPool[idx] = np.load(self.start_model_weights, allow_pickle=True)
            
    @Timer(name="GenAlg-init")
    def init_population(self, seeds=None):

        poolOfWeights = list()

        for idx in range(self.population):
            if len(seeds) != 0:
                poolOfWeights.append(DNNmodel(self.env, seeds[idx]).get_weights())
            else:
                poolOfWeights.append(DNNmodel(self.env).get_weights())

        return poolOfWeights

    def save_pool(self, best_model, score):
        np.save(str(self.output_dir) + "/best_models/model_best_in_generation" + str(self.generation) + "_score_" + str(score) + ".npy", best_model)
        print("Saved Best model!")

    def run_game(self, model, record = False):    
        """
        Play one pong game given a trained model
        """
        
        # Variables de entorno
        state = self.env.reset()

        DNN = DNNmodel(self.env)
        DNN.set_weights(model)

        if record:
            name = str(self.output_dir) + "/video/genetic_pong_generation_" + str(self.generation) + ".mp4"
            writer = skvideo.io.FFmpegWriter(name)

        while True:

            if self.visualize:
                self.env.render()

            if record:
                writer.writeFrame(self.env.render(mode='rgb_array'))

            input = np.expand_dims(state, 0)
            output = DNN.kerasModel.predict(input, batch_size=1)[0]
            action = np.argmax(output)
            state, _, done, _ = self.env.step(action)

            if done:
                break
        
        if record:
            writer.close()

    @Timer(name="GenAlg-mutate")
    def mutate(self,weights,mutation_power):
        """
        Add Gaussian noise to weights with factor (mutation_power)
        """
        for layers in [0,2,4,6,8]:
            
            mean,std = np.mean(weights[layers]), np.std(weights[layers]) 
            change = np.random.normal(mean,std,weights[layers].shape) # Gaussian noise 
            change = change * mutation_power
                        
            weights[layers] += change

        return weights

    def run(self, DNN_model):

        for _ in range(self.num_generations+1):
            """ Train models __num_generations times 
            """

            print("Running Generation: ", self.generation)
            print("="*70)

            DNN_model.update_pool(self.currentPool)

            # Lanzo una iteración para toda la población
            with Timer(name="GenAlg-episode"):
                fitness = DNN_model.run_episode()

            if(self.debug):
                print("Episode execution time: %.3fm" % (Timer().get_time("GenAlg-episode")[self.generation] / 60))

            print("Start training")
            print("*"*70)
            sorted_models = [x for _, x in sorted(zip(fitness,self.currentPool.copy()), key=lambda pair: pair[0],reverse=True)]
            parent_models = sorted_models[:self.model_to_keep] #keep only a percentage of the models
            new_gen_models = list()

            while len(new_gen_models) < self.population:
                # Higher the fitness score higher chance it is selected 
                selected_parent_id = np.random.choice(list(range(self.model_to_keep))) 

                new_gen_models.append(self.mutate(copy.deepcopy(parent_models[selected_parent_id]),self.mutation_power))

            max_fitness, min_fitness = np.max(fitness), np.min(fitness)
            print("Best model in this generation: ", np.argmax(fitness))
            print(max_fitness, min_fitness)

            print("Start parents re-evaluation")
            print("-"*70)
            # Lanzo n iteraciones para la población mejores modelos
            DNN_model.update_pool(parent_models)
            fitness_elite = DNN_model.run_episode(self.re_evaluations)
            elite_sorted_models = [x for _, x in sorted(zip(fitness_elite,parent_models.copy()), key=lambda pair: pair[0],reverse=True)]
            best_model = elite_sorted_models[0]

            # Elitism
            new_gen_models[0] = copy.deepcopy(best_model)

            self.currentPool = new_gen_models

            print("Finished training")

            if self.generation % 10 == 0:
                print("Saving gameplay")
                self.run_game(best_model, self.record)

            if max_fitness > 0:
                print("Saving best model and gameplay")
                self.save_pool(best_model, max_fitness)
                self.run_game(best_model, self.record)

            with open(str(self.output_dir) + "/Genetic_generation_score.txt", "a") as text_file:
                if (self.debug):
                    text_file.write("{}, {}, {}m".format(self.generation,max_fitness,Timer().get_time("GenAlg-episode")[self.generation] / 60))
                else:
                    text_file.write("{}, {}".format(self.generation,max_fitness))
                text_file.write("\n")
                
            self.generation += 1
            print("Finish current generation")
            print("Current best game score: ", max_fitness)
            print("_"*70)

        if(self.debug):
            print("Simulation execution time: %.3fh" % (sum(Timer().get_time("GenAlg-episode")) / 3600))

            with open(str(self.output_dir) + "/Genetic_generation_score.txt", "a") as text_file:
                text_file.write("Simulation execution time: %.3fh" % (sum(Timer().get_time("GenAlg-episode")) / 3600))
                text_file.write("\n")

        return
