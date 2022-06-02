"""
 CPU DNN model
"""

from dependencies.dnn import DNNmodel
import numpy as np
import matplotlib.pyplot as plt

from dependencies.utilities import Timer

class cpu_DNN(DNNmodel):

    def __init__(   self, env, pool_models, num_models, fitness_funtion, 
                    visualize=False, debug=False
                    ):
        super(cpu_DNN, self).__init__(env)

        self.__pool_models = pool_models
        self.__num_models = num_models
        self.__env = env
        self.__visualize = visualize
        self.__debug = debug
        self.__fitness_function = fitness_funtion

    def update_pool(self, pool_models):
        self.__pool_models = pool_models
        self.__num_models = len(pool_models)
        return

    @Timer(name="DNN-episode")
    def run_episode(self, iterations=1):

        fitness = [0 for _ in range(self.__num_models)]

        for model_num in range(self.__num_models):

            self.set_weights(self.__pool_models[model_num])

            for _ in range(0, iterations):

                with Timer(name="DNN-iteration"):
                    # Variables de entorno
                    state = self.__env.reset()
                    rewards_list = list()
                    action = 0

                    while True:

                        if self.__visualize:
                            if self.__debug:
                                myState = np.moveaxis(state, 2, 0)
                                fig=plt.figure(figsize=(2, 2), dpi=300)
                                for i in range(4):
                                    fig.add_subplot(2, 2, i + 1)
                                    plt.imshow(myState[i])
                                    plt.axis('off')
                                
                                plt.draw()
                                plt.pause(0.0001)
                                plt.close()

                            else:
                                self.__env.render()

                        input = np.expand_dims(state, 0)
                        output = self.predict(input)
                        action = np.argmax(output)

                        state, reward, done, _ = self.__env.step(action)

                        rewards_list.append(reward)

                        if done:
                            # Fitness fuction
                            fitness[model_num] = self.__fitness_function(rewards_list)
                            break

            fitness[model_num] /= iterations

            if(self.__debug):
                print("Game Over for model %d with average score %.3f. Average times: "
                        "DNN -> %.5fms, Env -> %.5fms. Iteration time: %.5fs" % (model_num, fitness[model_num],
                        Timer().get_average_ms("DNN-predict"), Timer().get_average_ms("Env-step"), Timer().get_time("DNN-iteration")[-1]))
            else:
                print("Game Over for model %d with average score %.3f" % (model_num, fitness[model_num]))

        return fitness