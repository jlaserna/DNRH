"""
 Fitness funtions
"""
class fitnessGenerator:

    def __init__(self, env=None):

        self.fitness_functions = {   "PongNoFrameskip-v4" : (lambda input : sum(input)),
                                    #"PongNoFrameskip-v4" : self.discount_rewards,
                                }
        self.env = env

        return

    def getFitnessFunction(self):
        return self.fitness_functions[self.env]

    def discount_rewards(self, rewards):

        gamma = 0.5

        discounted_rewards = list()

        frame_counter = 6

        sum_of_rewards = 0
        for t in reversed(range(0, len(rewards))):

            if frame_counter > 0:
                sum_of_rewards = sum_of_rewards * gamma - rewards[t]
                discounted_rewards.append(rewards[t])
            else:
                if rewards[t] != 0: 
                    sum_of_rewards = -rewards[t]
                    discounted_rewards.append(rewards[t])
                    frame_counter = 6
                else:
                    sum_of_rewards = sum_of_rewards * gamma
                    discounted_rewards.append(sum_of_rewards)

            frame_counter -= 1
        
        return sum(discounted_rewards)
