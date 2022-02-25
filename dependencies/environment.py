"""
 Environment wrappers
"""

import cv2
import gym
import gym.spaces
import numpy as np
import collections

from dependencies.utilities import Timer

class FireAndResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Wrapper usado para entornos donde el jugador debe presionar FIRE para comenzar."""
        super(FireAndResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class MaxFrameAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Procesamos el entorno cada n 'n=skip' fotogramas."""
        super(MaxFrameAndSkipEnv, self).__init__(env)
        # Tomamos las dos observaciones en bruto más recientes (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    @Timer(name="Env-step")
    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        img = np.reshape(frame, frame.shape).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToKeras(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToKeras, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=old_shape,
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=2),
                                                old_space.high.repeat(n_steps, axis=2), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer = np.moveaxis(self.buffer, 2, 0)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        self.buffer = np.moveaxis(self.buffer, 0, 2)
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxFrameAndSkipEnv(env)
    env = FireAndResetEnv(env)
    env = ProcessFrame(env)
    env = ImageToKeras(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
