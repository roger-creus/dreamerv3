import json
import jax
import embodied
import numpy as np
from IPython import embed

import gym
import cv2
from gym import spaces

import craftax
from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnvNoAutoReset
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset

class ResizeObservationWrapper(gym.Wrapper):
    def __init__(self, env, new_shape=None):
        super(ResizeObservationWrapper, self).__init__(env)
        self.new_shape = new_shape

        if new_shape is not None:
          self.new_shape = new_shape
          self.observation_space = spaces.Box(
              low=0, high=255,
              shape=(self.new_shape[0], self.new_shape[1], 3),
              dtype=np.uint8
          )
        else:
          self.observation_space = gym.spaces.Box(
            low=env.observation_space(env.default_params).low,
            high=env.observation_space(env.default_params).high,
            shape=env.observation_space(env.default_params).shape,
            dtype=np.float32
          )
          
        self.action_space = gym.spaces.Discrete(env.action_space().n)

    def reset(self):
        rng = jax.random.PRNGKey(np.random.randint(2**31))
        observation, self.state = self.env.reset(rng)
        return self._resize(observation)

    def step(self, action):
        rng = jax.random.PRNGKey(np.random.randint(2**31))
        observation, self.state, reward, done, info = self.env.step(rng, self.state, action)
        info['reward'] = np.array(reward)
        return self._resize(observation), np.array(reward), np.array(done), info

    def _resize(self, observation):
      if self.new_shape is not None:
        resized_observation = cv2.resize(np.array(observation), self.new_shape, interpolation=cv2.INTER_AREA)
        resized_observation = np.clip(resized_observation * 255, 0, 255).astype(np.uint8)
        return resized_observation
      else:
        return np.array(observation)

achievements = [
    "collect_wood",
    "place_table",
    "eat_cow",
    "collect_sapling",
    "collect_drink",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_plant",
    "defeat_zombie",
    "collect_stone",
    "place_stone",
    "eat_plant",
    "defeat_skeleton",
    "make_stone_pickaxe",
    "make_stone_sword",
    "wake_up",
    "place_furnace",
    "collect_coal",
    "collect_iron",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_arrow",
    "make_torch",
    "place_torch",
    "collect_sapphire",
    "collect_ruby",
    "make_diamond_pickaxe",
    "make_diamond_sword",
    "make_iron_armour",
    "make_diamond_armour",
    "enter_gnomish_mines",
    "enter_dungeon",
    "enter_sewers",
    "enter_vault",
    "enter_troll_mines",
    "enter_fire_realm",
    "enter_ice_realm",
    "enter_graveyard",
    "defeat_gnome_warrior",
    "defeat_gnome_archer",
    "defeat_orc_solider",
    "defeat_orc_mage",
    "defeat_lizard",
    "defeat_kobold",
    "defeat_knight",
    "defeat_archer",
    "defeat_troll",
    "defeat_deep_thing",
    "defeat_pigman",
    "defeat_fire_elemental",
    "defeat_frost_troll",
    "defeat_ice_elemental",
    "damage_necromancer",
    "defeat_necromancer",
    "eat_bat",
    "eat_snail",
    "find_bow",
    "fire_bow",
    "learn_fireball",
    "cast_fireball",
    "learn_iceball",
    "cast_iceball",
    "open_chest",
    "drink_potion",
    "enchant_sword",
    "enchant_armour"
]

class Craftax(embodied.Env):
  def __init__(self, task, size=None, logs=False, logdir=None, seed=None):
    assert task in ('pixels', 'symbolic')
    size = (64,64) if task == 'pixels' else None
    env_clss = CraftaxPixelsEnvNoAutoReset if task == 'pixels' else CraftaxSymbolicEnvNoAutoReset

    self._env = ResizeObservationWrapper(env_clss(), size)
    self._logs = logs
    self._logdir = logdir and embodied.Path(logdir)
    self._logdir and self._logdir.mkdir()
    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = achievements
    self._done = True

  @property
  def obs_space(self):
    obs_type = self._env.observation_space.dtype
    spaces = {
        'image': embodied.Space(obs_type, self._env.observation_space.shape),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_reward': embodied.Space(np.float32),
    }
    if self._logs:
      spaces.update({
          f'log_achievement_{k}': embodied.Space(np.int32)
          for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0
      self._done = False
      image = self._env.reset()
      return self._obs(image, 0.0, {}, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    self._reward += reward
    self._length += 1
    if self._done and self._logdir:
      self._write_stats(self._length, self._reward, info)
    return self._obs(
        image, reward, info,
        is_last=np.array(self._done),
        is_terminal=np.array(info['discount']) == 0)

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=np.array(is_first),
        is_last=np.array(is_last),
        is_terminal=np.array(is_terminal),
        log_reward=np.float32(info['reward'] if info else 0.0),
    )
    if self._logs:
      log_achievements = {
          f'log_achievement_{k}': info['achievements'][k] if info else 0
          for k in self._achievements}
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})
    return obs

  def _write_stats(self, length, reward, info):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        **{f'achievement_{k}': v for k, v in info['achievements'].items()},
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines)
    print(f'Wrote stats: {filename}')

  def render(self):
    return self._env.render()
