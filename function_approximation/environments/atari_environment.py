import tensorflow as tf
import numpy as np

from ale_python_interface import ALEInterface
from common import CircularBuffer

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('rom', '', 'Rom file')
flags.DEFINE_float('sticky_prob', 0.25, 'sticky action probability')
flags.DEFINE_integer('mode', 0, 'mode of rom to play')
flags.DEFINE_integer('difficulty', 0, 'difficulty of rom mode')
flags.DEFINE_integer('frame_skip', 5, 'frame skip')
flags.DEFINE_boolean('display_screen', False, 'display screen')
flags.DEFINE_integer('frame_buffer_size', 2, 'frames to perform max over on')
flags.DEFINE_integer('max_num_frames_per_episode', 18000, 'max number of frames per episode')
flags.DEFINE_string('record_dir', 'screen_record', 'Directory to store screen and audio recording')

class AtariEnvironment:
    def __init__(self, seed=1, record=False):
        self.ale = ALEInterface()
        self.ale.setBool(b'display_screen', FLAGS.display_screen or record)
        self.ale.setInt(b'frame_skip', 1)
        self.ale.setBool(b'color_averaging', False)
        self.ale.setInt(b'random_seed', seed)
        self.ale.setFloat(b'repeat_action_probability', FLAGS.sticky_prob)
        self.ale.setInt(b'max_num_frames_per_episode', FLAGS.max_num_frames_per_episode)

        if record:
            if not tf.gfile.Exists(FLAGS.record_dir):
                tf.gfile.MakeDirs(FLAGS.record_dir)
            self.ale.setBool(b'sound', True)
            self.ale.setString(b'record_screen_dir', str.encode(FLAGS.record_dir))
            self.ale.setString(b'record_sound_filename', str.encode(FLAGS.record_dir + '/sound.wav'))
            self.ale.setInt(b'fragsize', 64)

        self.ale.loadROM(str.encode(FLAGS.rom))

        self.ale.setMode(FLAGS.mode)
        self.ale.setDifficulty(FLAGS.difficulty)

        self.action_set = self.ale.getLegalActionSet()

        screen_dims = tuple(reversed(self.ale.getScreenDims())) + (1,)
        self._frame_buffer = CircularBuffer(FLAGS.frame_buffer_size, screen_dims, np.uint8)

        self.reset()

    def _is_terminal(self):
        return self.ale.game_over()

    def _get_single_frame(self):
        stacked_frames = np.concatenate(self._frame_buffer, axis=2)
        maxed_frame = np.amax(stacked_frames, axis=2)
        expanded_frame = np.expand_dims(maxed_frame, 3)

        return expanded_frame

    def reset(self):
        self._episode_frames = 0
        self._episode_reward = 0

        self.ale.reset_game()
        for _ in range(FLAGS.frame_buffer_size):
            self._frame_buffer.append(self.ale.getScreenGrayscale())

    def act(self, action):
        assert not self._is_terminal()

        cum_reward = 0
        for _ in range(FLAGS.frame_skip):
            cum_reward += self.ale.act(self.action_set[action])
            self._frame_buffer.append(self.ale.getScreenGrayscale())

        self._episode_frames += FLAGS.frame_skip
        self._episode_reward += cum_reward
        cum_reward = np.clip(cum_reward, -1, 1)

        return cum_reward, self._get_single_frame(), self._is_terminal()

    def state(self):
        assert len(self._frame_buffer) == FLAGS.frame_buffer_size
        return self._get_single_frame()

    def num_actions(self):
        return len(self.action_set)

    def episode_reward(self):
        return self._episode_reward

    def episode_frames(self):
        return self._episode_frames

    def frame_skip(self):
        return FLAGS.frame_skip
