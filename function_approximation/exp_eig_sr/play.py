import os
import sys
import tensorflow as tf
import numpy as np

from environments import AtariEnvironment
from common import CircularBuffer

from .exp_eig_sr import ExploreEigenvectorSR
from deep_sr.successor_representation import SR

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global params
flags.DEFINE_integer('random_seed', 1, 'random seed')
# DQN Specific
flags.DEFINE_integer('frame_history_size', 4, 'number of previous frames fed as input')
# Experiment params
flags.DEFINE_string('restore_dir', 'logdir/', 'directory to restore weights from')
flags.DEFINE_string('path_to_trajectory', None, 'path to directory with frames to be loaded/played')
flags.DEFINE_string('save_dir', '', 'directory I will save the data to generate the visualization later')
# Dummy variables which I don't use by I need
flags.DEFINE_integer('replay_buffer_size', 1000000, 'Maximum replay buffer size')
flags.DEFINE_integer('learning_freq', 4, 'Learning frequency (in terms of steps)')

action_labels = ["noop", "fire", "up", "right", "left", "down", "up-right", "up-left", "down-right", "down-left",
                 "up-fire", "right-fire", "left-fire", "down-fire", "up-right-fire", "up-left-fire", "down-right-fire",
                 "down-left-fire"]

np.set_printoptions(threshold=np.nan, linewidth=10000000)


def main(_):
    tf.reset_default_graph()
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    env = AtariEnvironment(seed=FLAGS.random_seed, record=True)
    sr = SR(84, 84, env.num_actions(), FLAGS.frame_history_size)
    eesr = ExploreEigenvectorSR(84, 84, env.num_actions(), FLAGS.frame_history_size)

    def load_pretrain(scaffold, sess):
        latest = tf.train.latest_checkpoint(FLAGS.restore_dir)
        tf.logging.info("Loading checkpoint %s" % latest)
        scaffold.saver.restore(sess, latest)

    to_use_trajectory = False
    if FLAGS.path_to_trajectory is not None:
        trajectory = np.load(FLAGS.path_to_trajectory)
        to_use_trajectory = True
        traj_idx = 0

    q_seen = []
    if not os.path.exists(FLAGS.save_dir + '/q'):
        os.mkdir(FLAGS.save_dir + '/q')
    sr_seen = []
    if not os.path.exists(FLAGS.save_dir + '/sr'):
        os.mkdir(FLAGS.save_dir + '/sr')
    frames_seen = []
    if not os.path.exists(FLAGS.save_dir + '/frames'):
        os.mkdir(FLAGS.save_dir + '/frames')
    intrinsic_rewards_seen = []
    if not os.path.exists(FLAGS.save_dir + '/intrinsic'):
        os.mkdir(FLAGS.save_dir + '/intrinsic')
    reconstructed_frames = []
    if not os.path.exists(FLAGS.save_dir + '/reconstructed'):
        os.mkdir(FLAGS.save_dir + '/reconstructed')

    top_eigenvector = np.load(FLAGS.restore_dir + "top_eigenvector_2_90000.npy")
    print(top_eigenvector)

    my_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    scaffold = tf.train.Scaffold(init_fn=load_pretrain, saver=my_saver)

    frame_history = CircularBuffer(FLAGS.frame_history_size, (84, 84, 1), np.uint8)
    with tf.train.SingularMonitoredSession(scaffold=scaffold) as sess:
        env.reset()
        for _ in range(FLAGS.frame_history_size):
            if to_use_trajectory:
                frame = np.expand_dims(trajectory[traj_idx], axis=2)
                traj_idx += 1
            else:
                frame = sess.run(eesr.process_frame, feed_dict={eesr.image: env._get_single_frame()})
            frame_history.append(frame)

        # I need to do this once to get the current state so I can compute the intrinsic reward
        frame_stack = np.concatenate(frame_history, axis=2)
        phi, reconstructed_output, sr_output = sess.run(sr.online_sr,
                                                        feed_dict={sr.X: [frame_stack], sr.actions: [0]})
        idx = 0
        next_state = frame
        terminal = False
        while not terminal:
            frame_stack = np.concatenate(frame_history, axis=2)
            action = sess.run(eesr.action, feed_dict={eesr.X: [frame_stack]})
            Q_vals = sess.run(eesr.Q, feed_dict={eesr.X: [frame_stack]})[0]

            if idx > 0:
                q_seen.append(Q_vals)
                sr_seen.append(sr_output)
                frames_seen.append(next_state)
                reconstructed_frames.append(reconstructed_output)

            if not to_use_trajectory: #I don't support the action here, thus the code is kind of bugged
                _, next_state, terminal = env.act(action)
                next_state = sess.run(eesr.process_frame, feed_dict={eesr.image: next_state})
            else:
                next_state =np.expand_dims(trajectory[traj_idx], axis=2)
                traj_idx += 1
                if traj_idx >= len(trajectory):
                    terminal = True
            if idx > 0:
                intrinsic_reward = np.dot(top_eigenvector.flatten(), (next_phi - phi).flatten())
                print(intrinsic_reward)
                intrinsic_rewards_seen.append(intrinsic_reward)

            frame_history.append(next_state)

            next_phi, reconstructed_output, sr_output = sess.run(sr.online_sr,
                                                                 feed_dict={sr.X: [frame_stack], sr.actions: [action]})

            idx += 1
            print("Frame", traj_idx)
            phi = np.copy(next_phi)

    np.save(FLAGS.save_dir + '/q/e_replay', q_seen)
    np.save(FLAGS.save_dir + '/sr/e_replay', sr_seen)
    np.save(FLAGS.save_dir + '/frames/e_replay', frames_seen)
    np.save(FLAGS.save_dir + '/intrinsic/e_replay', intrinsic_rewards_seen)
    np.save(FLAGS.save_dir + '/reconstructed/e_replay', reconstructed_frames)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
