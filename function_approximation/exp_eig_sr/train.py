import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from common import CircularBuffer
from common import ExperienceReplay
from common.utils import add_simple_summary
from environments import AtariEnvironment
from .ssr_dqn import SsrDqn

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global params
flags.DEFINE_integer('random_seed', 1, 'Random seed')
flags.DEFINE_string('network_architecture', 'exp_eig_sr.def_large_arch',
                    "Module containing the neural network's specification")
# Policy params
flags.DEFINE_float('eps_initial', 1.0, 'Initial epsilon value')
flags.DEFINE_float('eps_final', 0.1, 'Final epsilon value')
flags.DEFINE_integer('eps_anneal_over', 1000000, 'Linearly anneal epsilon over (in terms of steps)')

# DQN Params
flags.DEFINE_integer('replay_buffer_size', 1000000, 'Maximum replay buffer size')
flags.DEFINE_integer('learning_freq', 4, 'Learning frequency (in terms of steps)')
flags.DEFINE_integer('frame_history_size', 4, 'Frame history length given as input')
flags.DEFINE_integer('batch_size', 32, 'Batch size to sample from experience replay')
flags.DEFINE_integer('target_update_freq', 40000, 'Target network update frequency (in terms of steps)')
flags.DEFINE_integer('replay_seed_frames', 50000, 'Steps to execute random policy so to seed replay buffer')

# PER Params
flags.DEFINE_boolean('prioritized_experience_replay', False, 'Enable prioritized experience replay')
flags.DEFINE_float('per_alpha', 0.6, 'Alpha for prioritized experience replay')
flags.DEFINE_float('per_beta_initial', 0.4, 'Initial beta for prioritized experience replay')
flags.DEFINE_float('per_beta_final', 1.0, 'Final beta value for prioritized experience replay')
flags.DEFINE_float('per_td_increment_bias', 0.000001, 'Epsilon value to add to TD errors to avoid not revisiting')
flags.DEFINE_float('per_learning_rate_scale', 0.25, 'Value to scale --learning_rate by when PER enabled')

# Successor Representation params
flags.DEFINE_float('beta', 0.025, 'Weight we trade-off the intrinsic and extrinsic reward')

# Experiment params
flags.DEFINE_integer('num_frames', 100000000, 'Number of total frames to run for')
flags.DEFINE_string('restore_dir', '', 'Directory to restore weights from')
flags.DEFINE_string('save_dir', 'results', 'Save episode results directory')
flags.DEFINE_integer('save_freq', 1000000, 'Number of frames to save the model')
flags.DEFINE_string('log_dir', 'logdir/', 'Log directory')
flags.DEFINE_integer('train_summary_freq', 64, 'How frequently to log training summary (in training updates)')
flags.DEFINE_boolean('disable_progress', False, 'Disable progress bar output')
flags.DEFINE_boolean('log_train_summary', True, 'Logs tensorboard training summaries')

# Debug params
flags.DEFINE_bool('debug', False, 'If you want to debug the code, more data will be saved')
flags.DEFINE_integer('freq_save_episode_data', 1, 'The data for debugging is saved for every episode mod X')

def get_intrinsic_reward(previous_intrinsic_reward, current_sr, ssr_dqn):
    return  1./np.linalg.norm(current_sr).flatten(),  1./np.linalg.norm(current_sr).flatten()
    if previous_intrinsic_reward is not None:
        intr_rew = np.linalg.norm(current_sr).flatten()
        delta_intr_rew = previous_intrinsic_reward - intr_rew
        delta_intr_rew = max(0.0, delta_intr_rew)
        return intr_rew, delta_intr_rew
    else:
        return np.linalg.norm(current_sr).flatten(), 0.0


def main(_):
    tf.reset_default_graph()
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    input_height, input_width = 84, 84

    env = AtariEnvironment(seed=FLAGS.random_seed)
    ssr_dqn = SsrDqn(FLAGS.network_architecture,
                     learning_rate_scale=FLAGS.per_learning_rate_scale
                      if FLAGS.prioritized_experience_replay else 1.)

    global_step = tf.train.create_global_step()
    increment_step = tf.assign_add(global_step, 1)

    def restore_weights(scaffold, sess):
        latest = tf.train.latest_checkpoint(FLAGS.restore_dir)
        tf.logging.info("Restoring weights from checkpoint %s" % latest)
        scaffold.saver.restore(sess, latest)
    init_fn = restore_weights if FLAGS.restore_dir != '' else None

    my_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    scaffold = tf.train.Scaffold(init_fn=init_fn, saver=my_saver)
    summary = tf.summary.FileWriter(FLAGS.log_dir)

    hooks = [tf.train.CheckpointSaverHook(
        FLAGS.log_dir,
        save_steps=FLAGS.save_freq,
        scaffold=scaffold)]
    sess = tf.train.SingularMonitoredSession(hooks=hooks, scaffold=scaffold)

    path = FLAGS.save_dir + '/episodeResults.csv'
    if not tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.MakeDirs(FLAGS.save_dir)
    elif tf.gfile.Exists(path):
        tf.gfile.Remove(path)

    episode_results = tf.gfile.GFile(path, 'w')
    episode_results.write('episode,reward,steps\n')

    per_alpha = FLAGS.per_alpha if FLAGS.prioritized_experience_replay else 0
    replay_buffer = ExperienceReplay(FLAGS.replay_buffer_size, input_height, input_width,
                                     FLAGS.frame_history_size, FLAGS.gamma, per_alpha)
    frame_history = CircularBuffer(FLAGS.frame_history_size, (input_height, input_width, 1), np.uint8)

    if FLAGS.debug:
        q_seen = []
        sr_seen = []
        frames_seen = []
        observed_scores = []
        reconstructed_frames = []
        intrinsic_rewards_seen = []

        if not os.path.exists(FLAGS.save_dir + '/q'):
            os.mkdir(FLAGS.save_dir + '/q')
        if not os.path.exists(FLAGS.save_dir + '/sr'):
            os.mkdir(FLAGS.save_dir + '/sr')
        if not os.path.exists(FLAGS.save_dir + '/frames'):
            os.mkdir(FLAGS.save_dir + '/frames')
        if not os.path.exists(FLAGS.save_dir + '/scores'):
            os.mkdir(FLAGS.save_dir + '/scores')
        if not os.path.exists(FLAGS.save_dir + '/intrinsic'):
            os.mkdir(FLAGS.save_dir + '/intrinsic')
        if not os.path.exists(FLAGS.save_dir + '/reconstructed'):
            os.mkdir(FLAGS.save_dir + '/reconstructed')

    initial_episodes = 0
    with tqdm(total=FLAGS.replay_seed_frames, disable=FLAGS.disable_progress) as pbar:
        seed_steps = 0
        while seed_steps < FLAGS.replay_seed_frames and not sess.should_stop():
            action = np.random.randint(env.num_actions())
            reward, next_state, terminal = env.act(action)

            next_state = sess.run(ssr_dqn.process_frame, feed_dict={ssr_dqn.image: next_state})
            frame_history.append(next_state)
            replay_buffer.append(next_state, action, reward, terminal)

            if FLAGS.debug:
                frames_seen.append(next_state)

            if terminal:
                pbar.update(env.episode_frames() // env.frame_skip())

                if FLAGS.debug:
                    # Saving data I want
                    observed_scores.append(env.episode_reward())

                    if initial_episodes % FLAGS.freq_save_episode_data == 0:
                        np.save(FLAGS.save_dir + '/frames/e_' + str(initial_episodes), frames_seen)
                        np.save(FLAGS.save_dir + '/scores/e_' + str(initial_episodes), observed_scores)

                    frames_seen = []

                initial_episodes += 1
                env.reset()

            seed_steps += 1

    assert len(replay_buffer) == FLAGS.replay_seed_frames
    assert len(frame_history) == FLAGS.frame_history_size
    env.reset()

    steps_count = tf.train.global_step(sess, global_step)
    episode_count = 0

    if FLAGS.debug:
        frames_seen = []

    previous_intrinsic_reward = None
    with tqdm(total=FLAGS.num_frames, disable=FLAGS.disable_progress) as pbar:
        while steps_count * env.frame_skip() < FLAGS.num_frames and not sess.should_stop():
            frame_stack = np.concatenate(frame_history, axis=2)

            # Epsilon greedy policy with epsilon annealing
            if steps_count < FLAGS.eps_anneal_over:
                # Only compute epsilon step while we're still annealing epsilon
                epsilon = FLAGS.eps_initial - steps_count * (
                (FLAGS.eps_initial - FLAGS.eps_final) / FLAGS.eps_anneal_over)
            else:
                epsilon = FLAGS.eps_final

            # Epsilon greedy policy
            if np.random.uniform() < epsilon:
                action = np.random.randint(0, env.num_actions())
            else:
                action = sess.run(ssr_dqn.action, feed_dict={ssr_dqn.X: [frame_stack]})

            reward, next_state, terminal = env.act(action)
            next_state, steps_count = sess.run([ssr_dqn.process_frame, increment_step],
                                               feed_dict={ssr_dqn.image: next_state})
            frame_history.append(next_state)

            # Intrinsic motivation part
            q_val, reconstructed_screen, current_sr, phi = sess.run(ssr_dqn.NN_target, feed_dict={
                ssr_dqn.X_t: [frame_stack], ssr_dqn.actions: [action]})
            previous_intrinsic_reward, curr_intrinsic_reward = get_intrinsic_reward(previous_intrinsic_reward, current_sr, ssr_dqn)
            reward = reward + FLAGS.beta * curr_intrinsic_reward
            reward = np.clip(reward, -1, 1)

            replay_buffer.append(next_state, action, reward, terminal)

            if FLAGS.debug:
                sr_seen.append(current_sr)
                frames_seen.append(next_state)
                intrinsic_rewards_seen.append(FLAGS.beta * curr_intrinsic_reward)
                reconstructed_frames.append(reconstructed_screen)
                # intrinsic_rewards_seen.append(intrinsic_reward)
                q_seen.append(sess.run(ssr_dqn.NN[0], feed_dict={ssr_dqn.X: [frame_stack], ssr_dqn.actions: [0]}))

            if steps_count % FLAGS.learning_freq == 0:
                beta = 0.
                # If prioritized experience replay is enabled linearly anneal beta towards its final value
                if FLAGS.prioritized_experience_replay:
                    beta = FLAGS.per_beta_initial + ((steps_count * FLAGS.frame_skip) * (
                    (FLAGS.per_beta_final - FLAGS.per_beta_initial) / FLAGS.num_frames))

                batch_indices, batch_curr_obs, batch_actions, batch_rewards, batch_next_obs, batch_terminals, \
                    batch_mc_returns, batch_importance_weights = replay_buffer.sample(
                        FLAGS.batch_size, beta=beta, segment_samples=FLAGS.prioritized_experience_replay)

                # Train the control algorithm
                training_op = [ssr_dqn.td_errors, ssr_dqn.train]
                if FLAGS.log_train_summary \
                        and (steps_count // FLAGS.learning_freq) % FLAGS.train_summary_freq == 0:
                    training_op.append(ssr_dqn.train_summary)

                result = sess.run(training_op, feed_dict={
                    ssr_dqn.X: batch_curr_obs,
                    ssr_dqn.actions: batch_actions,
                    ssr_dqn.rewards: batch_rewards,
                    ssr_dqn.X_t: batch_next_obs,
                    ssr_dqn.terminals: batch_terminals,
                    ssr_dqn.importance_weights: batch_importance_weights,
                    ssr_dqn.mc_return: batch_mc_returns
                })

                if FLAGS.prioritized_experience_replay:
                    replay_buffer.update_priorities(batch_indices, np.abs(result[0]) + FLAGS.per_td_increment_bias)
                if len(result) > 2:
                    summary.add_summary(result[-1], global_step=steps_count)

            if steps_count % FLAGS.target_update_freq == 0:
                sess.run(ssr_dqn.copy_to_target)

            if terminal:
                episode_results.write('%d,%d,%d\n' % (episode_count, env.episode_reward(), env.episode_frames()))
                episode_results.flush()
                pbar.update(env.episode_frames())

                add_simple_summary(summary, 'episode/reward', env.episode_reward(), episode_count)
                add_simple_summary(summary, 'episode/frames', env.episode_frames(), episode_count)

                if FLAGS.debug:
                    # Saving data I want
                    episode_idx = initial_episodes + episode_count
                    observed_scores.append(env.episode_reward())

                    if episode_idx % FLAGS.freq_save_episode_data == 0:
                        np.save(FLAGS.save_dir + '/q/e_' + str(episode_idx), q_seen)
                        np.save(FLAGS.save_dir + '/sr/e_' + str(episode_idx), sr_seen)
                        np.save(FLAGS.save_dir + '/frames/e_' + str(episode_idx), frames_seen)
                        np.save(FLAGS.save_dir + '/scores/e_' + str(episode_idx), observed_scores)
                        np.save(FLAGS.save_dir + '/intrinsic/e_' + str(episode_idx), intrinsic_rewards_seen)
                        np.save(FLAGS.save_dir + '/reconstructed/e_' + str(episode_idx), reconstructed_frames)

                    q_seen = []
                    sr_seen = []
                    frames_seen = []
                    reconstructed_frames = []
                    intrinsic_rewards_seen = []

                episode_count += 1
                env.reset()

    episode_results.close()
    tf.logging.info('Finished %d frames' % (steps_count * env.frame_skip()))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
