import importlib
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Optimizer params (RMSProp)
flags.DEFINE_float('learning_rate', 0.00025, 'learning rate')
flags.DEFINE_float('rmsprop_decay', 0.95, 'rmsprop gradient momentum')
flags.DEFINE_float('rmsprop_epsilon', 0.000009765625, 'rmsprop denominator epsilon')

# RL Params
flags.DEFINE_float('gamma', 0.99, 'discount rate')

# MC-Return Params
flags.DEFINE_float('mc_return_beta', 0.1, 'Trade-off between the TD-error (1 - beta) and the MC return (beta)')

# NN Params
flags.DEFINE_float('wgt_td_loss', 1.0, "weight given to the squared td error when training the whole neural network")
flags.DEFINE_float('wgt_sr_loss', 1000.0, 'weight given to the SR loss when training the whole neural network')
flags.DEFINE_float('wgt_recons_loss', 0.001, 'weight given to the reconstruction loss when training the whole neural net')


class SsrDqn:
    def __init__(self, path_to_arch_params, learning_rate_scale=1.):
        """
        input_height: Downsized input height
        input_width: Downsized input width
        num_actions: Minimal action set length
        learning_rate_scale: Used to scale the learning rate, what Schaul et al. (2016) call \eta.
                             This is utilized in prioritized experience replay.
        """
        self.params = self.conf = importlib.import_module(path_to_arch_params)
        self._input_height = self.params.input_height
        self._input_width = self.params.input_width
        self._num_actions = self.params.num_actions
        self._learning_rate_scale = learning_rate_scale
        self._frame_history_size = self.params.frame_history_size
        self._total_num_updates = int(FLAGS.replay_buffer_size / FLAGS.learning_freq)

        self.train_summary = []

        self.X = tf.placeholder(tf.float32, shape=[None, self._input_height, self._input_width,
                                                   self._frame_history_size], name='ssr_dqn/inputs')
        self.actions = tf.placeholder(tf.uint8, shape=[None], name='ssr_dqn/actions')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name='ssr_dqn/rewards')
        self.mc_return = tf.placeholder(tf.float32, shape=[None], name='ssr_dqn/mc_return')
        self.X_t = tf.placeholder(tf.float32, shape=[None, self._input_height, self._input_width,
                                                     self._frame_history_size], name='ssr_dqn/target_inputs')
        self.terminals = tf.placeholder(tf.float32, shape=[None], name='ssr_dqn/terminals')

        # Importance weights to scale loss by, used in prioritized experience replay
        self.importance_weights = tf.placeholder(tf.float32, [None], name='ssr_dqn/importance_weights')

        self.image = tf.placeholder(tf.uint8, shape=[None, None, 1], name='ssr_dqn/raw_image')

        self.NN = self.build_nn()
        self.NN_prime = self.build_nn_prime()
        self.NN_target = self.build_nn_target()

        self.train = self.build_train()

        self.action = self.build_action()
        self.copy_to_target = self.build_copy()
        self.process_frame = self.build_process_frame()

        self.train_summary = tf.summary.merge(self.train_summary)
        tf.logging.info("Built DQN graph")

    def build_q_network(self, X, A, prefix):
        """
        Builds architecture to learn the successor representation.

        Parameters:
          X: Scope for operations. Please wrap all operations inside variable_scope. See SR.build_copy
          A:
          prefix:

        Returns: Final output layer which can be fed X as input
        """
        conv_input = tf.divide(X, 255.)  # Normalize inputs

        # initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG', uniform=True)
        with tf.variable_scope(prefix + "_representation_learning"):
            conv_input = tf.pad(tensor=conv_input,
                                paddings=[[0, 0], self.params.padding_conv1, self.params.padding_conv1, [0, 0]],
                                name=prefix + "_pad_conv1")
            conv1 = tf.layers.conv2d(conv_input,
                                     filters=self.params.num_filters_conv1,
                                     kernel_size=self.params.kernel_size_conv1,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     strides=self.params.stride_conv1,
                                     activation=tf.nn.relu)
            aug_conv1 = tf.pad(tensor=conv1,
                               paddings=[[0, 0], self.params.padding_conv2, self.params.padding_conv2, [0, 0]],
                               name=prefix + "_pad_conv2")
            conv2 = tf.layers.conv2d(aug_conv1,
                                     filters=self.params.num_filters_conv2,
                                     kernel_size=self.params.kernel_size_conv2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     strides=self.params.stride_conv2,
                                     activation=tf.nn.relu)
            aug_conv2 = tf.pad(tensor=conv2,
                               paddings=[[0, 0], self.params.padding_conv3, self.params.padding_conv3, [0, 0]],
                               name=prefix + "_pad_conv3")
            conv3 = tf.layers.conv2d(aug_conv2,
                                     filters=self.params.num_filters_conv3,
                                     kernel_size=self.params.kernel_size_conv3,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     strides=self.params.stride_conv3,
                                     activation=tf.nn.relu)
            conv_flat = tf.contrib.layers.flatten(conv3)

            phi = tf.layers.dense(inputs=conv_flat,
                                  units=self.params.num_nodes_fc_representation,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  bias_initializer=tf.constant_initializer(0.0),
                                  activation=tf.nn.relu)

            normalized_phi = tf.nn.l2_normalize(x=phi, dim=1)

        with tf.variable_scope(prefix + "_action_values"):
            value_function = tf.layers.dense(normalized_phi,
                                             units=self._num_actions,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             bias_initializer=tf.constant_initializer(0.0))

        with tf.variable_scope(prefix + "_reconstruction_module"):

            fc2 = tf.layers.dense(inputs=normalized_phi,
                                  units=self.params.num_nodes_fc_in_reconstruction,
                                  kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                  bias_initializer=tf.constant_initializer(0.0),
                                  activation=tf.nn.relu)

            action_taken = tf.one_hot(indices=A,
                                      depth=self._num_actions)
            # I have to do this because the shape of the output of action_taken is [?, 1, size_action_set],
            # but I need to be [?, size_action_set], otherwise the broadcasting of tf.multiply explodes everything.
            # squeezed_action_taken = tf.reshape(tensor=action_taken,
            #                                    shape=[-1, self._num_actions])
            action_embedding = tf.layers.dense(inputs=action_taken,
                                               units=self.params.num_nodes_fc_action_embedding,
                                               kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               activation=None)

            action_dot_phi = tf.multiply(x=action_embedding,
                                         y=fc2)

            fc_deconv1 = tf.layers.dense(inputs=action_dot_phi,
                                         units=self.params.num_nodes_fc1_out_reconstruction,
                                         kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                         bias_initializer=tf.constant_initializer(0.0),
                                         activation=None,)

            fc_deconv2 = tf.layers.dense(inputs=fc_deconv1,
                                         units=self.params.num_nodes_fc2_out_reconstruction,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         bias_initializer=tf.constant_initializer(0.0),
                                         activation=tf.nn.relu,)

            deconv1_in = tf.reshape(tensor=fc_deconv2,
                                    shape=[-1, self.params.shape_input_deconv1[0], self.params.shape_input_deconv1[1],
                                           self.params.num_filters_deconv1])

            deconv1 = tf.layers.conv2d_transpose(inputs=deconv1_in,
                                                 filters=self.params.num_filters_deconv1,
                                                 kernel_size=self.params.kernel_size_deconv1,
                                                 strides=self.params.stride_deconv1,
                                                 padding="VALID",
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0))

            crop_deconv1 = tf.slice(input_=deconv1,
                                    begin=[0, self.conf.padding_deconv1[0], self.conf.padding_deconv1[1], 0],
                                    size=[-1, self.params.shape_input_deconv2[0], self.params.shape_input_deconv2[1], 64],
                                    name=prefix + "_pad_deconv1")

            deconv2 = tf.layers.conv2d_transpose(inputs=crop_deconv1,
                                                 filters=self.params.num_filters_deconv2,
                                                 kernel_size=self.params.kernel_size_deconv2,
                                                 strides=self.params.stride_deconv2,
                                                 padding="VALID",
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0))

            crop_deconv2 = tf.slice(input_=deconv2,
                                    begin=[0, self.conf.padding_deconv2[0], self.conf.padding_deconv2[1], 0],
                                    size=[-1, self.params.shape_input_deconv3[0], self.params.shape_input_deconv3[1], 1],
                                    name=prefix + "_pad_deconv2")

            deconv3 = tf.layers.conv2d_transpose(inputs=crop_deconv2,
                                                 filters=self.params.num_filters_deconv3,
                                                 kernel_size=self.params.kernel_size_deconv3,
                                                 strides=self.params.stride_deconv3,
                                                 padding="VALID",
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.constant_initializer(0.0))

            crop_deconv3 = tf.slice(input_=deconv3,
                                    begin=[0, self.conf.padding_deconv3[0], self.conf.padding_deconv3[1], 0],
                                    size=[-1, self._input_height, self._input_width, 1],
                                    name=prefix + "_pad_deconv3")

            reconstructed_output = tf.contrib.layers.flatten(crop_deconv3)

        with tf.variable_scope(prefix + "_sr_estimator"):
            block_gradient = tf.stop_gradient(input=phi)

            fc_sr1 = tf.layers.dense(inputs=block_gradient,
                                     units=self.params.num_nodes_fc1_sr,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.0),
                                     activation=tf.nn.relu)

            sr_output = tf.layers.dense(inputs=fc_sr1,
                                        units=self.params.num_nodes_fc2_sr,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.constant_initializer(0.0),
                                        activation=tf.nn.relu)

        return value_function, reconstructed_output, sr_output, normalized_phi

    def build_nn(self):
        return self.build_q_network(self.X, self.actions, "online")

    def build_nn_prime(self):
        return self.build_q_network(self.X_t, self.actions, "prime")

    def build_nn_target(self):
        return self.build_q_network(self.X_t, self.actions, "target")

    def build_action(self):
        assertions = [tf.assert_equal(tf.shape(self.X)[0], tf.constant(1, dtype=tf.int32))]
        with tf.control_dependencies(assertions):
            Q = self.NN[0]
            deterministic_action = tf.argmax(Q, axis=1)
            return tf.squeeze(deterministic_action)  # Return action as a scalar

    def define_q_vals_loss(self, q_vals, q_target_vals):
        q_selected = tf.reduce_sum(q_vals * tf.one_hot(self.actions, self._num_actions), axis=1)  # Compute Q(s, a)
        self.train_summary.append(tf.summary.scalar('average_q', tf.reduce_mean(q_selected)))
        self.train_summary.append(tf.summary.scalar('max_q', tf.reduce_mean(tf.reduce_max(q_vals, axis=1))))

        # argmax_a Q'(s', a)
        q_prime = self.NN_prime[0]
        q_prime_actions = tf.argmax(q_prime, axis=1)
        # Compute Q(S', argmax_a Q'(s', a))
        q_target_rhs = tf.reduce_sum(q_target_vals * tf.one_hot(q_prime_actions, self._num_actions), axis=1)

        # Compute r + \gamma * max_a(Q(s', a)) or r + \gamma * Q(s', argmax_a Q'(s', a)) for DoubleQ
        # q_target_selected = tf.stop_gradient(self.rewards + FLAGS.gamma * (1.0 - self.terminals) * q_target_rhs)
        q_target_selected = tf.stop_gradient(
            self.rewards + FLAGS.gamma * (1.0 - self.terminals) * tf.reduce_max(q_target_vals, axis=1))

        # TD Errors: r + \gamma * max_a(Q(s', a)) - Q(s, a)
        td_errors = q_target_selected - q_selected
        mc_errors = self.mc_return - q_selected
        self.td_errors = tf.verify_tensor_all_finite(td_errors, "TD error tensor is NOT finite")
        self.train_summary.append(tf.summary.scalar('td_error', tf.reduce_mean(td_errors)))

        mmc_return_error = (1.0 - FLAGS.mc_return_beta) * td_errors + FLAGS.mc_return_beta * mc_errors

        # Compute huber loss of TD errors (clipping beyond [-1, 1])
        delta = 1.0
        errors = tf.where(
            tf.abs(mmc_return_error) < delta,
            tf.square(mmc_return_error) * 0.5,
            delta * (tf.abs(mmc_return_error) - 0.5 * delta))
        errors = tf.verify_tensor_all_finite(errors, "DQN training loss is NOT finite")

        loss = tf.reduce_mean(self.importance_weights * errors)

        self.train_summary.append(tf.summary.scalar('td_loss', loss))

        return loss

    def define_reconstruction_loss(self, reconstructed_vals):
        # The ground truth is the next set of observations, stored in X_t.
        # We compare it against the output of the reconstruction module
        ground_truth = tf.reshape(tensor=self.X_t[:, :, :, 3], shape=[-1, 84 * 84])
        squared_loss_reconstr = tf.reduce_mean(tf.squared_difference(reconstructed_vals, ground_truth))

        self.train_summary.append(tf.summary.scalar('recons_loss', squared_loss_reconstr))

        return squared_loss_reconstr

    def define_sr_loss(self, phi_target_vals, sr_vals, sr_target_vals):
        # We implement (target_phi + gamma * target_sr - online_sr)^2
        is_terminal = tf.tile(tf.expand_dims(self.terminals, 1), [1, self.params.num_nodes_fc2_sr])
        squared_loss_sr = tf.reduce_mean(tf.squared_difference(phi_target_vals, sr_vals -
                                                               FLAGS.gamma * (1.0 - is_terminal) * sr_target_vals))

        self.train_summary.append(tf.summary.scalar('sr_loss', squared_loss_sr))

        return squared_loss_sr

    def build_train(self):
        q_vals, reconstructed_vals, sr_vals, _ = self.NN
        q_target_vals, _, sr_target_vals, phi_target_vals = self.NN_target

        # Defining the loss for the reconstruction:
        recons_loss = self.define_reconstruction_loss(reconstructed_vals)

        # Define SR loss
        sr_loss = self.define_sr_loss(phi_target_vals, sr_vals, sr_target_vals)

        # Defining the loss for the Q-values:
        td_errors = self.define_q_vals_loss(q_vals, q_target_vals)

        # Define overall loss
        final_loss = FLAGS.wgt_td_loss * td_errors + FLAGS.wgt_sr_loss * sr_loss + FLAGS.wgt_recons_loss * recons_loss

        return tf.train.RMSPropOptimizer(
            FLAGS.learning_rate * self._learning_rate_scale,
            decay=FLAGS.rmsprop_decay,
            epsilon=FLAGS.rmsprop_epsilon,
            centered=True).minimize(final_loss,
                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="online"))

    def build_copy(self):
        """
        Builds copy operation to copy all trainable variables from
        our online network to our target network.
        """
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="online")
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")

        copy_ops = [target.assign(online) for target, online in zip(target_vars, online_vars)]

        return tf.group(*copy_ops)

    def build_process_frame(self):
        """
        Builds frame processing operation.

        Crops the image to (input_height, input_width) and returns image as uint8
        """
        cropped = tf.image.resize_images(self.image,
                                         (self._input_height, self._input_width), tf.image.ResizeMethod.AREA)
        return tf.cast(cropped, tf.uint8)
