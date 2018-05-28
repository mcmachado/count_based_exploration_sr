import tensorflow as tf

from common.decorators import operation

flags = tf.app.flags
FLAGS = flags.FLAGS

# Optimizer params (RMSProp)
flags.DEFINE_float('learning_rate', 0.00025, 'learning rate')
flags.DEFINE_float('rmsprop_decay', 0.95, 'rmsprop gradient momentum')
flags.DEFINE_float('rmsprop_epsilon', 0.000009765625, 'rmsprop denominator epsilon')

# NN Params
flags.DEFINE_boolean('pad_first_conv_layer', False, 'add padding to first conv layer as per DeepMind')

# RL Params
flags.DEFINE_float('gamma', 0.99, 'discount rate')

# MC-Return Params
flags.DEFINE_float('mc_return_beta', 0.1, 'Trade-off between the TD-error (1 - beta) and the MC return (beta)')


class ExploreEigenvectorSR:
    def __init__(self, input_height, input_width, num_actions, frame_history_size, learning_rate_scale=1.):
        """
        input_height: Downsized input height
        input_width: Downsized input width
        num_actions: Minimal action set length
        learning_rate_scale: Used to scale the learning rate, what Schaul et al. (2016) call \eta.
                             This is utilized in prioritized experience replay.
        """
        self._input_height = input_height
        self._input_width = input_width
        self._num_actions = num_actions
        self._learning_rate_scale = learning_rate_scale

        self.train_summary = []

        self.X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, frame_history_size],
                                name='eesr/inputs')
        self.actions = tf.placeholder(tf.uint8, shape=[None], name='exp_eig_sr/actions')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name='exp_eig_sr/rewards')
        self.mc_return = tf.placeholder(tf.float32, shape=[None], name='exp_eig_sr/mc_return')
        self.X_t = tf.placeholder(tf.float32, shape=[None, input_height, input_width, frame_history_size],
                                  name='exp_eig_sr/target_inputs')
        self.terminals = tf.placeholder(tf.float32, shape=[None], name='exp_eig_sr/terminals')

        # Importance weights to scale loss by, used in prioritized experience replay
        self.importance_weights = tf.placeholder(tf.float32, [None], name='exp_eig_sr/importance_weights')

        self.image = tf.placeholder(tf.uint8, shape=[None, None, 1], name='exp_eig_sr/raw_image')

        self.Q, self.Q_prime, self.Q_target, self.train, self.action, self.process_frame, self.copy_to_target

        self.train_summary = tf.summary.merge(self.train_summary)
        tf.logging.info("Built DQN graph")

    def build_q_network(self, X, num_actions):
        """
        Builds architecture for our Q-network.

        Notes: Same arch as DQN Nature paper.

        Parameters:
          num_actions: number of actions possible. Usually used as size of output
          scope: Scope for operations. Please wrap all operations inside variable_scope. See DQN.build_copy

        Returns: Final output layer which can be fed X as input
        """
        conv_input = tf.divide(X, 255.)  # Normalize inputs
        # DeepMind pads the first convolutional layer with zeros on width and height
        # Making the image now (BATCH_SIZE, 86, 86, FRAME_HISTORY_LEN)
        if FLAGS.pad_first_conv_layer:
            conv_input = tf.pad(conv_input, [[0, 0], [1, 1], [1, 1], [0, 0]])

        initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG', uniform=True)
        with tf.variable_scope("conv"):
            conv1 = tf.layers.conv2d(conv_input,
                                     filters=32,
                                     kernel_size=8,
                                     kernel_initializer=initializer,
                                     bias_initializer=initializer,
                                     strides=4,
                                     activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv1,
                                     filters=64,
                                     kernel_size=4,
                                     kernel_initializer=initializer,
                                     bias_initializer=initializer,
                                     strides=2,
                                     activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(conv2,
                                     filters=64,
                                     kernel_size=3,
                                     kernel_initializer=initializer,
                                     bias_initializer=initializer,
                                     strides=1,
                                     activation=tf.nn.relu)
            conv_output = tf.contrib.layers.flatten(conv3)
        with tf.variable_scope("action_values"):
            fully_connected = tf.layers.dense(conv_output,
                                              units=512,
                                              activation=tf.nn.relu,
                                              kernel_initializer=initializer,
                                              bias_initializer=initializer)
            actions = tf.layers.dense(fully_connected,
                                      units=num_actions,
                                      activation=None,
                                      kernel_initializer=initializer,
                                      bias_initializer=initializer)
        return actions

    @operation('q')
    def Q(self):
        return self.build_q_network(self.X, self._num_actions)

    @operation('q', reuse=True)
    def Q_prime(self):
        return self.build_q_network(self.X_t, self._num_actions)

    @operation('q_target')
    def Q_target(self):
        return self.build_q_network(self.X_t, self._num_actions)

    @operation('exp_eig_sr/greedy_action')
    def action(self):
        """
        TODO: Update
        """
        assertions = [
            tf.assert_equal(tf.shape(self.X)[0], tf.constant(1, dtype=tf.int32))]
        with tf.control_dependencies(assertions):
            deterministic_action = tf.argmax(self.Q, axis=1)
            return tf.squeeze(deterministic_action)  # Return action as a scalar

    @operation('exp_eig_sr/train')
    def train(self):
        """
        Builds training operation.
        Returns:
          Training operation which should be fed:
            self.X - Input image
            self.X_t - Target input image
            self.actions - Actions taken in batch
            self.rewards - Rewards received in batch
            self.terminals - Whether state was terminal or not
        """
        q_vals = self.Q
        q_target_vals = self.Q_target

        # Compute Q(s, a)
        q_selected = tf.reduce_sum(q_vals * tf.one_hot(self.actions, self._num_actions), axis=1)
        self.train_summary.append(tf.summary.scalar('average_q', tf.reduce_mean(q_selected)))
        self.train_summary.append(tf.summary.scalar('max_q', tf.reduce_mean(tf.reduce_max(q_vals, axis=1))))

        # argmax_a Q'(s', a)
        q_prime_actions = tf.argmax(self.Q_prime, axis=1)
        # Compute Q(S', argmax_a Q'(s', a))
        q_target_rhs = tf.reduce_sum(q_target_vals * tf.one_hot(q_prime_actions, self._num_actions), axis=1)

        # Compute r + \gamma * max_a(Q(s', a)) or r + \gamma * Q(s', argmax_a Q'(s', a)) for DoubleQ
        # Q(s', a) = 0 for terminal s
        q_target_selected = tf.stop_gradient(self.rewards + FLAGS.gamma * (1.0 - self.terminals) * q_target_rhs)

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
        self.train_summary.append(tf.summary.scalar('total_loss', loss))

        return tf.train.RMSPropOptimizer(
            FLAGS.learning_rate * self._learning_rate_scale,
            decay=FLAGS.rmsprop_decay,
            epsilon=FLAGS.rmsprop_epsilon,
            centered=True).minimize(loss,
                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Q.scope))

    @operation
    def copy_to_target(self):
        """
        Builds copy operation to copy all trainable variables from
        our online network to our target network.
        """
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Q.scope)
        target_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.Q_target.scope)

        copy_ops = [target_var.assign(q_var) for target_var, q_var in zip(target_q_vars, q_vars)]

        return tf.group(*copy_ops)

    @operation
    def process_frame(self):
        """
        Builds frame processing operation.

        Crops the image to (input_height, input_width) and returns image as uint8
        """
        cropped = tf.image.resize_images(self.image, (self._input_height, self._input_width), tf.image.ResizeMethod.AREA)
        return tf.cast(cropped, tf.uint8)
