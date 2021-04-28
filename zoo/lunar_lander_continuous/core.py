import tensorflow as tf


class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_shape, output_dim, env_action_max, env_action_min):
        super().__init__()
        self.env_action_min = env_action_min
        self.env_action_max = env_action_max
        self.action_min, self.action_max = -1.0, 1.0

        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu', input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=output_dim, activation='tanh')

        self.call(tf.ones((1, *input_shape)))

    def get_config(self):
        super().get_config()

    def call(self, observations, **kwargs):
        x = self.dense1(observations)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def sample(self, observations, noise=None, noise_clip=None):
        actions = self.call(observations)
        actions = (actions - self.action_min) / (self.action_max - self.action_min)
        actions = actions * (self.env_action_max - self.env_action_min) + self.env_action_min
        if noise:
            epsilon = tf.random.normal(actions.shape, mean=0.0, stddev=noise)
            if noise_clip:
                epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            actions += epsilon
        actions = tf.clip_by_value(actions, self.env_action_min, self.env_action_max)
        return actions


class QFunctionNetwork(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu', input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=1, activation='linear')

        self.call(tf.ones((1, *input_shape)))

    def get_config(self):
        super().get_config()

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def compute(self, observations, actions):
        x = tf.concat([observations, actions], axis=-1)
        x = self.call(x)
        return tf.squeeze(x)
