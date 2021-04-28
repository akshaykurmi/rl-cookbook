import tensorflow as tf
import tensorflow_probability as tfp


class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=20, activation='relu', input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=output_dim, activation='linear')

    def get_config(self):
        super().get_config()

    def call(self, observations, **kwargs):
        x = self.dense1(observations)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def distribution(self, observations):
        logits = self.call(observations)
        return tfp.distributions.Categorical(logits=logits)

    def sample(self, observations):
        return self.distribution(observations).sample()

    def log_prob(self, observations, actions):
        return self.distribution(observations).log_prob(actions)


class ValueFunctionNetwork(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=20, activation='relu', input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=1, activation='linear')

    def get_config(self):
        super().get_config()

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def compute(self, observations):
        return self.call(observations)
