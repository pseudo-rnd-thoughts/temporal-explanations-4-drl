"""
Code for the agent autoencoder and related loss functions
for encoding the agent's observation into a smaller latent space
"""
import gymnasium as gym
import tensorflow as tf

from temporal_explanations_4_xrl.agent_networks import (
    flax_to_tf_weights,
    load_dopamine_dqn_flax_model,
)


class AtariAutoencoder(tf.keras.Model):
    """Atari autoencoder"""

    def __init__(self, latent_dims: int = 16, output_dims: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dims = latent_dims
        self.output_dims = output_dims

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=8,
                    strides=4,
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=4,
                    strides=2,
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(
                    latent_dims
                ),  # Embedding layer with linear activation
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(9 * 9 * 64),
                tf.keras.layers.Reshape((9, 9, 64)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, activation="relu"
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=4, strides=2, activation="relu"
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=4, kernel_size=6, strides=2, activation="sigmoid"
                ),
            ]
        )

    def call(self, x):
        """Encodes and then decodes the observation"""
        return self.decoder(self.encoder(tf.cast(x, tf.float32)))

    def encode(self, x):
        """Encodes the observation"""
        return self.encoder(tf.cast(x, tf.float32))


def policy_reconstruction_error(
    env_name: str, network, model_root_folder: str = "../models/dopamine/jax"
):
    """A custom policy reconstruction loss using the MSE of the agent policy with the true and predicted observation"""
    env = gym.make(f"ALE/{env_name}-v5")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    model_def, model_params = load_dopamine_dqn_flax_model(env_name, model_root_folder)
    tf_policy = network(env.action_space.n)
    flax_to_tf_weights(model_params, tf_policy)

    tf_policy.trainable = False

    def policy_error(y_true, y_pred):
        """Error function for the policy reconstruction."""
        true_policy = tf_policy(y_true)  # q-value for y_true
        pred_policy = tf_policy(y_pred)  # q-value for y_pred
        return tf.reduce_mean(tf.square(true_policy - pred_policy))

        # return tf.reduce_mean(tf.square(y_true - y_pred))

    return policy_error


class Reparameterize(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class AtariVariationalAutoencoder(tf.keras.Model):
    def __init__(self, latent_dims: int, output_dims: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dims = latent_dims
        self.output_dims = output_dims

        encoder_inputs = tf.keras.Input(shape=(84, 84, output_dims))
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation="relu",
            padding="same",
        )(encoder_inputs)
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation="relu",
            padding="same",
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation="relu",
            padding="same",
        )(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        z_mean = tf.keras.layers.Dense(latent_dims)(x)
        z_log_var = tf.keras.layers.Dense(latent_dims, activation="relu")(x)
        z = Reparameterize()([z_mean, z_log_var])
        self.encoder = tf.keras.Model(
            encoder_inputs, [z_mean, z_log_var, z], name="encoder"
        )

        decoder_inputs = tf.keras.Input(shape=(latent_dims,))
        x = tf.keras.layers.Dense(9 * 9 * 64)(decoder_inputs)
        x = tf.keras.layers.Reshape((9, 9, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, activation="relu"
        )(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=4, strides=2, activation="relu"
        )(x)
        decoder_output = tf.keras.layers.Conv2DTranspose(
            filters=output_dims, kernel_size=6, strides=2, activation="sigmoid"
        )(x)
        self.decoder = tf.keras.Model(decoder_inputs, decoder_output, name="decoder")

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mean_squared_error(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mean_squared_error(data, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def encode(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        return z_mean

    def reconstruct(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        return self.decoder(z_mean)
