import meta_attack.config as config
import tensorflow as tf


from meta_attack.generator import Generator
from meta_attack.discriminator import Discriminator


if __name__ == "__main__":

    generator = Generator(config.network_depth)
    discriminator = Discriminator(config.network_depth, config.label_size)
    optimizer = tf.keras.optimizers.Adam()

    images = tf.random.normal([1, config.image_size, config.image_size, 3])
    labels = tf.random.uniform([1], maxval=config.label_size, dtype=tf.int32)

    for o in range(config.outer_steps):
        with tf.GradientTape(persistent=True) as tape:
            for i in range(config.inner_steps):

                real_images = images
                fake_images = generator(images, training=True)

                real_logits = discriminator(images)
                fake_logits = discriminator(fake_images)

                real_loss = tf.reduce_mean(
                    tf.losses.sparse_categorical_crossentropy(labels, real_logits))
                fake_loss = tf.reduce_mean(
                    tf.losses.sparse_categorical_crossentropy(labels, fake_logits))

                inner_loss = (config.real_loss_weight * real_loss +
                              config.fake_loss_weight * fake_loss)
                gradients = tape.gradient(inner_loss, discriminator.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, discriminator.trainable_variables))

                print("[{:09d}] Inner loss: {:.5f}\tReal loss: {:.5f}\tFake loss: {:.5f}".format(
                    o * config.inner_steps + i,
                    inner_loss.numpy(),
                    real_loss.numpy(),
                    fake_loss.numpy()))

            content_loss = tf.reduce_mean(tf.abs(real_images - fake_images))
            outer_loss = (config.content_loss_weight * content_loss -
                          config.adverse_loss_weight * inner_loss)

        gradients = tape.gradient(outer_loss, generator.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, generator.trainable_variables))

        print("[{:09d}] Outer loss: {:.5f}\tAdverse loss: {:.5f}\tContent loss: {:.5f}".format(
            o,
            outer_loss.numpy(),
            -inner_loss.numpy(),
            content_loss.numpy()))
