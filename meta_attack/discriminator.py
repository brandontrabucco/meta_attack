import tensorflow as tf


class Discriminator(tf.keras.Model):

    def __init__(self, depth, output_size, *args, dropout_rate=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth

        self.conv_layers = [
            tf.keras.layers.Conv2D(
                2**(i + 3), (3, 3), strides=(2, 2), padding="same")
                for i in range(depth)]
        self.conv_batch_norm_layers = [
            tf.keras.layers.BatchNormalization()
            for i in range(depth)]
        self.conv_dropout_layers = [
            tf.keras.layers.SpatialDropout2D(dropout_rate)
            for i in range(depth)]

        self.flatten_layer = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(output_size)

    def call(self, image, training=True, **kwargs):
        x = image

        for i in range(self.depth):
            x = self.conv_layers[i](x)
            x = self.conv_batch_norm_layers[i](x, training=training)
            x = tf.nn.relu(x)
            x = self.conv_dropout_layers[i](x, training=training)

        x = self.flatten_layer(x)
        return self.output_layer(x)
