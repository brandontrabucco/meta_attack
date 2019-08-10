import tensorflow as tf


class Generator(tf.keras.Model):

    def __init__(self, depth, *args, dropout_rate=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth

        self.conv_layers = [
            tf.keras.layers.Conv2D(
                2**(i + 3), 3, strides=2, padding="same")
                for i in range(depth)]
        self.conv_batch_norm_layers = [
            tf.keras.layers.BatchNormalization()
            for i in range(depth)]
        self.conv_dropout_layers = [
            tf.keras.layers.SpatialDropout2D(dropout_rate)
            for i in range(depth)]

        self.deconv_layers = [
            tf.keras.layers.Conv2DTranspose(
                (2**(i + 2) if i > 0 else 3), 3, strides=2, padding="same")
            for i in reversed(range(depth))]
        self.deconv_batch_norm_layers = [
            tf.keras.layers.BatchNormalization()
            for i in range(depth)]
        self.deconv_dropout_layers = [
            tf.keras.layers.SpatialDropout2D(dropout_rate)
            for i in range(depth)]

    def call(self, image, training=True, **kwargs):
        x = image

        for i in range(self.depth):
            x = self.conv_layers[i](x)
            x = self.conv_batch_norm_layers[i](x, training=training)
            x = tf.nn.relu(x)
            x = self.conv_dropout_layers[i](x, training=training)

        for i in range(self.depth):
            x = self.deconv_layers[i](x)
            x = self.deconv_batch_norm_layers[i](x, training=training)
            x = tf.nn.relu(x)
            x = self.deconv_dropout_layers[i](x, training=training)

        x = tf.nn.tanh(x)
        return tf.clip_by_value(image + x, -1.0, 1.0)
