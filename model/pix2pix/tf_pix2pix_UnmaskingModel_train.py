import tensorflow as tf


def load_mask_pair_images(img_path):
    pass


def resize(input_image, target_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.imgae.ResizeMethod.NEAREST_NEIGHBOR
    )
    target_image = tf.image.resize(
        target_image, [height, width], method=tf.imgae.ResizeMethod.NEAREST_NEIGHBOR
    )

    return input_image, target_image


def random_crop(input_image, target_image, IMG_HEIGHT=256, IMG_WIDTH=256):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]


# normalize iamge to [-1, 1]
def normalize(input_image, target_image):
    input_image = (input_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1

    return input_image, target_image


@tf.function
def random_jitter(input_image, target_image):
    input_image, target_image = resize(input_image, target_image, 286, 286)
    input_image, target_image = random_crop(input_image, target_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    layers = tf.keras.Sequential()
    layers.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernal_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        layers.add(tf.keras.layers.BatchNormalization())

    layers.add(tf.keras.layers.LeakyReLU())

    return layers


def upsample(filters, size, apply_dropout=False, dropout_ratio=0.5):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    layers = tf.keras.Sequential()
    layers.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernal_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_dropout:
        layers.add(tf.keras.layers.Dropout(dropout_ratio))

    layers.add(tf.keras.layers.ReLU())

    return layers


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        3,
        4,
        stride=2,
        padding="same",
        kernal_initializer=initializer,
        activation="tanh",
    )

    x = inputs

    skips = []
    for down in down_stak:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.concatenate()([x, skip])

    x = last(x)

    return tf.keras.Mode(inputs=inputs, outputs=x)
