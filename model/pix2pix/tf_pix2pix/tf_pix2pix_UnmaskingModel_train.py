# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name="cpu")  # mac M1 optimized
# mlcompute.set_mlc_device(device_name="gpu")

import tensorflow as tf

import os, sys
import argparse
import time
import datetime


def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img)
    img = tf.cast(img, tf.float32)

    return img


def load_image_train(img):
    img = random_jitter(img)
    img = normalize(img)

    return img


def load_image_test(img, IMG_HEIGHT=256, IMG_WIDTH=256):
    img = resize(img, IMG_HEIGHT, IMG_WIDTH)
    img = normalize(img)

    return img


def resize(img, height, width):
    img = tf.image.resize(
        img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return img


def random_crop(img, IMG_HEIGHT=256, IMG_WIDTH=256):
    img = tf.image.random_crop(img, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return img


# normalize iamge to [-1, 1]
def normalize(img):
    img = (img / 127.5) - 1

    return img


@tf.function
def random_jitter(img):
    img = resize(img, 286, 286)
    img = random_crop(img, 256, 256)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        img = tf.image.flip_left_right(img)

    return img


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    layers = tf.keras.Sequential()
    layers.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
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
            kernel_initializer=initializer,
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
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )

    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        # x = tf.keras.concatenate()([x, skip])
        x += skip

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="generator")


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    targets = tf.keras.layers.Input(shape=[256, 256, 3])

    x = tf.keras.layers.concatenate([inputs, targets])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inputs, targets], outputs=last, name="discriminator")


def create_pix2pix_loss_func(
    unmask_inputs, gen_output, disc_output_real, disc_output_fake, gen_loss_weight=100
):
    # create loss for discriminator
    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    disc_loss1 = loss_func(tf.ones_like(disc_output_real), disc_output_real)
    disc_loss2 = loss_func(tf.zeros_like(disc_output_fake), disc_output_fake)

    # create loss for generator
    gen_loss = tf.keras.losses.MAE(unmask_inputs, gen_output)

    return {
        "loss": (disc_loss1 + disc_loss2) + (gen_loss_weight * gen_loss), # loss value weighting
        #"disc_real_loss": disc_loss1,
        #"disc_fake_loss": disc_loss2,
        #"gen_loss" : gen_loss_weight * gen_loss
    }


def create_pix2pix_model(lr=1e-4, beta_1=0.5, beta_2=0.999):
    generator = Generator()
    discriminator = Discriminator()

    mask_inputs = tf.keras.layers.Input(shape=[256, 256, 3], name="mask_inputs")
    unmask_inputs = tf.keras.layers.Input(shape=[256, 256, 3], name="unmask_inputs")

    gen_output = generator(mask_inputs)
    disc_output_real = discriminator([mask_inputs, unmask_inputs])
    disc_output_fake = discriminator([mask_inputs, gen_output])

    model = tf.keras.Model(
        inputs=[mask_inputs, unmask_inputs],
        outputs=[gen_output, unmask_inputs, disc_output_real, disc_output_fake],
    )
    model.add_loss(
        create_pix2pix_loss_func(
            unmask_inputs, gen_output, disc_output_real, disc_output_fake
        )
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2),
        run_eagerly=False,
    )  # , loss_weights=[0.25,0.25,0.5])
    model.summary()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unmask_img_folder",
        type=str,
        default="../../../data/celeba-mask-pair/unmask_images/raw",
    )
    parser.add_argument(
        "--mask_img_folder",
        type=str,
        default="../../../data/celeba-mask-pair/mask_images/raw",
    )
    parser.add_argument("--shuffle_size", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--logdir", type=str, default="./logs")
    args = parser.parse_args()

    file_writer = tf.summary.create_file_writer(args.logdir)
    file_writer.set_as_default()

    unmask_dataset = tf.data.Dataset.list_files(
        args.unmask_img_folder + "/*.png", shuffle=False
    )
    mask_dataset = tf.data.Dataset.list_files(
        args.mask_img_folder + "/*.png", shuffle=False
    )
    dataset_length = len(mask_dataset)

    unmask_dataset = unmask_dataset.map(load_image).map(
        load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    mask_dataset = mask_dataset.map(load_image).map(
        load_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    full_dataset = tf.data.Dataset.zip((mask_dataset, unmask_dataset))
    full_dataset = tf.data.Dataset.zip((full_dataset, unmask_dataset))

    train_dataset = (
        full_dataset.take(int(args.train_ratio * dataset_length))
        .batch(args.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    val_dataset = (
        full_dataset.skip(int(args.train_ratio * dataset_length))
        .batch(args.batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # register callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir)

    model = create_pix2pix_model(lr=args.lr)
    model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=int(dataset_length / args.batch_size),
        validation_data=val_dataset,
        validation_steps=int(dataset_length / args.batch_size),
        callbacks=[tensorboard_callback],
    )
