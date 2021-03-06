from tqdm.auto import tqdm

import tensorflow.keras.mixed_precision as mixed_precision
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


# normalize image to [-1, 1]
def normalize(img):
    img = (img / 127.5) - 1

    return img


# denormalize image to [0, 255]
def denormalize(img):
    img = (img + 1) * 127.5

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


class Pix2Pix(tf.keras.Model):
    def __init__(self, gen_loss_weight=100):
        super(Pix2Pix, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.gen_loss_weight = gen_loss_weight
        self.disc_loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.gen_loss_func = tf.keras.losses.MAE

    def call(self, mask_img, unmask_img=None):
        gen_img = self.generator(mask_img)

        if unmask_img is not None:
            disc_output_real = self.discriminator([mask_img, unmask_img])
            disc_output_fake = self.discriminator([mask_img, gen_img])

            """
            disc_real_loss = self.disc_loss_func(tf.ones_like(disc_output_real), disc_output_real)
            disc_fake_loss = self.disc_loss_func(tf.zeros_like(disc_output_fake), disc_output_fake)
            gen_loss = self.gen_loss_func(unmask_img, gen_img)
            loss = gen_loss + self.gen_loss_weight + disc_real_loss + disc_fake_loss
            self.add_loss(loss)
            """
            return gen_img, disc_output_real, disc_output_fake

        return denormalize(gen_img)


@tf.function
def train_step(model, mask_imgs, unmask_imgs, optimizer):
    with tf.GradientTape() as tape:
        gen_img, disc_real_output, disc_fake_output = model(mask_imgs, unmask_imgs)

        gen_loss = tf.reduce_mean(model.gen_loss_func(unmask_imgs, gen_img))
        disc_real_loss = model.disc_loss_func(
            tf.ones_like(disc_real_output), disc_real_output
        )
        disc_fake_loss = model.disc_loss_func(
            tf.zeros_like(disc_fake_output), disc_fake_output
        )

        loss = gen_loss * model.gen_loss_weight + disc_real_loss + disc_fake_loss
        scaled_loss = optimizer.get_scaled_loss(loss)

    scaled_grads = tape.gradient(loss, model.trainable_weights)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    tf.summary.scalar("train/gen_loss", gen_loss, step=optimizer.iterations)
    tf.summary.scalar("train/disc_real_loss", disc_real_loss, step=optimizer.iterations)
    tf.summary.scalar("train/disc_fake_loss", disc_fake_loss, step=optimizer.iterations)
    tf.summary.scalar("loss", loss, step=optimizer.iterations)

    if optimizer.iterations != 0 and optimizer.iterations % 1000 == 0:
        tf.summary.image(f"train/gen-unmask-img", gen_img, step=optimizer.iterations)

    return loss


@tf.function
def valid_step(model, mask_imgs, unmask_imgs, optimizer):
    gen_img, disc_real_output, disc_fake_output = model(mask_imgs, unmask_imgs)

    gen_loss = tf.reduce_mean(model.gen_loss_func(unmask_imgs, gen_img))
    disc_real_loss = model.disc_loss_func(
        tf.ones_like(disc_real_output), disc_real_output
    )
    disc_fake_loss = model.disc_loss_func(
        tf.zeros_like(disc_fake_output), disc_fake_output
    )
    loss = gen_loss * model.gen_loss_weight + disc_real_loss + disc_fake_loss

    tf.summary.scalar("valid/gen_loss", gen_loss, step=optimizer.iterations)
    tf.summary.scalar("valid/disc_real_loss", disc_real_loss, step=optimizer.iterations)
    tf.summary.scalar("valid/disc_fake_loss", disc_fake_loss, step=optimizer.iterations)
    tf.summary.scalar("val_loss", loss, step=optimizer.iterations)


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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--ckpt_name", type=str, default="tf_pix2pix_UnmaskingModel")
    parser.add_argument("--ckpt_bucket_name", type=str)
    args = parser.parse_args()

    summary_writer = tf.summary.create_file_writer(args.logdir)
    summary_writer.set_as_default()

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

    full_dataset = tf.data.Dataset.zip((mask_dataset, unmask_dataset)).shuffle(
        args.shuffle_size
    )

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

    model = Pix2Pix()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)

    # Iterate over epochs.
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for epoch in tqdm(range(args.epochs), desc="epochs"):
        print("Start of epoch {}".format(epoch))

        # Iterate over the batches of the train dataset.
        for step, (mask_imgs, unmask_imgs) in tqdm(
            enumerate(train_dataset),
            desc="train_steps",
            total=int(args.train_ratio * dataset_length) // args.batch_size,
        ):
            loss = train_step(model, mask_imgs, unmask_imgs, optimizer)

            ckpt.step.assign_add(1)
            if int(ckpt.step) % 1000 == 0 and ckpt.step > 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("loss {:1.2f}".format(loss.numpy()))
                model.generator.save(f"{args.ckpt_name}_generator_savedModel")

        # Iterate over the batches of the valid dataset.
        gen_loss, disc_real_loss, disc_fake_loss, loss = None, None, None, None
        for step, (mask_imgs, unmask_imgs) in tqdm(
            enumerate(val_dataset),
            desc="valid_steps",
            total=(dataset_length - int(args.train_ratio * dataset_length))
            // args.batch_size,
        ):
            valid_step(model, mask_imgs, unmask_imgs, optimizer)
