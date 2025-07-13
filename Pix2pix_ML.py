import tensorflow as tf
import os
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

class Pix2PixModel:
    def __init__(self, img_height=256, img_width=256, learning_rate=2e-4):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = 3
        self.lambda_l1 = 100
        self.learning_rate = learning_rate

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate * 0.5, beta_1=0.5)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')

        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer
        )
        self.checkpoint_manager = None

    def build_generator(self):
        inputs = layers.Input(shape=[self.img_height, self.img_width, self.channels])
        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),
            self.downsample(128, 4),
            self.downsample(256, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),
        ]
        up_stack = [
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4),
            self.upsample(256, 4),
            self.upsample(128, 4),
            self.upsample(64, 4),
        ]
        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(self.channels, 4, strides=2, padding='same',
                                      kernel_initializer=initializer, activation='tanh')

        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = layers.Input(shape=[self.img_height, self.img_width, self.channels])
        tar = layers.Input(shape=[self.img_height, self.img_width, self.channels])
        x = layers.Concatenate()([inp, tar])
        down1 = self.downsample(64, 4, False)(x)
        down2 = self.downsample(128, 4)(down1)
        down3 = self.downsample(256, 4)(down2)
        zero_pad1 = layers.ZeroPadding2D()(down3)
        conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
        batchnorm1 = layers.BatchNormalization()(conv)
        leaky_relu = layers.LeakyReLU()(batchnorm1)
        leaky_relu = layers.Dropout(0.3)(leaky_relu)
        zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
        last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(layers.BatchNormalization())
        result.add(layers.LeakyReLU())
        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                          kernel_initializer=initializer, use_bias=False))
        result.add(layers.BatchNormalization())
        if apply_dropout:
            result.add(layers.Dropout(0.5))
        result.add(layers.ReLU())
        return result

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output) * 0.9, disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        return real_loss + generated_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.lambda_l1 * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)
            gen_total_loss, _, _ = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        self.gen_loss_metric.update_state(gen_total_loss)
        self.disc_loss_metric.update_state(disc_loss)

        return {
            "gen_loss": gen_total_loss,
            "disc_loss": disc_loss
        }

    def setup_checkpoint_manager(self, checkpoint_dir):
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=checkpoint_dir, max_to_keep=5
        )
        latest_ckpt = self.checkpoint_manager.latest_checkpoint
        if latest_ckpt:
            self.checkpoint.restore(latest_ckpt)
            print(f" Checkpoint restored from {latest_ckpt}")
            return True
        print(" No checkpoint found. Training from scratch.")
        return False

class DatasetGenerator:
    def __init__(self, img_height=256, img_width=256):
        self.img_height = img_height
        self.img_width = img_width

    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.img_height, self.img_width])
        image = tf.cast(image, tf.float32)
        return (image / 127.5) - 1.0

    def create_paired_dataset(self, input_dir, target_dir, test_size=0.2, batch_size=16):
        input_images = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png")))
        target_images = sorted(glob.glob(os.path.join(target_dir, "*.jpg")) + glob.glob(os.path.join(target_dir, "*.png")))
        paired_paths = []
        for input_path in input_images:
            base_name = os.path.basename(input_path)
            target_path = os.path.join(target_dir, base_name)
            if target_path in target_images:
                paired_paths.append((input_path, target_path))

        train_pairs, test_pairs = train_test_split(paired_paths, test_size=test_size, random_state=42)

        def load_and_map(input_path, target_path):
            return self.load_image(input_path), self.load_image(target_path)

        train_ds = tf.data.Dataset.from_tensor_slices(([pair[0] for pair in train_pairs], [pair[1] for pair in train_pairs]))
        test_ds = tf.data.Dataset.from_tensor_slices(([pair[0] for pair in test_pairs], [pair[1] for pair in test_pairs]))

        train_ds = train_ds.map(load_and_map, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(load_and_map, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, test_ds

def train_and_evaluate():
    model = Pix2PixModel(learning_rate=2e-4)
    dataset_gen = DatasetGenerator()

    input_dir = r"C:\\Users\\moham\\OneDrive\\Documents\\CODE\\ImageProcessing\\DatasetRollFilm\\Negative"
    target_dir = r"C:\\Users\\moham\\OneDrive\\Documents\\CODE\\ImageProcessing\\DatasetRollFilm\\RGB"

    train_ds, test_ds = dataset_gen.create_paired_dataset(input_dir, target_dir, test_size=0.2, batch_size=16)

    checkpoint_dir = "pix2pix_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_loaded = model.setup_checkpoint_manager(checkpoint_dir)
    start_epoch = 0

    if checkpoint_loaded:
        print(f" Melanjutkan training dari checkpoint.")
    else:
        print(" Tidak ada checkpoint ditemukan. Training dari awal...")

    epochs = 50
    checkpoint_interval = 10

    print(" Memulai training...\n")

    for epoch in range(start_epoch, epochs):
        model.gen_loss_metric.reset_state()
        model.disc_loss_metric.reset_state()

        progress_bar = tqdm(train_ds, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')

        for input_image, target in progress_bar:
            losses = model.train_step(input_image, target)
            progress_bar.set_postfix({
                'Gen Loss': f"{losses['gen_loss']:.4f}",
                'Disc Loss': f"{losses['disc_loss']:.4f}"
            })

        print(f"Epoch {epoch + 1}/{epochs} Summary:")
        print(f"Generator Loss     : {model.gen_loss_metric.result():.4f}")
        print(f"Discriminator Loss : {model.disc_loss_metric.result():.4f}")

        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == epochs:
            model.checkpoint_manager.save()
            print(f" Checkpoint disimpan di: {checkpoint_dir}")

            h5_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch + 1}.h5")
            model.generator.save(h5_path)
            print(f" Generator disimpan sebagai H5: {h5_path}")

    print("Training selesai!")

if __name__ == "__main__":
    train_and_evaluate()
