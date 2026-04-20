# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# In this notebook, we are building Generative Adversarial Network to restore clean images from noisy images as present in the dataset "humanface8000". The Generator network has a UNet Architecture.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Import necessary libraries

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
import numpy as np
import matplotlib.pyplot as plt
import PIL
import time

import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
from skimage.metrics import mean_squared_error, normalized_root_mse, structural_similarity

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. View a sample image from the dataset

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
image_color = PIL.Image.open("humanface8000/landscape Images/color/10006.jpg")
image_gray = PIL.Image.open("humanface8000/landscape Images/gray/10006.jpg")

print(np.array(image_color).shape, np.array(image_gray).shape)
print(np.max(image_color), np.min(image_color), np.max(image_gray), np.min(image_gray))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image_color)

plt.subplot(1,2,2)
plt.imshow(image_gray)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Load the dataset
#
# The dataset is load using the function "image_dataset_from_directory" in batches of size 64. The dataset is splitted into 90-10 train-test subsets.
#
# The images are then scaled to the range (0-1) for efficient computation.

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
color_dir = 'humanface8000/landscape Images/color/'
gray_dir = 'humanface8000/landscape Images/gray/'

img_height, img_width = 256, 256
batch_size = 32

color_train_dataset = tf.keras.utils.image_dataset_from_directory(
    color_dir, labels=None,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

color_test_dataset = tf.keras.utils.image_dataset_from_directory(
    color_dir, labels=None,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

gray_train_dataset = tf.keras.utils.image_dataset_from_directory(
    gray_dir, labels=None,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

gray_test_dataset = tf.keras.utils.image_dataset_from_directory(
    gray_dir, labels=None,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
color_train_dataset = color_train_dataset.map(lambda x: normalization_layer(x))
color_test_dataset = color_test_dataset.map(lambda x: normalization_layer(x))
gray_train_dataset = gray_train_dataset.map(lambda x: normalization_layer(x))
gray_test_dataset = gray_test_dataset.map(lambda x: normalization_layer(x))


train_dataset = tf.data.Dataset.zip((color_train_dataset, gray_train_dataset))
test_dataset = tf.data.Dataset.zip((color_test_dataset, gray_test_dataset))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Visualize images from the training and testing dataset

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
for (color_images, blur_images) in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(color_images[i])
        plt.axis('off')
        plt.title('Colorful Image')

        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(blur_images[i])
        plt.axis('off')
        plt.title('Blurred Image')
    plt.show()
    print(color_images.shape, blur_images.shape)

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
for (color_images, blur_images) in test_dataset.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(color_images[i])
        plt.axis('off')
        plt.title('Colorful Image')

        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(blur_images[i])
        plt.axis('off')
        plt.title('Blurred Image')
    plt.show()
    print(color_images.shape, blur_images.shape)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. GAN Model
#
# ### 5.1 UNet Architecture

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer='he_normal', use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                    padding='same',kernel_initializer='he_normal',use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### 5.2 Generator Model

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
        downsample(64, 3, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 3), # (bs, 64, 64, 128)
        downsample(128, 3), # (bs, 32, 32, 128)
        downsample(256, 3), # (bs, 16, 16, 256)
        downsample(256, 3), # (bs, 8, 8, 256)
        downsample(512, 3), # (bs, 4, 4, 512)
        downsample(512, 3), # (bs, 2, 2, 512)
        downsample(512, 3), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 3, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 3, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(256, 3, apply_dropout=True), # (bs, 8, 8, 512)
        upsample(256, 3), # (bs, 16, 16, 512)
        upsample(128, 3), # (bs, 32, 32, 256)
        upsample(128, 3), # (bs, 64, 64, 256)
        upsample(64, 3), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 5, strides=2,
                    padding='same',kernel_initializer=initializer,activation='relu') # (bs, 256, 256, 3)

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

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
generator = Generator()
generator.summary()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### 5.3 Discriminator Model

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')
    tar = layers.Input(shape=[256, 256, 3], name='target_image')

    x = layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 3, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 3)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 3)(down2) # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 3, strides=1,
                kernel_initializer=initializer, use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    last = layers.Conv2D(1, 3, strides=1,kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
    flat = layers.Flatten()(last)
    out = layers.Dense(1)(flat)
    return tf.keras.Model(inputs=[inp, tar], outputs=out)

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
discriminator = Discriminator()
discriminator.summary()

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
tf.keras.utils.plot_model(generator, to_file= "generator.png", show_shapes=True)
tf.keras.utils.plot_model(discriminator, to_file= "discriminator.png", show_shapes=True)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### 5.4 Loss Functions and Optimizers

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def generator_loss(disc_generated_output, gen_output, target):

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = l1_loss

    return total_gen_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Training

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
    return gen_total_loss, disc_loss

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# A set of 5 images are taken from the test subset and predictions are made on those 5 images which is viewed after every 10 epochs.

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
def generate_samples(epoch, num=5):
    prediction = generator(blur_images, training = False)
    plt.figure(figsize=(10, num*2))
    for i in range(num):
        plt.subplot(5, 3, 3*i + 1)
        plt.imshow(blur_images[i])
        plt.axis('off')
        plt.title('Noisy Image')

        plt.subplot(5, 3, 3*i + 2)
        plt.imshow(color_images[i])
        plt.axis('off')
        plt.title('Ground Truth')

        plt.subplot(5, 3, 3*i + 3)
        plt.imshow(prediction[i])
        plt.axis('off')
        plt.title('Predicted Image')
    plt.savefig("Results after {} th epoch".format(epoch))
    plt.tight_layout()
    plt.show()

for (color_images, blur_images) in test_dataset.take(1):
    color_images = color_images
    blur_images = blur_images

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
def fit(train_ds, epochs):
    total_start = time.time()
    for epoch in range(1,epochs+1):
        start = time.time()

        print("Epoch: ", epoch)
        generator_loss = 0
        discriminator_loss = 0
        for n, (target, input_image) in train_ds.enumerate():
            gen_total_loss, disc_loss = train_step(input_image, target, epoch)
            generator_loss += gen_total_loss
            discriminator_loss += disc_loss
        if epoch%10 == 0:
            generate_samples(epoch)

        print ('Time: {} sec, gen loss = {}, disc loss = {}.'.format(time.time()-start, generator_loss, discriminator_loss))

    total_time = time.time() - total_start
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Total training time: {:.2f} sec ({:02d}h {:02d}m {:05.2f}s)'.format(total_time, int(hours), int(minutes), seconds))

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
fit(train_dataset, epochs = 100)
generator.save("generator.keras")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Evaluation

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}

def transform_image(image):
    maximum = np.max(image)
    image = image/maximum*255
    image = np.array(image).astype('uint8')
    return image

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### 7.1 Initial values of metrics between noisy and clear image

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
mse = 0
nrmse = 0
ssim = 0
total_images = 0
for n, (target, input_image) in tqdm(test_dataset.enumerate()):
    total_images += target.shape[0]
    for i in range(target.shape[0]):
        t_im = transform_image(target[i])
        p_im = transform_image(input_image[i])
        mse += mean_squared_error(t_im, p_im)
        nrmse += normalized_root_mse(t_im, p_im)
        ssim += structural_similarity(t_im, p_im, channel_axis=2)
mse /= total_images
nrmse /= total_images
ssim /= total_images

print(mse, nrmse, ssim)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### 7.2 Final values of metrics between generated and clear image

# %% [code] {"execution":{},"jupyter":{"outputs_hidden":false}}
mse = 0
nrmse = 0
ssim = 0
total_images = 0
for n, (target, input_image) in tqdm(test_dataset.enumerate()):
    predicted_images = generator(input_image)
    total_images += target.shape[0]
    for i in range(target.shape[0]):
        t_im = transform_image(target[i])
        p_im = transform_image(predicted_images[i])
        mse += mean_squared_error(t_im, p_im)
        nrmse += normalized_root_mse(t_im, p_im)
        ssim += structural_similarity(t_im, p_im, channel_axis=2)
mse /= total_images
nrmse /= total_images
ssim /= total_images

print(mse, nrmse, ssim)
