# Restoration GAN Playground
A playground for iterating on U-Net GAN image restoration.
Initial code use for all experiments ([gan_model_original.py](gan_model_original.py)) was sourced from [Image Restoration using GAN | Kaggle](https://www.kaggle.com/code/arindombora10/image-restoration-using-gan).

# Training Environments

## Local #1

- Video Card: nVidia GeForce RTX 3090


# Experiments

- Experiment 1 (baseline)
    - Model: [gan_model_exp_01.py](gan_model_exp_01.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 loss
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32
    - Results:
        - Training successful

- Experiment 2
    - Model: [gan_model_exp_02.py](gan_model_exp_02.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 loss
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 32-256
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32

- Experiment 3
    - Model: [gan_model_exp_03.py](gan_model_exp_03.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 loss
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 85
    - Activation function: ReLU
    - Batch Size: 32

- Experiment 4
    - Model: [gan_model_exp_04.py](gan_model_exp_04.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 + perceptual loss
        - adding a perceptual loss (or VGG perceptual loss) to the generator objective
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32
    - Results:
        - Considerable color shift
        - Checkerboard pattern
        - Generator collapse across entire training time

- Experiment 4.1
    - Model: [gan_model_exp_04.1.py](gan_model_exp_04.1.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 + perceptual loss
        - adding a perceptual loss (or VGG perceptual loss) to the generator objective
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32

- Experiment 4.2
    - Model: [gan_model_exp_04.2.py](gan_model_exp_04.2.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 + perceptual loss
        - adding a perceptual loss (or VGG perceptual loss) to the generator objective
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32
    - Implemented fix for the perceptual loss scaling bug.
        - VGG19 expects images with pixel values in the range [0, 255] — standard 8-bit image values.
        - The "perceptual_loss" function applies (x + 1.0) * 127.5, which assumes inputs are in [-1, 1], but the images were normalized to [0, 1]. This means the VGG preprocessing receives values in [127.5, 255] instead of [0, 255]
        - fix:
            ```python
                gen = tf.keras.applications.vgg19.preprocess_input(gen_clipped * 255)
                tar = tf.keras.applications.vgg19.preprocess_input(target * 255)
            ```
    - Results:
        - Fixed color shift issue
        - Did not fix discriminator collapse issue

- Experiment 4.2.1
    - Model: [gan_model_exp_04.2.1.py](gan_model_exp_04.2.1.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 + perceptual loss
        - adding a perceptual loss (or VGG perceptual loss) to the generator objective
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32
    - Results:
        - Fixed discriminator collapse issue slightly; still collapsed at the end

- Experiment 4.2.2
    - Model: [gan_model_exp_04.2.2.py](gan_model_exp_04.2.2.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 + perceptual loss
        - adding a perceptual loss (or VGG perceptual loss) to the generator objective
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32
    - Results:
        - Fixed discriminator collapse issue completely

- Experiment 5
    - Model: [gan_model_exp_05.py](gan_model_exp_05.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 loss
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: Sigmoid
    - Batch Size: 32

- Experiment 6
    - Model: [gan_model_exp_06.py](gan_model_exp_06.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 loss + SSIM loss
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32

- Experiment 7
    - Model: [gan_model_exp_07.py](gan_model_exp_07.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 loss
    - Discriminator loss function: Binary cross-entropy GAN loss
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 64

- Experiment 8
    - Model: [gan_model_exp_08.py](gan_model_exp_08.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 loss
    - Discriminator loss function: Binary cross-entropy GAN loss change
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32

- Experiment 9
    - Model: [gan_model_exp_09.py](gan_model_exp_09.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 loss
    - Discriminator loss function: Binary cross-entropy GAN loss change
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32

- Experiment 10
    - Model: [gan_model_exp_10.py](gan_model_exp_10.py)
    - Training environment: Local #1
    - Generator loss function: cGAN + L1 loss
    - Discriminator loss function: Binary cross-entropy GAN loss change
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32

- Experiment 11
    - Model: [gan_model_exp_11.py](gan_model_exp_11.py)
    - Training environment: Local #1
    - Generator loss function: L1 loss
    - Discriminator loss function: Binary cross-entropy GAN loss change
    - Filter Size: 64-512
    - Lambda value: 100
    - Activation function: ReLU
    - Batch Size: 32
