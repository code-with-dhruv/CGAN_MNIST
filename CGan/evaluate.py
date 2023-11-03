import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os

# Load the trained generator model


generator = load_model(r"C:\Users\dhruv\Downloads\GAN_MNIST\CGan\mnist_conditional_generator_epochs.h5")#model path goes here

# Generate and plot sample images for each digit (0-9)
latent_dim = 100
n_samples_per_digit = 10

for digit in range(10):
    labels = np.full((n_samples_per_digit, 1), digit)
    latent_points = np.random.randn(n_samples_per_digit, latent_dim)
    generated_images = generator.predict([latent_points, labels])
    
    plt.figure(figsize=(10, 1))
    for i in range(n_samples_per_digit):
        plt.subplot(1, n_samples_per_digit, i + 1)
        plt.axis('off')
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')

    plt.suptitle(f'Sample Images for Digit {digit}')
    plt.show()




plt.show()
