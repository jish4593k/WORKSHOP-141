import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans
from imageio import imread, imsave
from scipy.cluster.vq import vq
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def load_image():
    file_path = filedialog.askopenfilename()
    return imread(file_path)

def display_image(image):
    img = Image.fromarray(np.uint8(image))
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(window, image=img)
    panel.image = img
    panel.pack()

def k_means_clustering(image, num_clusters):
    # Reshape the image into an array
    rows, cols, channels = image.shape
    image_reshaped = image.reshape((rows * cols, channels))

    # Run MiniBatchKMeans on the reshaped image
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=1000)
    kmeans.fit(image_reshaped)
    labels, _ = vq(image_reshaped, kmeans.cluster_centers_)

    print(min(labels), max(labels))

    # Substitute each pixel's RGB coordinates with the coordinates of its centroid
    image_compressed = kmeans.cluster_centers_[labels, :]

    # Recover the original image shape
    image_compressed_reshaped = image_compressed.reshape((rows, cols, channels))

    return image_compressed_reshaped

def compress_image():
    num_clusters = 5

    # Load image
    original_image = load_image()

    # Display the original image
    display_image(original_image)

    print('\nApplying K-Means to perform image compression.\n\n')

    # Perform K-Means clustering
    compressed_image = k_means_clustering(original_image, num_clusters)


    display_image(compressed_image)

    # Save the compressed image
    imsave('compressed_image.png', compressed_image)

window = tk.Tk()
window.title("Image Compression with K-Means")
window.geometry("800x600")


compress_button = tk.Button(window, text="Load and Compress Image", command=compress_image)
compress_button.pack()

window.mainloop()
