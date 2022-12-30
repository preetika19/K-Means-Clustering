import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, img_as_float

def closest_centroids(X, c):
    K = np.size(c, 0)
    index = np.zeros((np.size(X, 0), 1))
    array = np.empty((np.size(X, 0), 1))
    for i in range(0, K):
        y = c[i]
        temp = np.ones((np.size(X, 0), 1)) * y
        b = np.power(np.subtract(X,temp), 2)
        a = np.sum(b, axis = 1)
        a = np.asarray(a)
        a.resize((np.size(X,0), 1))
        array = np.append(array, a, axis=1)
    array = np.delete(array, 0, axis=1)
    index = np.argmin(array, axis=1)
    return index

def compute_centroids(X, index, K):
    n = np.size(X, 1)
    centroids = np.zeros((K, n))
    for i in range(0, K):
        ci = index
        ci = ci.astype(int)
        total_number = sum(ci);
        ci.resize((np.size(X,0), 1))
        total_matrix = np.matlib.repmat(ci, 1, n)
        ci = np.transpose(ci)
        total = np.multiply(X, total_matrix)
        centroids[i] = ( 1 / total_number ) * np.sum(total, axis=0)
    return centroids

def plot_image_colors_by_color(name, image_vectors):
    figure = plt.figure()
    ax = Axes3D(figure)
    for rgb in image_vectors:
        ax.scatter(rgb[0], rgb[1], rgb[2], c = rgb, marker = 'o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    figure.savefig(name + '.png')
    

def plot_image_colors_by_label(name, image_vectors, lbs, cluster_proto):
    figure = plt.figure()
    ax = Axes3D(figure)
    for rgb_i, rgb in enumerate(image_vectors):
        ax.scatter(rgb[0], rgb[1], rgb[2], c = cluster_proto[lbs[rgb_i]], marker = 'o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    figure.savefig(name + '.png')

def k_means_clustering(image_vectors, k, iterations):
    lbs = np.full((image_vectors.shape[0],), -1)
    cluster_proto = np.random.rand(k, 3)
    for i in range(iterations):
        print('Iteration No.' , i)
        points_label = [None for k_i in range(k)]

        for rgb_i, rgb in enumerate(image_vectors):
            rgb_row = np.repeat(rgb, k).reshape(3, k).T
            closest_label = np.argmin(np.linalg.norm(rgb_row - cluster_proto, axis=1))
            lbs[rgb_i] = closest_label

            if (points_label[closest_label] is None):
                points_label[closest_label] = []

            points_label[closest_label].append(rgb)
       
        for k_i in range(k):
            if (points_label[k_i] is not None):
                new_cluster_prototype = np.asarray(points_label[k_i]).sum(axis = 0) / len(points_label[k_i])
                cluster_proto[k_i] = new_cluster_prototype

    return (lbs, cluster_proto)


  
img = sys.argv[1]
K = [2, 5, 10, 15, 20]
iterations = 20
image = io.imread(img)[:, :, :3] 
image = img_as_float(image)
image_dimensions = image.shape
image_name = image
image_vectors = image.reshape(-1, image.shape[-1])

info = os.stat(img)
print("Image size before : ", info.st_size / 1024, "KB")

for key in K:
  print('K = ' + str(key))
  lbs, color_centroids = k_means_clustering(image_vectors, key, iterations)
  output_image = np.zeros(image_vectors.shape)
  for i in range(output_image.shape[0]):
      output_image[i] = color_centroids[lbs[i]]

  comp_img = img.split('.')[0]
  output_image = output_image.reshape(image_dimensions)
  io.imsave(comp_img + 'CompressedK' + str(key) + '.jpg' , output_image)
  info = os.stat(comp_img + 'CompressedK' + str(key) + '.jpg')
  print("Compressed Image size with K="+ str(key) + ":" +  str(info.st_size / 1024) + "KB")
  

      