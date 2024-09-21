import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import numpy as np

img = cv2.imread('/grp.jpeg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

plt.imshow(img)
plt.show()

X = img.reshape(-1, 3)
print(X.shape)

k = 4
model = KMeans(n_clusters=k)
model.fit(X)

centroids = model.cluster_centers_
print(centroids)

colors = np.array(centroids, dtype = 'uint8')
print(colors)

#color patch

i = 1
for color in colors:
    plt.subplot(1, k, i)    # Create subplots with 1 row and k columns
    plt.axis('off')         # Turn off the axis
    i += 1                  # Increment subplot index
    mat = np.zeros((100, 100, 3), dtype='uint8')  # Create 100x100 matrix with 3 channels (RGB)
    mat[:, :, :] = color    # Fill the matrix with the current color

    plt.imshow(mat)         # Display the matrix as a color patch

plt.show()


#  Basic segmentation based upon similar color regions
# segmentation partitions an image into regions
# having similar visual appearance corresponding to parts of objects

print(colors)

np.unique(model.labels_)
model.labels_.shape

newImg = np.zeros(X.shape, dtype = 'uint8')
print(newImg.shape)

for i in range(newImg.shape[0]):
  newImg[i] = colors[model.labels_[i]]
newImg = newImg.reshape(img.shape)
print(newImg.shape)

plt.imshow(newImg)
