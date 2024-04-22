# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Pick customer segment quantity (k)
2. Seed cluster centers with random data points.
3. Assign customers to closest centers
4. Re-center clusters and repeat until stable

## Program:

/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: pavithra D

RegisterNumber:  212223230146

*/
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data = pd.read_csv("/content/Mall_Customers_EX8.csv")
data

X = data[['Annual Income (k$)' , 'Spending Score (1-100)']]
X

plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel("Spending Score (1-100)")
plt.show()

k = 5
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("Centroids: ")
print(centroids)
print("Label:")
print(labels)

colors = ['r', 'g', 'b', 'c', 'm']

for i in range(k):
  cluster_points = X[labels == i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i+1}')

distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)

circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, color='k', label='Centroids')

plt.title('K-means Clustering')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal') 
plt.show()
```
## Output:
HEAD:
![image](https://github.com/PavithraD23004871/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/138955967/a0198e82-90a0-45be-8072-72a975df76c3)

X VALUE:

![image](https://github.com/PavithraD23004871/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/138955967/6efebdb2-4acf-43ce-b546-2bfe15c1a058)

PLOT:

![image](https://github.com/PavithraD23004871/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/138955967/f2a0aea9-f99e-455a-84bd-dfc1b113256a)

Centroid and Label:

![image](https://github.com/PavithraD23004871/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/138955967/6881b2a8-ad13-48ee-b0ec-10ec17dc9439)

K-means clustering:

![image](https://github.com/PavithraD23004871/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/138955967/7856a053-006f-4029-a2e3-001a2ff7fa0f)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
