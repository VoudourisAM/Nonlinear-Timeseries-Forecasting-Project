#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.spatial.distance import cdist
import numpy as np


# In[2]:


def find_nearest_neighbors(embedded, theiler_window):
    number_of_points = len(embedded)

    distance_matrix = cdist(embedded, embedded, metric="euclidean")

    np.fill_diagonal(distance_matrix, np.inf)

    for i in range(number_of_points):
        start = max(0, i - theiler_window)
        end = min(number_of_points, i + theiler_window + 1)

        distance_matrix[i, start:end] = np.inf

    nearest_neighbors = np.argmin(distance_matrix, axis=1)

    return nearest_neighbors


# In[ ]:




