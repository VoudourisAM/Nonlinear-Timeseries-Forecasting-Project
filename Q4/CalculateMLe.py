#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import linregress


# In[2]:


def calculate_average_log_divergence(embedded, nearest_neighbors, max_steps=50):
    number_of_points = len(embedded)
    average_log_divergence = []

    for step in range(max_steps):
        log_distances = []

        for i, neighbor in enumerate(nearest_neighbors):
            future_i = i + step
            future_neighbor = neighbor + step

            if (
                future_i < number_of_points
                and future_neighbor < number_of_points
            ):
                distance = np.linalg.norm(
                    embedded[future_i] -
                    embedded[future_neighbor]
                )

                if distance > 0:
                    log_distances.append(np.log(distance))

        if len(log_distances) == 0:
            break

        average_log_divergence.append(
            np.mean(log_distances)
        )

    return np.asarray(average_log_divergence)


# In[ ]:




