#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import linregress


# In[2]:


def create_embedding(series, embedding_dim, tau):

    series = np.asarray(series, dtype=float)

    # Υπολογίζουμε πόσα ολοκληρωμένα σημεία μπορούν να δημιουργηθούν.
    number_of_points = len(series) - (embedding_dim - 1) * tau

    if number_of_points <= 0:
        raise ValueError("The time series is too short for these parameters.")

    # Δημιουργούμε τον πίνακα του attractor.
    embedded = np.empty((number_of_points, embedding_dim))

    # Κάθε στήλη είναι μία καθυστερημένη έκδοση της σειράς.
    for dimension in range(embedding_dim):
        start = dimension * tau
        end = start + number_of_points

        embedded[:, dimension] = series[start:end]

    return embedded 


# In[3]:


def calculate_correlation_integral(embedded, radii):

    distances = pdist(embedded, metric="euclidean")

    correlation_integrals = []

    for radius in radii:

        correlation_value = np.mean(distances < radius)

        correlation_integrals.append(correlation_value)

    return np.asarray(correlation_integrals)


# In[4]:


def estimate_correlation_dimension(radii, correlation_integrals):
    """
    Estimates the correlation dimension from the slope
    of the approximately linear log-log region.
    """

    valid = (
        (correlation_integrals > 0.01)
        & (correlation_integrals < 0.20)
    )

    log_r = np.log(radii[valid])
    log_c = np.log(correlation_integrals[valid])

    if len(log_r) < 3:
        return np.nan, np.nan

    result = linregress(log_r, log_c)

    correlation_dimension = result.slope
    r_squared = result.rvalue ** 2

    return correlation_dimension, r_squared


# In[ ]:




