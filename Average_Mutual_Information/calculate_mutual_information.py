#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import mutual_info_score
import numpy as np


# In[2]:


def calculate_mutual_information(series, max_lag=100, bins=32):
    series = np.asarray(series, dtype=float) #Transforn data (NumPy array)
    
    if max_lag >= len(series):
        raise ValueError("max_lag must be sorted from data")
        
    bin_edges = np.histogram_bin_edges(series, bins=bins)
    
    series_discrete = np.digitize(series, bin_edges[1:-1]) #Το bin_edges[1:-1]
    
    mi_values = [] # null value with append method
    
    for lag in range(1, max_lag + 1): #lag1...max_lag
        current_values = series_discrete[:-lag]
        delayed_values = series_discrete[lag:]
        
        mi = mutual_info_score(current_values, delayed_values)
        
        mi_values.append(mi)

    return np.array(mi_values)


# In[2]:


def find_first_local_minimum(mi_values, lags):
    tau = None
    minimum_index = None

    for i in range(1, len(mi_values) - 1):

        if (mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]):
            tau = lags[i]
            minimum_index = i
            break

    if tau is None:
        print("No local minimum was found")
        return None, None
    
    selected_mi = mi_values[minimum_index]
    print("Selected time delay τ =", tau)
    print("Mutual Information at the selected τ =", mi_values[minimum_index])
    return tau, minimum_index


# In[ ]:




