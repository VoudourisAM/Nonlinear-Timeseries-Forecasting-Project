#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt

import mplcyberpunk as mplcp

import numpy as np


# In[5]:


def f_data(data):
    plt.figure(figsize=(12, 4))
    plt.plot(data, linewidth=0.7, color='orange')
    plt.xlabel("Χρονικό βήμα")
    plt.ylabel("Τιμή")
    plt.title("Χρονοσειρά w31")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    mplcp.make_lines_glow()
    plt.show()


# In[1]:


def plot_mutual_information(lags, mi_values, tau, minimum_index):

    plt.figure(figsize=(12, 6))

    # Σχεδιάζουμε τη Mutual Information για κάθε lag.
    plt.plot(lags, mi_values, marker="o", markersize=3, linewidth=1)

    # Αν βρέθηκε το πρώτο τοπικό ελάχιστο,
    # το σημειώνουμε πάνω στο γράφημα.
    if tau is not None:
        plt.scatter(tau, mi_values[minimum_index], s=80, zorder=3, label=f"First local minimum: τ = {tau}")

        # Κάθετη διακεκομμένη γραμμή στο επιλεγμένο τ.
        plt.axvline(tau, linestyle="--", color="red")

    plt.title("Average Mutual Information")
    plt.xlabel("Time delay τ")
    plt.ylabel("Mutual Information")
    plt.grid(alpha=0.3)

    if tau is not None:
        plt.legend()

    plt.tight_layout()
    plt.show()


# In[1]:


def plot_correlation_integral(radii, correlation_integrals):
    valid = correlation_integrals > 0

    log_r = np.log(radii[valid])
    log_c = np.log(correlation_integrals[valid])

    plt.figure(figsize=(8, 5))
    plt.plot(log_r, log_c, marker="o", markersize=4)

    plt.title("Correlation Integral")
    plt.xlabel("log(r)")
    plt.ylabel("log(C(r))")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# In[2]:


def plot_correlation_dimension(embedding_dimensions, estimated_dimensions):
    plt.figure(figsize=(8, 5))

    plt.plot(list(embedding_dimensions), estimated_dimensions,marker="o")

    plt.title("Correlation Dimension vs Embedding Dimension")
    plt.xlabel("Embedding dimension m")
    plt.ylabel("Estimated correlation dimension D₂")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# In[1]:


def plot_lyapunov_fit(steps, average_log_divergence, fit_start, fit_end,lyapunov_exponent, intercept):

    selected_steps = steps[fit_start:fit_end]

    fitted_values = (intercept + lyapunov_exponent * selected_steps)

    plt.figure(figsize=(8, 5))

    plt.plot(steps, average_log_divergence, marker="o", markersize=4, label="Average log divergence")

    plt.plot(selected_steps, fitted_values, linewidth=2, label=f"Linear fit: λ = {lyapunov_exponent:.4f}")

    plt.title("Maximal Lyapunov Exponent")
    plt.xlabel("Time step k")
    plt.ylabel("Average log distance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[ ]:




