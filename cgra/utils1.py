import scipy
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm, chisquare
import numpy as np

def plot_error_bar(axis, data, bins, name="unnamed", domain=100):
    # domain of 100 works well for cgra model
    filtered = data[(data>=-domain) & (data<=domain)]
    (mu, sigma) = norm.fit(filtered)
    pdf_fit = norm.pdf(bins, mu, sigma)
    # chi = chisquare(f_obs=pdf_fit, f_exp=data)
    print(f'{name} mu: {mu}, sigma: {sigma}, sigma (numpy): {np.std(data)}')
    axis.plot(bins, pdf_fit, 'r--', linewidth=5)