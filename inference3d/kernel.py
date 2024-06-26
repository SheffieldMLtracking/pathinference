import numpy as np
import tensorflow as tf

def EQ(X1,X2):
    """
    Our kernel. An EQ kernel. The prior is UNCORRELATED between dimensions.
    """
    cov = (50**2)*np.exp(-np.sum(np.subtract(X1[:,None],X2[None,:])**2/(2*1.2**2),2))
    axsel = tf.cast((X1[:,1][:,None]==X2[:,1][None,:]),dtype=tf.float32)
    cov = cov * axsel
    return cov
