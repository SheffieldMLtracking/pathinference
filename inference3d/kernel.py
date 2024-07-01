import numpy as np
import tensorflow as tf

def EQ(X1,X2,ls,scale):
    """
    Our kernel. An EQ kernel. The prior is UNCORRELATED between dimensions.
    ls : The lengthscale of the EQ kernel
    scale : The scale of the kernel
    """
    cov = (scale**2)*np.exp(-np.sum(np.subtract(X1[:,None],X2[None,:])**2/(2*ls**2),2))
    axsel = tf.cast((X1[:,1][:,None]==X2[:,1][None,:]),dtype=tf.float32)
    cov = cov * axsel
    return cov
    
def kernelwrap(kernel, *args, **kwargs):
    def newkernel(X1, X2):
        return kernel(X1, X2, *args, **kwargs)
    return newkernel
