import tensorflow as tf
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import random
from LAHMC.python import LAHMC
import pdb
import sys
import traceback
class lahmc_sampler:
    batch_data = None
    def __init__(self, num_receptive_fields, size_of_patch, batch_size,lambda_parameter, session_object, LR = 1e-1):
        self.num_receptive_fields = num_receptive_fields
        self.size_of_patch = size_of_patch
        self.batch_size = batch_size
        self.lambda_parameter = lambda_parameter
        self.LR = LR
        self.sess = session_object
        self.phis = tf.Variable(tf.truncated_normal(shape = (size_of_patch**2, num_receptive_fields)))
        self.a_matr =  tf.Variable(tf.truncated_normal(shape = ( num_receptive_fields, batch_size)))
       # self.phis = np.random.randn(size_of_patch**2, num_receptive_fields)
        self.sess.run(tf.initialize_all_variables())
        ##Instantiating images just to make the initialiation of the energy function work
        self.batch_data = np.random.randn(size_of_patch**2, batch_size)
    def load_batch(self,images):
        self.batch_data = images


    def E(self,a,sigma = 1.):
        assign_step = tf.assign(self.a_matr, a)
        self.sess.run(assign_step)
        a = self.a_matr
        reconstruction_error = self.batch_data - tf.matmul(self.phis,a)
        tmp = reconstruction_error**2/(2*sigma**2)
        tmp = tf.reduce_mean(tf.reduce_sum(tmp, reduction_indices = 0))
        sparsity = self.lambda_parameter * np.abs(a).sum(axis = 0).mean()
        print("Sparsity is", self.sess.run(sparsity))
        print("In E,Tmp and sparsity term is", tmp + sparsity)
        return tmp + sparsity

    def gradient(self,a,     sigma = 1.):
        return tf.gradients(self.E(a),[a])
    def E_sample(self,a):
        og_energy = self.E(a)
        return self.sess.run(og_energy)
    def sample(self,num_steps = 100):
        Ainit = np.random.randn(self.num_receptive_fields,self.batch_size)
        sampler = LAHMC.LAHMC(Ainit,self.E_sample,self.gradient)
        A_final = sampler.sample(num_steps)
