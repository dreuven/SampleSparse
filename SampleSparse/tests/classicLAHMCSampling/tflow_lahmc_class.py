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
       # self.phis = np.random.randn(size_of_patch**2, num_receptive_fields)
        self.a_matr = tf.Variable(tf.truncated_normal(shape = (num_receptive_fields, batch_size)))
        self.sess.run(tf.initialize_all_variables())
        ##Instantiating images just to make the initialiation of the energy function work
        self.batch_data = np.random.randn(size_of_patch**2, batch_size)
        self.energy_function = self.E(self.a_matr)
    def load_batch(self,images):
        self.batch_data = images


    def E_sample(self,a,sigma = 1.):
        assign_a = tf.assign(self.a_matr, a)
        self.sess.run(assign_a)
        reconstruction_error = self.batch_data - self.sess.run(self.phis).dot(self.sess.run(self.a_matr))
        tmp = reconstruction_error**2/(2*sigma**2)
        tmp = tmp.sum(axis = 0).mean()
        sparsity = self.lambda_parameter * np.abs(self.sess.run(self.a_matr)).sum(axis = 0).mean()
        print("in E sample, Tmp and sparsity term is", tmp + sparsity)
        return tmp + sparsity
    def E(self,a,sigma = 1.):
        assign_a = tf.assign(self.a_matr, a)
        self.sess.run(assign_a)
        reconstruction_error = self.batch_data - tf.matmul(self.phis,self.a_matr)
        tmp = reconstruction_error**2/(2*sigma**2)
        tmp = tf.reduce_mean(tf.reduce_sum(tmp, reduction_indices = 0))
        sparsity = self.lambda_parameter * tf.reduce_mean(tf.reduce_sum(tf.abs(self.a_matr), reduction_indices=0))
        print("In E,Tmp and sparsity term is", tmp + sparsity)
        return tmp + sparsity

    def de_da(self,a, sigma = 1.):
        x = tf.assign(self.a_matr, a)
        self.sess.run(x)
        gradient = tf.gradients(self.energy_function,[self.a_matr])
        return self.sess.run(gradient)[0]
    def sample(self,num_steps = 100):
        Ainit = np.random.randn(self.num_receptive_fields,self.batch_size)
        sampler = LAHMC.LAHMC(Ainit,self.E_sample,self.de_da)
        A_final = sampler.sample(num_steps)
