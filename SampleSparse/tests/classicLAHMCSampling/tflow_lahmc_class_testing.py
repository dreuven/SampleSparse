import tensorflow as tf
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import random
from LAHMC.python import LAHMC
import pdb
import sys
import traceback
import numpy as np
import numpy.matlib
import ipdb
class lahmc_sampler:
    batch_data = None
    def __init__(self, num_receptive_fields, size_of_patch, batch_size,lambda_parameter, session_object, LR = 1e-1, num_particles_per_batch = 1):
        self.num_receptive_fields = num_receptive_fields
        self.size_of_patch = size_of_patch
        self.batch_size = batch_size
        self.lambda_parameter = lambda_parameter
        self.LR = LR
        self.sess = tf.Session()
        self.phis = tf.Variable(tf.truncated_normal(shape = (size_of_patch**2, num_receptive_fields)))
        #Could set the below to  values to None, None or k x N.
        self.a_matr = tf.placeholder("float32", [None, None])
        # self.batch_data = tf.Variable(tf.truncated_normal(shape = (size_of_patch**2, batch_size * num_particles_per_batch)))
        self.batch_data = tf.placeholder("float32", [None, None])
        self.data_locker = None
        self.sampling_results = None
        self.num_particles_per_batch = num_particles_per_batch
        self.sum_variable = tf.reduce_sum(self.E())
        self.gradient_w_resp_a = tf.gradients(self.sum_variable, [self.a_matr])
        self.sess.run(tf.initialize_all_variables())
    def load_batch(self,images):
        print("Loading Data.")
        print("Original size of images is", images.shape)
        print("Now going to normalize images")
        images_ = images ** 2
        l_2_norm_columns = np.sqrt(images_.sum(axis = 0))
        images_ = images/l_2_norm_columns
        # check_im = images_**2
        # check_im = np.sum(check_im, axis = 0)
        # check_im = np.sqrt(check_im)
        copied_data = np.matlib.repmat(images_,1, self.num_particles_per_batch)
        print("Copied size of the data is", copied_data.shape)
        # assign_data = tf.assign(self.batch_data, copied_data)
        # self.sess.run(assign_data)
        self.data_locker = copied_data

    def E(self,sigma = 1.):
        reconstruction_error = self.batch_data - tf.matmul(self.phis,self.a_matr)
        tmp = reconstruction_error**2/(2*sigma**2)
        tmp = tf.reduce_sum(tmp, reduction_indices = 0)
        sparsity = tf.reduce_sum(self.lambda_parameter * np.abs(self.a_matr), reduction_indices = 0)
        return tmp + sparsity

    def gradient(self,a,sigma = 1.):
        if a.shape[1] != self.batch_size * self.num_particles_per_batch:
            ipdb.set_trace()
        # print("Dimensions of _a_ input to gradient are", a.shape)
        gradient = self.sess.run(self.gradient_w_resp_a, feed_dict = {self.a_matr:a, self.batch_data:self.data_locker})
        # print("Gradient is", gradient)
        return gradient
    def E_sample(self,a):
        return self.sess.run(self.sum_variable, feed_dict = {self.a_matr:a, self.batch_data:self.data_locker})
    def sample(self,num_steps = 100):
        Ainit = np.random.randn(self.num_receptive_fields,self.batch_size * self.num_particles_per_batch)
        sampler = LAHMC.LAHMC(Ainit,self.E_sample,self.gradient)
        A_final = sampler.sample(num_steps)
        print("A final shape is", A_final.shape)
        self.sampling_results = A_final
    def ret_ze_sample_energies(self):
        energy_values = self.sess.run(self.E(), feed_dict = {self.a_matr : self.sampling_results})
        return energy_values
