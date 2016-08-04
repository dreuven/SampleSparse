#Sparse Coding with the inference of the A matrix handled by LAHMC.
#Implemented by Doron Reuven

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
import  pickle
import os
class lahmc_sampler:
    batch_data = None
    def __init__(self, num_receptive_fields, size_of_patch, batch_size,lambda_parameter, session_object, LR = 1e-1, num_particles_per_batch = 1, phis = None):
        self.num_receptive_fields = num_receptive_fields
        self.size_of_patch = size_of_patch
        self.batch_size = batch_size
        self.lambda_parameter = lambda_parameter
        self.LR = LR
        self.sess = session_object
        if phis is not None:
            self.phis = tf.Variable(phis)
            print("Load the phis!!")
        else:
            self.phis = tf.Variable(tf.truncated_normal(shape = (size_of_patch**2, num_receptive_fields)))
        #Could set the below to  values to None, None or k x N.
        self.a_matr = tf.placeholder("float32", [None, None])
        # self.batch_data = tf.Variable(tf.truncated_normal(shape = (size_of_patch**2, batch_size * num_particles_per_batch)))
        self.batch_data = tf.placeholder("float32", [None, None])
        self.data_locker = None
        self.sampling_results = dict()
        self.num_particles_per_batch = num_particles_per_batch
        self.sum_variable = tf.reduce_sum(self.E())
        self.gradient_w_resp_a = tf.gradients(self.sum_variable, [self.a_matr])[0]
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
        gradient = self.sess.run(self.gradient_w_resp_a, feed_dict =
                                 {self.a_matr:a, self.batch_data:self.data_locker})
        # print("Gradient is", gradient)
        return gradient
    def E_sample(self,a):
        return self.sess.run(self.sum_variable, feed_dict = {self.a_matr:a, self.batch_data:self.data_locker})
    def sample(self,ii,num_steps = 100, dirname = "Pickled_a_matrices"):
        print("In sample")
        Ainit = np.random.randn(self.num_receptive_fields, self.batch_size * self.num_particles_per_batch)
        sampler = LAHMC.LAHMC(Ainit,self.E_sample,self.gradient, epsilon=1., beta=0.1, num_look_ahead_steps=1)
        A_final = sampler.sample(num_steps)
        self.sampling_results[ii] = A_final
        if ii % 100 == 0 and ii != 0:
            if not os.path.exists(dirname):
                print("Creating directory:", dirname)
                os.mkdir(dirname)
            the_path = os.getcwd() + "/" +dirname
            name_of_pickle_file = dirname + "/" + "data_{0}-{1}.pkl".format(ii - 100, ii)
            output = open(name_of_pickle_file,'wb')
            print("Now pickling file for iterations {0} to {1}".format(ii - 100, ii))
            pickle.dump(self.sampling_results, output)
            print("Done pickling")
            #Visualizing energies
            self.plot_energies()
            #Resetting dictionary to empty dict to preserve space in the program.
            self.sampling_results = dict()
    def plot_energies(self):
        energy_values = []
        for a_matrix in self.sampling_results.values():
            energy_value = self.sess.run(self.E(), feed_dict = {self.a_matr:a_matrix,self.batch_data:self.data_locker})
            energy_values.append(energy_value)
        x_vals = np.arange(len(energy_values))
        plt.figure(1)
        plt.clf()
        plt.plot(x_vals, energy_values, 'ro')
        plt.draw()
        plt.pause(0.0001)
        plt.show(block=False)

    # def ret_ze_sample_energies(self):
    #     energy_vals_to_ret = []
    #     for results in self.sampling_results:
    #         energy_value = self.sess.run(self.E(), feed_dict = {self.a_matr: results, self.batch_data: self.data_locker})
    #         energy_vals_to_ret.append(energy_value)
    #     return energy_vals_to_ret
