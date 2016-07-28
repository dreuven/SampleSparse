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
class lahmc_sampler:
    batch_data = None
    def __init__(self, num_receptive_fields, size_of_patch, batch_size,lambda_parameter, session_object, LR = 1e-1, num_particles_per_batch = 1):
        self.num_receptive_fields = num_receptive_fields
        self.size_of_patch = size_of_patch
        self.batch_size = batch_size
        self.lambda_parameter = lambda_parameter
        self.LR = LR
        self.sess = session_object
        self.phis = tf.Variable(tf.truncated_normal(shape = (size_of_patch**2, num_receptive_fields)))
        self.a_matr =  tf.Variable(tf.truncated_normal(shape = ( num_receptive_fields, batch_size * num_particles_per_batch)))
       # self.phis = np.random.randn(size_of_patch**2, num_receptive_fields)
        self.batch_data = tf.Variable(tf.truncated_normal(shape = (size_of_patch**2, batch_size * num_particles_per_batch)))
        ##Instantiating images just to make the initialiation of the energy function work
        self.sample_results = None
        self.num_particles_per_batch = num_particles_per_batch
        self.train_a_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.energy_function, var_list = [self.a_matr])
        # The following variables are for comparison with the sampler, using the old approach of scipy optimize
        self.test_batch = tf.Variable(tf.truncated_normal(shape = (size_of_patch**2, batch_size)))
        self.test_a_matr = tf.Variable(tf.truncated_normal(shape = (num_receptive_fields, batch_size)))
        self.energy_function = tf.reduce_mean(self.E(self.a_matr), reduction_indices = 0)
        self.sess.run(tf.initialize_all_variables())
    def load_batch(self,images):
        print("In load data")
        print("Original size of images is", images.shape)
        copied_data = np.matlib.repmat(images,1, self.num_particles_per_batch)
        print("Copied size of the data is", copied_data.shape)
        print("Original data is", images)
        print("Copied data is",copied_data)
        assign_data = tf.assign(self.batch_data, copied_data)
        self.sess.run(assign_data)

    def E(self,a,sigma = 1.):
        assign_step = tf.assign(self.a_matr, a)
        self.sess.run(assign_step)
        reconstruction_error = self.batch_data - tf.matmul(self.phis,self.a_matr)
        tmp = reconstruction_error**2/(2*sigma**2)
        tmp = tf.reduce_sum(tmp, reduction_indices = 0)
        sparsity = tf.reduce_sum(self.lambda_parameter * np.abs(self.a_matr), reduction_indices = 0)
       # print("Sparsity is", self.sess.run(sparsity))
        #print("In E,Tmp and sparsity term is", tmp + sparsity)
        return tmp + sparsity

    def gradient(self,a,sigma = 1.):
        #Wondering whether I must assign a to self.a_matr...?
        #Also should the gradient be with respect to E or E sample
        gradient = tf.gradients(tf.reduce_sum(self.E(a)),[self.a_matr])
        return self.sess.run(gradient)
    def E_sample(self,a):
        og_energy = self.E(a)
        return self.sess.run(tf.reduce_sum(og_energy))
    def sample(self,num_steps = 100):
        Ainit = np.random.randn(self.num_receptive_fields,self.batch_size)
        sampler = LAHMC.LAHMC(Ainit,self.E_sample,self.gradient)
        A_final = sampler.sample(num_steps)
        print("A final shape is", A_final.shape)
        self.sample_results = A_final
    def infer_coefficients_LBFGS(self):
        epsilon = 1e-8
        curr_val = 0.0
        prev_val = float("inf")
        num_iters = 0
        while abs(curr_val - prev_val) > epsilon and num_iters < 8000:
            prev_val = curr_val
            self.sess.run(self.train_a_step)
            curr_val = session_object.run(self.energy_function)
            num_iters += 1
        a_evaled = self.sess.run(self.a_matr)
        print("Value of objective function after doing inference",curr_val)
        print("Percent active coefficients after we do inference is: {}".format(len(a_evaled[np.abs(a_evaled)>1e-2])/float(self.num_receptive_fields*self.batch_size)))
