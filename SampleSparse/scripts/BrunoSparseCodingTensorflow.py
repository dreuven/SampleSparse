from PersonalPlotting import PPlotting
from scipy import io
import tensorflow as tf
import scipy
import numpy as np
import sys
from scipy.optimize import minimize
import Miscellaneous as ms
import os
import traceback
class TensorSparse:
    num_receptive_fields = None
    size_of_patch = None
    num_patches_from_image = None
    lambda_parameter = None
    LR = None
    plot_directory = None
    plot_obj = None
    our_images = None
    batch_size = None
    phis = None
    data = None
    a_matr = None
    train_a_step = None
    energy_function = None
    sess = None
    reconstruction_error_array = []
    def __init__(self, num_receptive_fields, size_of_patch, batch_size,lambda_parameter, session_object, LR = 1e-1, plot_directory = 'Plots', phis = None):
        self.num_receptive_fields = num_receptive_fields
        self.size_of_patch = size_of_patch
        self.batch_size = batch_size
        self.lambda_parameter = lambda_parameter
        self.LR = LR
        if os.path.exists(plot_directory):
            ms.clear_Paths_folder(plot_directory)
        else:
            os.mkdir(plot_directory)
        self.sess = session_object
        self.plot_obj = PPlotting(plot_directory)
        self.data = tf.Variable(tf.truncated_normal(shape = (self.size_of_patch**2, self.batch_size)))
        self.phis = self.create_phis()
        self.a_matr = tf.Variable(tf.zeros(shape = (self.num_receptive_fields, self.batch_size), dtype = tf.float32))
        self.energy_function = self.energy_function()
        self.train_a_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.energy_function,var_list = [self.a_matr])
    def energy_function(self):
        reconstruction = tf.matmul(self.phis, self.a_matr)
        reconstruction = tf.cast(reconstruction, tf.float64)
        squared_error = (tf.cast(self.data,tf.float64) - reconstruction) ** 2.0
        norm_array = tf.abs(self.a_matr)
        left_side = tf.reduce_mean(0.5 * tf.reduce_sum(squared_error, reduction_indices = 0))
        left_side = tf.cast(left_side, tf.float64)
        print("LAMBDA PARAM IS: ", self.lambda_parameter)
        a_value_summed = tf.reduce_mean(self.lambda_parameter * tf.reduce_sum(norm_array, reduction_indices=0))
        right_side =  a_value_summed
        right_side = tf.cast(right_side, tf.float64)
        to_ret = left_side + right_side
        return tf.cast(to_ret, tf.float32)
    def create_phis(self):
        return tf.Variable(tf.truncated_normal(shape = (self.size_of_patch**2, self.num_receptive_fields)))
    def load_data(self, data):
        print("The norm of the data is ", np.mean(np.linalg.norm(data,axis=0)))
        assign = tf.assign(self.data, data)
        self.sess.run(assign)
    def infer_a_coefficients(self, session_object, iter_num):
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
    def update_phis(self):
       # Assumes a_matr is a_optimal is set. Will have to be so by script that runs this function after the loop
       a_matr_cast = tf.cast(self.a_matr, tf.float64)
       phis_cast = tf.cast(self.phis, tf.float64)
       residual = tf.cast(self.data,tf.float64) - tf.matmul(phis_cast,a_matr_cast)
       residual_sum = tf.reduce_mean(tf.reduce_sum(residual, reduction_indices = 0))
       val_error = self.sess.run(residual_sum)
       # Visualize input here
       print("plotting data")
       self.plot_obj.plot_input_data(self.sess.run(self.data), 6)
       self.plot_obj.plot_reconstructions(self.sess.run(tf.matmul(phis_cast, a_matr_cast)), 00)
       print("Val of Residual error after we do learningis: {}".format(val_error))
       self.reconstruction_error_array.append(val_error)
       dbasis = tf.matmul(residual, tf.transpose(a_matr_cast))
       norm_grad_basis = tf.sqrt(tf.reduce_sum(dbasis ** 2, reduction_indices = 0))
       dbasis = dbasis / norm_grad_basis
       phis = self.phis + tf.cast(self.LR * dbasis,tf.float32)
       phi_norm = tf.sqrt(tf.reduce_sum(phis ** 2.0, reduction_indices = 0))
       self.phis = phis/phi_norm
       print("The value sum of active coefficients after we do learning", np.sum(np.abs(self.sess.run(a_matr_cast))))
