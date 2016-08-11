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
import pickle
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
        self.data_dump_location = plot_directory
        self.plot_obj = PPlotting(plot_directory)
        self.data = tf.placeholder("float32",[self.size_of_patch**2, self.batch_size])
        self.loaded_data = None
        self.phis = tf.placeholder("float32",[self.size_of_patch**2, self.num_receptive_fields])
        if phis != None:
            print("Phis loaded in and are of shape", phis.shape)
            self.phis_so_far = phis
        else:
            self.phis_so_far = np.random.randn(self.size_of_patch**2, self.num_receptive_fields)
        self.a_matr = tf.Variable(np.random.randn(num_receptive_fields, batch_size),dtype = "float32")
        # self.a_matr = tf.placeholder([num_receptive_fields, batch_size])
        self.energy_function = self.energy_function_func()
        self.train_a_step = tf.train.GradientDescentOptimizer(1e-3).minimize(self.energy_function,var_list = [self.a_matr])
        self.sess.run(tf.initialize_all_variables())
    def energy_function_func(self):
        reconstruction = tf.matmul(self.phis, self.a_matr)
        squared_error = (self.data - reconstruction) ** 2.0
        norm_array = tf.abs(self.a_matr)
        left_side = tf.reduce_mean(0.5 * tf.reduce_sum(squared_error, reduction_indices = 0))
        a_value_summed = tf.reduce_mean(self.lambda_parameter * tf.reduce_sum(norm_array, reduction_indices=0))
        to_ret = left_side + a_value_summed
        return to_ret
    def energy_function_value(self):
        #This function returns the scalar value of the energy function
        return self.sess.run(self.energy_function, feed_dict = {self.phis:self.phis_so_far, self.data:self.loaded_data})
    def load_data(self, data):
        print("Now normalizing data")
        data_ = data ** 2
        l_2_norm_columns = np.sqrt(data_.sum(axis = 0))
        norm_data = data/l_2_norm_columns
        print("The norm of the data is ", np.mean(np.linalg.norm(norm_data,axis=0)))
        self.loaded_data = norm_data
    def infer_a_coefficients(self):
        assign_a = tf.assign(self.a_matr, np.random.randn(self.num_receptive_fields, self.batch_size))
        self.sess.run(assign_a)
        i = 0
        while i < 8000:
            if i %500 == 0:
                print("Inferring a coefficients in Bruno sparse coding model, on iter {0}".format(i))
            i += 1
            self.sess.run(self.train_a_step, feed_dict = {self.data: self.loaded_data, self.phis: self.phis_so_far})
        a_evaled = self.sess.run(self.a_matr)
        print("Value of objective function after doing inference",self.sess.run(self.energy_function, feed_dict = {self.data:self.loaded_data, self.phis:self.phis_so_far}))
        print("Percent active coefficients after we do inference is: {}".format(len(a_evaled[np.abs(a_evaled)>1e-2])/float(self.num_receptive_fields*self.batch_size)))
    def update_phis(self, ii, dirname):
        # Assumes a_matr is a_optimal is set. Will have to be so by script that runs this function after the loop
        residual = self.data - tf.matmul(self.phis, self.a_matr)
        residual_sum = tf.reduce_mean(tf.reduce_sum(residual, reduction_indices = 0))
        dbasis = (1/float(self.batch_size))* tf.matmul(residual, tf.transpose(self.a_matr))
        norm_grad_basis = tf.sqrt(tf.reduce_sum(dbasis ** 2, reduction_indices = 0))
        dbasis = dbasis / norm_grad_basis
        phis = self.phis + self.LR * dbasis
        phi_norm = tf.sqrt(tf.reduce_sum(phis ** 2.0, reduction_indices = 0))
        self.phis_so_far = self.sess.run(phis/phi_norm, feed_dict = {self.phis:self.phis_so_far, self.data: self.loaded_data})
        if ii % 100 == 0:
            name_of_pickle_file = self.data_dump_location + "phis.pkl"
            output = open(name_of_pickle_file, 'wb')
            print("Now pickling phis for sparse coding")
            pickle.dump(self.phis_so_far,output)
            print("Done pickling")
            output.close()
        print("The value sum of active coefficients after we do learning", np.sum(np.abs(self.sess.run(self.a_matr))))
