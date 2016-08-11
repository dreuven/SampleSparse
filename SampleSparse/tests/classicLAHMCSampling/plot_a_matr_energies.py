import pickle
import numpy as np
import sys
import os
import argparse
from lahmc_sparse import lahmc_sampler
from sparsecoding.BrunoSparseCodingTflow import TensorSparse
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("search_location", help = "Input directory to search through for phi and a matrices.")
args = parser.parse_args()
search_path = args.search_location
print("Search path is {0}".format(search_path))
batch_data_path = search_path + "/batch_data.pkl"
print("batch_data path is")
print(batch_data_path)
# print(batch_data_regex)
# batch_data_file = glob.glob(batch_data_regex)
file_pkl = open(batch_data_path, 'rb')
########
batch_data = pickle.load(file_pkl)
########
print("Batch Data shape is: ")
print(batch_data.shape)

phi_path = search_path + "/phis.pkl"
phi_file = open(phi_path, 'rb')
#########
phis = pickle.load(phi_file)
########
################
#Now loading specifications in order to instantiate classes
model_path = search_path + "/model.pkl"
model_file = open(model_path, 'rb')
model_dict = pickle.load(model_file)
num_receptive_fields = model_dict["num_receptive_fields"]
size_of_patch = model_dict["size_of_patch"]
batch_size = model_dict["batch_size"]
lambda_parameter = model_dict["lambda_parameter"]
LR = model_dict["LR"]
num_particles_per_image = model_dict["num_particles_per_image"]
#################
#Grab the a matrices of the LAHMC sampler
a_matrix_file_regex = search_path + "/data*.pkl"
list_a_matrices = glob.glob(a_matrix_file_regex)
#A matrices above is the list of all the files
#each of which contains 100, except for the first which has 101,
#a matrices. These a matrices are contained in a dictionary
#Thus the below is a file.

#A matrix dictionary, is a dictionary mapping the original input images indices to the a vectors learned from sampling
#There is an assumption I will make below that I should check, whether the E values returned are in the same order
#as the matrix. Pretty darn sure they are but still should check.
a_matrix_dict = dict()
for i in range(batch_size):
    a_matrix_dict = []
#Histogram dict is a dictionary mapping the original input images index to the energy values obtained from sampling
histogram_dict = dict()
for i in range(batch_size):
    histogram_dict[i] = []
#Instantiate the class to get histogram of a value energies
output_folder = "temp"
lahmc_class = lahmc_sampler(num_receptive_fields,size_of_patch,batch_size,lambda_parameter, num_particles_per_batch = num_particles_per_image, phis = phis, output_folder = output_folder)
lahmc_class.load_batch(batch_data)

histogram_dictionary_path = search_path + "/histogram_dictionary.pkl"
a_vector_path = search_path + "/a_vectors.pkl"
if not os.path.exists(histogram_dictionary_path) or not os.path.exists(a_vector_path):
    for file in list_a_matrices:
        print("Currently handling file {0}:\n".format(file))
        a_matrix_file_name = open(file, 'rb')
        a_matrix_file = pickle.load(a_matrix_file_name)
        print("Number of a_matrices in this file is",len(a_matrix_file.keys()))
        for value in a_matrix_file.values():
            E_arr = lahmc_class.sess.run(lahmc_class.E(),
                                     feed_dict = {lahmc_class.a_matr: value,
                                                  lahmc_class.batch_data:lahmc_class.data_locker})
            len_E_arr = E_arr.shape[0]
            for i in range(batch_size):
                #A_matrices
                slice_cols = slice(i:len_E_arr:batch_size)
                a_vectors_of_interest = value[:,slice_cols]
                for j in range(a_vectors_of_interest.shape[1]):
                    a_matrix_dict[i].append(a_vectors_of_interest[:,j])
                ##Below for histogram dict
                elems = E_arr[i:len_E_arr:batch_size]
                items = elems.tolist()
                histogram_dict[i].append(items)

    # Pickle a vectors
    a_vector_file = open(a_vector_path, 'wb')
    pickle.dump(a_matric_dict, a_vector_file)
    a_vector_file.close()
    #Pickle Energy values
    dictionary_file = open(histogram_dictionary_path, 'wb')
    pickle.dump(histogram_dict,dictionary_file)
    dictionary_file.close()
else:
    a_vector_dict_file = open(a_vector_path, 'rb')
    a_vector_dict = pickle.load(a_vector_dict_file)
    a_vector_dict_file.close()
    #####
    histogram_dict_file = open(histogram_dictionary_path, 'rb')
    histogram_dict = pickle.load(histogram_dict_file)
    histogram_dict_file.close()
######
#Begin Post Processing
for key in histogram_dict.keys():
    plt.figure()
    values = np.array(histogram_dict[key])
    plt_title = "Histogram for Img {0}".format(key)
    plt.xlabel("Energy Values Per Particle")
    plt.ylabel("Frequency of occurence")
    plt.hist(values, bins = 100, range = (0, 20))
    name_fig = search_path + "/Histogram_For_Img_{0}".format(key)
    plt.savefig(name_fig)

##Plotting lowest values and original images as well as reconstructions from sparse coding model
#Batch data is a p x N matrix so can plot each
sanity_plotting_directory = search_path + "/SanityPlots"
if not os.path.exists(sanity_plotting_directory):
    os.mkdir(sanity_plotting_directory)

for image_index in range(batch_data.shape[1]):
    plt.figure()
    name_fig = sanity_plotting_directory + "/image_data_og_{0}".format(image_index)
    plt.imshow(batch_data[:,image_index])
    plt.savefig(name_fig)

#Getting the 5 lowest values
N = 5
for key in histogram_dict.keys():
    arr = np.array(histogram_dict[key])
    lowest_values = arr.argsort()[:N]
    len_values = lowest_values.shape[0]
    #Now get those indices on the values for the a_matrices
    print("Should see the same length here, num values in E dict at key {0} is {1} and num values in a-dict at key {0} is {2}".format(key, len(histogram_dict[key].values()), len(a_vector_dict[key].values())))
    for i in range(len_values):
        print()



##########
#Need to instantiate sparse coding class for comparison with LAHMC.
#Note that the plot directory is negligible since no plotting is occuring in the
#below code.
plot_directory = 'garbage'
sess = tf.Session()
our_class = TensorSparse(num_receptive_fields = num_receptive_fields, size_of_patch = size_of_patch, session_object = sess, batch_size = batch_size, lambda_parameter = lambda_parameter, LR = LR, plot_directory = plot_directory, phis = phis)
our_class.load_data(batch_data)
our_class.infer_a_coefficients()
bruno_energy = our_class.energy_value_for_each_sample()
print("Energy from original sparse coding model for these phis is", bruno_energy)
bruno_a_matrix = our_class.sess.run(our_class.a_matr)
#######################################
