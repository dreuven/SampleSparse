import numpy as np
from LAHMC.python import LAHMC
import ipdb
import numpy.matlib
class numpy_sampler:
    def __init__(self, num_receptive_fields, size_of_patch, batch_size, lambda_parameter, num_particles_per_batch, LR = 1e-1):
        self.num_receptive_fields = num_receptive_fields
        self.size_of_patch = size_of_patch
        self.batch_size = batch_size
        self.lambda_parameter = lambda_parameter
        self.LR = LR
        self.phis = np.random.randn(size_of_patch**2, num_receptive_fields)
        self.batch_data = None
        self.sampling_results = None
        self.num_particles_per_batch = num_particles_per_batch
    def load_batch(self,images):
        #Maybe I want to normalize... I dont think it matters for this part of sampling since no update of phis.
        print("In load data")
        print("Original size of images is", images.shape)
        copied_data = np.matlib.repmat(images,1, self.num_particles_per_batch)
        print("Copied size of the data is", copied_data.shape)
        self.batch_data = copied_data
        print("Finished Loading images.")
    def E(self,a, sigma = 1.):
        recon_error = self.batch_data - self.phis.dot(a)
        tmp = recon_error**2/(2*sigma**2)
        tmp = tmp.sum(axis = 0).mean()
        sparsity = self.lambda_parameter * np.abs(a).sum(axis = 0).mean()
        return tmp + sparsity
    def gradient(self,a, sigma = 1.):
        print("Dimensions of _a_ input to gradient are", a.shape)
        if a.shape[1] != self.batch_size * self.num_particles_per_batch:
            ipdb.set_trace()
        gradient = -2*((self.batch_data - self.phis.dot(a)).T.dot(self.phis)).T + self.lambda_parameter * np.sign(a)
        return gradient
    def sample(self,num_steps = 100):
        Ainit = np.random.randn(self.num_receptive_fields,self.batch_size * self.num_particles_per_batch)
        sampler = LAHMC.LAHMC(Ainit,self.E,self.gradient)
        A_final = sampler.sample(num_steps)
        print("A final shape is", A_final.shape)
        self.sampling_results = A_final
        print("Finished Sampling!")
    def ret_sample_energies(self):
        energy_values = self.E(self.sampling_results)
        return energy_values
