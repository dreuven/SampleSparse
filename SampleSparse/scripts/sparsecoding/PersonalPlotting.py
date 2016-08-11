import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
class PPlotting:
    root_directory = None
    def __init__(self, directory):
        # try:
        #     str(directory)
        # except:
        #     print("Cannot convert input to string. Put in a name!")
        self.root_directory = str(directory)
    def plot_a_matrix_mean(self,a_mean_matrix):
        savepath = self.root_directory + "/a_mean_plot.png"
        # print("SAVE PATH IS {0}".format(savepath))
        plt.figure()
        plt.title("A matrix mean")
        plt.plot(a_mean_matrix)
        plt.savefig(savepath)
        plt.close()
        
    def plot_a_activitation(self,a_coeff_matrix, i, num_receptive_fields, num_patches_from_image):
        # num_receptive_fields = a_coeff_matrix.shape[0]
        # num_patches_from_image = a_coeff_matrix.shape[1]
        a_to_plot = a_coeff_matrix.reshape(num_receptive_fields, num_patches_from_image)[:,0]
        plt.figure()
        plt.title("Activity of a for an image patch graph")
        plt.plot(a_to_plot)
        savepath = self.root_directory + "/a_activity_iter:" + str(i) + ".png"
        plt.savefig(savepath)
        plt.close()
        
    def plot_phis(self,phis, i):
        plt.figure(figsize=(10,10))
        k = phis.shape[1]
        val_x_y = int(np.ceil(np.sqrt(k)))
        size_of_patch = int(np.sqrt(phis.shape[0]))
        plt.imshow(self.tile_raster_images(phis.T,(size_of_patch, size_of_patch), [val_x_y,val_x_y]), cmap = cm.Greys_r, interpolation="nearest")
        savepath = self.root_directory +'/PhiPlots_iter:' + str(i) + '.png'
        plt.title("Receptive fields")
        plt.savefig(savepath, format='png', dpi=500)
        plt.close()
    def plot_energy_over_time(self,energy_values):
        plt.figure()
        plt.title("Energy Value")
        plt.plot(energy_values)
        savepath = self.root_directory +"/EnergyVals.png"
        plt.savefig(savepath)
        plt.close()
    def plot_reconstruction_error_over_time(self,reconstruction_error_arr):
        plt.figure()
        plt.title("Reconstruction Error Over Time")
        plt.plot(reconstruction_error_arr)
        savepath = self.root_directory +"/ReconstructionError.png"
        plt.savefig(savepath)
        plt.close()
    def plot_input_data(self,image_patch_data, i):
    #     Note: Assuming image_patch_data is p x N matrix
        size = np.sqrt(image_patch_data.shape[0])
        num_images = int(np.ceil(np.sqrt(image_patch_data.shape[1])))
        im_arr = self.tile_raster_images(image_patch_data.T,[size ,size ],[num_images,num_images])
        savePath = self.root_directory +"/input_data_iter_{0}.png".format(i)
        plt.title("Input Data")
        plt.imshow(im_arr, cmap=cm.Greys_r)
        plt.savefig(savePath)
        plt.close()
    def plot_reconstructions(self,reconstruction,i):
    #     Note: assuming reconstruction is p x N matrix
        size = np.sqrt(reconstruction.shape[0])
        num_images = int(np.ceil(np.sqrt(reconstruction.shape[1])))
        im_arr = self.tile_raster_images(reconstruction.T,[size ,size],[num_images,num_images])
        savePath = self.root_directory +"/reconstructions_iter_{0}.png".format(i)
        plt.title("Reconstruction Data")
        plt.imshow(im_arr, cmap=cm.Greys_r)
        plt.savefig(savePath)
        plt.close()
    def create_and_show_receptive_field_poster(self,receptive_fields, size_space_between, num_per_row, num_per_column, iteration_num):
        num_receptive_fields = receptive_fields.shape[1]
    #     Making assumption that all receptive fields are square!
        size_receptive = int(np.sqrt(receptive_fields.shape[0]))
        if num_receptive_fields > num_per_row * num_per_column:
            print("Impossible to fit all receptive fields onto this poster")
            return
        size_row_of_poster = num_per_row * size_receptive + (num_per_row - 1) * size_space_between
        size_col_of_poster = num_per_column * size_receptive + (num_per_column - 1) * size_space_between
        poster_image = np.zeros((size_row_of_poster, size_col_of_poster))
        row_index = 0
        col_index = 0
        for r_field in range(num_receptive_fields):
            curr_receptive_field = receptive_fields[:,r_field].reshape(size_receptive, size_receptive)
            poster_image[row_index:row_index + size_receptive, col_index: col_index + size_receptive] = curr_receptive_field
            col_index = col_index + size_receptive + size_space_between
            if col_index - size_space_between == size_col_of_poster:
                col_index = 0
                row_index = row_index + size_receptive + size_space_between
        
        plt.imshow(poster_image, cmap=cm.Greys_r)
        savepath = self.root_directory +'/PhiPlots_iter:' + str(iteration_num) + '.png'
        plt.title("Receptive fields")
        plt.savefig(savepath)
        plt.close()

    def scale_to_unit_interval(self,ndar, eps=1e-8):
    #   """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = ndar.copy()
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar


    def tile_raster_images(self,X, img_shape, tile_shape, tile_spacing=(2, 2),
                           scale_rows_to_unit_interval=True,
                           output_pixel_vals=True):
        """
      Transform an array with one flattened image per row, into an array in
      which images are reshaped and layed out like tiles on a floor.
      This function is useful for visualizing datasets whose rows are images,
      and also columns of matrices for transforming those rows
      (such as the first layer of a neural net).
      :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
      be 2-D ndarrays or None;
      :param X: a 2-D array in which every row is a flattened image.
      :type img_shape: tuple; (height, width)
      :param img_shape: the original shape of each image
      :type tile_shape: tuple; (rows, cols)
      :param tile_shape: the number of images to tile (rows, cols)
      :param output_pixel_vals: if output should be pixel values (i.e. int8
      values) or floats
      :param scale_rows_to_unit_interval: if the values need to be scaled before
      being plotted to [0,1] or not
      :returns: array suitable for viewing as an image.
      (See:`Image.fromarray`.)
      :rtype: a 2-d array with same dtype as X.
      """
        assert len(img_shape) == 2
        assert len(tile_shape) == 2
        assert len(tile_spacing) == 2

    #   The expression below can be re-written in a more C style as
    #   follows :
      
    #   out_shape = [0,0]
    #   out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                  tile_spacing[0]
    #   out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                  tile_spacing[1]
        out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                          in zip(img_shape, tile_shape, tile_spacing)]

        if isinstance(X, tuple):
            assert len(X) == 4
        #       Create an output np ndarray to store the image
            if output_pixel_vals:
                out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
            else:
                out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #       colors default to 0, alpha defaults to 1 (opaque)
            if output_pixel_vals:
                channel_defaults = [0, 0, 0, 255]
            else:
                channel_defaults = [0., 0., 0., 1.]

            for i in range(4):
                if X[i] is None:
        #               if channel is None, fill it with zeros of the correct
                      # dtype
                    out_array[:, :, i] = np.zeros(out_shape,
                              dtype='uint8' if output_pixel_vals else out_array.dtype
                              ) + channel_defaults[i]
                else:
        #               use a recurrent call to compute the channel and store it
                      # in the output
                    out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
            return out_array

        else:
        #       if we are dealing with only one channel
            H, W = img_shape
            Hs, Ws = tile_spacing

        #       generate a matrix to store the output
            out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


            for tile_row in range(tile_shape[0]):
                for tile_col in range(tile_shape[1]):
                    if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                        if scale_rows_to_unit_interval:
                              # if we should scale values to be between 0 and 1
        #                       do this by calling the `scale_to_unit_interval`
                              # function
                            this_img = self.scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                        else:
                            this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
        #                   add the slice to the corresponding position in the
                          # output array
                        out_array[
                            tile_row * (H+Hs): tile_row * (H + Hs) + H,
                            tile_col * (W+Ws): tile_col * (W + Ws) + W
                            ] \
                            = this_img * (255 if output_pixel_vals else 1)
            return out_array