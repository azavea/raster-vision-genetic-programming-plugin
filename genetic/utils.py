import functools
import numpy as np
import os
from os.path import join
import rasterio
import zipfile

from rastervision.utils.files import list_paths


def zipdir(dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as ziph:
        for root, dirs, files in os.walk(dir):
            for file in files:
                ziph.write(join(root, file),
                           join('/'.join(dirs),
                                os.path.basename(file)))


def apply_to_raster(func, input_pix, shape):
    input_pixels = input_pix.T.astype('float64')
    output = np.apply_along_axis(lambda arr: func(*arr),
                                 2, input_pixels).reshape(shape)
    return output


@functools.lru_cache(None)
def read_input_truth(input_file, truth_file):
    with rasterio.open(input_file, 'r') as input_ds:
        with rasterio.open(truth_file, 'r') as truth_ds:
            input_pixels = input_ds.read()
            truth_pixels = truth_ds.read()

            return (input_pixels, truth_pixels)


def fitness(data_img_dir, truth_img_dir, compiler, individual):
    """
    Return a score representing the fitness of a particular program.

    Params individual: The individual to be evaluated.  train_files: A list of tuples of the
    form (raster, geojson), where raster is a filename of a GeoTIFF containing multiband raster
    data, and geojson is the name of a GeoJSON file representing ground-truth.
    """
    # We will assume for the time being that list_paths() returns in a consistent order, because
    # it seems to.
    eval_data = zip(list_paths(data_img_dir), list_paths(truth_img_dir))

    total_error = 0
    func = compiler(expr=individual)
    for input_file, truth_file in eval_data:
        # Load truth data
        input_pixels, truth_pixels = read_input_truth(input_file, truth_file)
        output = apply_to_raster(func, input_pixels, truth_pixels.shape)
        errors = output - truth_pixels
        #total_error += np.sum(np.square(errors))
        # Return mean squared error
        total_error += np.mean(np.square(errors))
    return (total_error,)
