from modelhublib.processor import ImageProcessorBase
from modelhublib.imageloaders import PilImageLoader, SitkImageLoader, NumpyImageLoader
from modelhublib.imageconverters import PilToNumpyConverter, SitkToNumpyConverter, NumpyToNumpyConverter
from niftiImageLoader import NiftiImageLoader
from nibImageConverter import NibToNumpyConverter

import nibabel as nib
import PIL
import SimpleITK
import numpy as np
import json


class ImageProcessor(ImageProcessorBase):

    def __init__(self, config):
        self._config = config
        self._imageLoader = NiftiImageLoader(self._config)
        self._imageToNumpyConverter = NibToNumpyConverter()

    # OPTIONAL: Use this method to preprocess images using the image objects
    #           they've been loaded into automatically.
    #           You can skip this and just perform the preprocessing after
    #           the input image has been convertet to a numpy array (see below).
    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            # TODO: implement preprocessing of PIL image objects
            pass
        elif isinstance(image, SimpleITK.Image):
            # TODO: implement preprocessing of SimpleITK image objects
            pass
        elif isinstance(image, np.ndarray):
            pass
        elif isinstance(image, nib.nifti1.Nifti1Image):
            pass
        else:
            raise IOError("Image Type not supported for preprocessing.")
        return image


    def _preprocessAfterConversionToNumpy(self, npArr):
        # TODO: implement preprocessing of image after it was converted to a numpy array
        return npArr


    def computeOutput(self, inferenceResults):
        # TODO: implement postprocessing of inference results
        return inferenceResults

