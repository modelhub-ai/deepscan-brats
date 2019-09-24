import nibabel as nib
import numpy as np

from modelhublib.imageconverters import ImageConverter


class NibToNumpyConverter(ImageConverter):
    """
    Converts SimpltITK.Image objects to Numpy
    """

    def _convert(self, image):
        """
        Args:
            image (SimpleITK.Image): Image object to convert.
        
        Returns:
            Input image object converted to numpy array with 4 dimensions [batchsize, z/color, height, width]
        
        Raises:
            IOError if input is not of type SimpleITK.Image or cannot be converted for other reasons.
        """
        if isinstance(image, nib.nifti1.Nifti1Image):
            return self.__convertToNumpy(image)
        else:
            raise IOError("Image is not of type \"SimpleITK.Image\".")
    

    def __convertToNumpy(self, image):
        # this returns the raw image object!!
        return image

