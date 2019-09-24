import json
import torch
from processing import ImageProcessor
from modelhublib.model import ModelBase
import model
from model import UNET_3D_to_2D
from model import load_checkpoint
from DeepSCAN_BRATS import segment

class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL model (change this if you are not using ONNX)
        target_label_names = ['necrosis', 'contrast_enhancing', 'core', 'tumor', 'brain']
        net1 = UNET_3D_to_2D(0,channels_in=4,channels=128, growth_rate =12, dilated_layers=[6,6,6,6], output_channels=len(target_label_names))
        net2 = UNET_3D_to_2D(1,channels_in=4,channels=128, growth_rate =12, dilated_layers=[6,6,6], output_channels=len(target_label_names))
        net1 = net1.cuda()
        net2 = net2.cuda()
        load_checkpoint(net1, 'model/checkpoint.pth.tar')
        load_checkpoint(net2, 'model/checkpoint_2.pth.tar')
        self._model1 = net1
        self._model2 = net2


    def infer(self, input):
        # load preprocessed input
        # CAUTION: these custum functions return the complete nib.Nifti1Image! 
        t1 = self._imageProcessor.loadAndPreprocess(input["t1"]["fileurl"], id="t1")
        t1c = self._imageProcessor.loadAndPreprocess(input["t1c"]["fileurl"], id="t1c")
        t2 = self._imageProcessor.loadAndPreprocess(input["t2"]["fileurl"], id="t2")
        flair = self._imageProcessor.loadAndPreprocess(input["flair"]["fileurl"], id="flair")
        output = segment(flair, t1, t2, t1c, self._model1 , self._model2)
        # compute output
        output = self._imageProcessor.computeOutput(output)
        return output
