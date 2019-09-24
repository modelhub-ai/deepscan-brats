import json
import torch
from processing import ImageProcessor
from modelhublib.model import ModelBase
import model
from model import UNET_3D_to_2D
from model import load_checkpoint

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
        #inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        #TODO for loading: segment needs AT LEAST the affine and the header of the flair modality
        print('Finished preprocessing')
        output = segment(input["flair"]["fileurl"], input["t1"]["fileurl"], input["t2"]["fileurl"], input["t1c"]["fileurl"], self._model1 , self._model2)
        output = self._imageProcessor.computeOutput(output)
        return output
