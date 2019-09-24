import os
import cv2
import numpy as np
import numpy
import pandas as pd
import nibabel as nib
import torch.optim as optim
import random
import torch
from torch.nn import BCELoss, NLLLoss, BCEWithLogitsLoss, MSELoss, ModuleList, ReplicationPad2d
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data.sampler import SequentialSampler
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, RandomSampler, WeightedRandomSampler
from torch._six import int_classes as _int_classes



data_shape = 192*192
target_label_sets = [[1],
                     [4],
                     [1,3,4],
                     [1,2,3,4],
                     [1,2,3,4,5]
                    ]
target_label_names = ['necrosis', 'contrast_enhancing', 'core', 'tumor', 'brain']

label_prevalences = [0.1]*2+[0.2]+[0.4]+[0.9]

list(zip(label_prevalences,target_label_names))




np.random.seed(13375) # for reproducibility
random.seed(133567)


def rotateImage(image, angle, interp = cv2.INTER_NEAREST):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=interp)
  return result


def get_stack(axis,
              volume,
              gt_volume,
              central_slice,
              first_slice=None,
              last_slice=None,
              stack_depth=5,
              size = (192,192),
              rotate_angle = None,
              rotate_axis = 0,
              flipLR = False,
             lower_threshold = None,
             upper_threshold= None):
    if (first_slice is not None or last_slice is not None) and central_slice is not None:
        raise ValueError('Stack location overspecified')
    if (first_slice is not None and last_slice is not None) and stack_depth is not None:
        raise ValueError('Stack location overspecified')
    image_data = volume
    if flipLR:
        image_data = image_data[:,:,::-1]
    if rotate_angle is not None:
        image_data = np.swapaxes(rotateImage(np.swapaxes(image_data,0,rotate_axis),rotate_angle,cv2.INTER_LINEAR)
                                 ,0,rotate_axis)
    #image_data = np.swapaxes(volume, 0, axis)
    image_data = np.array(np.swapaxes(image_data, 0, axis), copy = True)

    mean = np.mean(image_data[image_data>0])
    sd = np.sqrt(np.var(image_data[image_data>0]))


    if lower_threshold is None:
        lower_threshold = 0

    if upper_threshold is None:
        upper_threshold = np.percentile(image_data[image_data>0], 99.9)



    image_data[image_data<lower_threshold]=lower_threshold
    image_data[image_data>upper_threshold]=upper_threshold



    if first_slice is None:
        if central_slice is not None:
            first_slice = central_slice - stack_depth//2
            last_slice = central_slice + stack_depth//2 + 1
        elif last_slice is not None:
            first_slice = last_slice - stack_depth
    elif last_slice is None:
            last_slice = min(first_slice + stack_depth, len(image_data))
    pad_up = max(0, -first_slice)

    pad_down = -min(0, len(image_data)-last_slice)

    first_slice = max(first_slice,0)
    last_slice = min(last_slice, len(image_data))
    initial_stack = image_data[first_slice:last_slice]
    initial_shape = initial_stack.shape[1:]
    shape_difference = (size[0] - initial_shape[0],size[1] - initial_shape[1])
    pad_size = ((pad_up,pad_down),
                (shape_difference[0]//2, shape_difference[0] - shape_difference[0]//2),
                (shape_difference[1]//2, shape_difference[1] - shape_difference[1]//2) )
    initial_stack = np.pad(initial_stack, pad_size, mode = 'constant', constant_values = lower_threshold)


    nonzero_mask = (initial_stack>lower_threshold).astype(np.int)

    gt = gt_volume

    if flipLR:
        gt = gt[:,:,::-1]
    if rotate_angle is not None:
        gt = np.swapaxes(rotateImage(np.swapaxes(gt,0,rotate_axis),rotate_angle),0,rotate_axis)

    gt  = np.swapaxes(gt, 0, axis)

    gt_stack = gt[first_slice:last_slice]



    gt_stack = np.pad(gt_stack, pad_size, mode = 'constant', constant_values = 0)
    return (initial_stack - mean)/sd, gt_stack, nonzero_mask



def get_stack_no_augment(axis, volume, gt_volume, first_slice, last_slice,size=(192,192)):
    return get_stack(axis = axis,
                     volume = volume,
                     gt_volume = gt_volume,
                     central_slice=None,
                     stack_depth=None,
                     first_slice = first_slice,
                     last_slice = last_slice,
                    size=size)





# ## Define the U-net variant
class GradMultiplier(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_multiply(x, lambd):
    return GradMultiplier(lambd)(x)

def reduce_3d_depth (in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad3d((1,1,1,1,0,0))),
            ("conv1", nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.BatchNorm2d(out_channel)),
            ("relu1", nn.ReLU())
    ]))
    return layer

def down_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.BatchNorm2d(out_channel)),
            ("relu1", nn.ReLU()),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.BatchNorm2d(out_channel)),
            ("relu1", nn.ReLU())]))
    return layer

def up_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.BatchNorm2d(out_channel)),
            ("relu1", nn.ReLU()),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.BatchNorm2d(out_channel)),
            ("relu2", nn.ReLU())]))
    return layer

class DilatedDenseUnit(nn.Module):
    def __init__(self, in_channel, growth_rate , kernel_size, dilation):
        super(DilatedDenseUnit,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.BatchNorm2d(in_channel)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(dilation)),
            ("conv1", nn.Conv2d(in_channel, growth_rate, kernel_size=kernel_size, dilation = dilation,padding=0))]))
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = concatenate(x, out)
        return out


def center_crop(layer, target_size):
    _, _, layer_width, layer_height = layer.size()
    start = (layer_width - target_size) // 2
    crop = layer[:, :, start:(start + target_size), start:(start + target_size)]
    return crop

def concatenate(link, layer):
    crop = center_crop(link, layer.size()[2])
    concat = torch.cat([crop, layer], 1)
    return concat

def dense_atrous_bottleneck(in_channel, growth_rate = 12, depth = [4,4,4,4]):
    layer_dict = OrderedDict()
    for idx, growth_steps in enumerate(depth):
        dilation_rate = 2**idx
        for y in range(growth_steps):
            layer_dict["dilated_{}_{}".format(dilation_rate,y)] = DilatedDenseUnit(in_channel,
                                                                        growth_rate,
                                                                        kernel_size=3,
                                                                        dilation = dilation_rate)
            in_channel = in_channel + growth_rate
    return nn.Sequential(layer_dict), in_channel




class UNET_3D_to_2D(nn.Module):
    def __init__(self, depth, channels_in = 1,
                 channels_2d_to_3d=32, channels=32, output_channels = 1, slices=5,
                 variance_gradient_multiplier = 1,
                 dilated_layers = [4,4,4,4],
                growth_rate = 12):
        super(UNET_3D_to_2D, self).__init__()
        self.main_modules = []

        self.depth = depth
        self.slices = slices
        self.variance_gradient_multiplier = variance_gradient_multiplier

        self.depth_reducing_layers = ModuleList([reduce_3d_depth(in_channel, channels_2d_to_3d, kernel_size=3, padding=0)
                                                 for in_channel in [channels_in]+[channels_2d_to_3d]*(slices//2 - 1)])


        self.down1 = down_layer(in_channel=channels_2d_to_3d, out_channel=channels, kernel_size=3, padding=0)
        self.main_modules.append(self.down1)
        self.max1 = nn.MaxPool2d(2)
        self.down_layers = ModuleList([down_layer(in_channel = channels*(2**i),
                                  out_channel = channels * (2**(i+1)),
                                  kernel_size = 3,
                                  padding=0
                                 ) for i in range(self.depth)])
        self.main_modules.append(self.down_layers)
        self.max_layers = ModuleList([nn.MaxPool2d(2) for i in range(self.depth)])

        self.bottleneck, bottleneck_features  = dense_atrous_bottleneck(channels*2**self.depth, growth_rate = growth_rate,
                                                                       depth = dilated_layers)
        self.main_modules.append(self.bottleneck)

        self.upsampling_layers = ModuleList([nn.Sequential(OrderedDict([
                ("upsampling",nn.Upsample(scale_factor=2, mode = 'bilinear')),
                ("pad", nn.ReplicationPad2d(1)),
                ("conv", nn.Conv2d(in_channels= bottleneck_features,
                                   out_channels=bottleneck_features,
                                   kernel_size=3,
                                   padding=0))]))  for i in range(self.depth, -1, -1)])
        self.main_modules.append(self.upsampling_layers)
        self.up_layers = ModuleList([up_layer(in_channel= bottleneck_features+ channels*(2**(i)),
                                   out_channel=bottleneck_features,
                                   kernel_size=3,
                                   padding=0) for i in range(self.depth, -1, -1)])

        self.main_modules.append(self.up_layers)
        self.last = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)
        self.main_modules.append(self.last)

        self.logvar = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)

    def forward(self, x):


        # down

        out = x

        for i in range(self.slices//2):
            out = self.depth_reducing_layers[i](out)

        out.transpose_(1, 2).contiguous()
        size = out.size()
        out = out.view((-1, size[2], size[3], size[4]))

        links = []
        out = self.down1(out)
        links.append(out)
        out = self.max1(out)

        for i in range(self.depth):
            out = self.down_layers[i](out)
            links.append(out)
            out = self.max_layers[i](out)





        out = self.bottleneck(out)


        links.reverse()

        # up

        for i in range(self.depth+1):

            out = self.upsampling_layers[i](out)

            out = concatenate(links[i], out)
            out = self.up_layers[i](out)

        pred = self.last(out)
        logvar = self.logvar(out)

        logvar = grad_multiply(logvar,self.variance_gradient_multiplier)


        return pred, logvar


class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count



def torch_dice(x, y):
    epsilon = 1e-4
    intersection = torch.mul(x,y)
    intersection_sum = intersection.sum(1).sum(1)+epsilon/2
    sum_true = x.sum(1).sum(1) + epsilon
    sum_pred = y.sum(1).sum(1)
    return 2*intersection_sum/(sum_pred+sum_true)

def torch_dice_local(x, y, radius):
    epsilon = 1e-4
    intersection = torch.mul(x,y)
    local_mean = nn.Conv2d(1, 1, kernel_size=radius, padding=radius//2, bias=False)
    local_mean.weight = torch.nn.Parameter(torch.ones(1,1,3,3).cuda()/(radius*radius))
    local_intersection = local_mean(intersection.unsqueeze(1)) + epsilon
    local_true = local_mean(x.unsqueeze(1)) + epsilon
    local_pred = local_mean(y.unsqueeze(1))
    return 2*local_intersection/(local_pred+local_true)

num_epochs = 3

class BCE_from_logits(nn.modules.Module):
    def __init__(self):
        super(BCE_from_logits,self).__init__()
    def forward(self, input, target):
        #input = input.clamp(min = -1, max = 1)
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        return loss

class BCE_from_logits_focal(nn.modules.Module):
    def __init__(self, gamma):
        super(BCE_from_logits_focal,self).__init__()
        self.gamma = gamma
    def forward(self, input, target):
        #input = input.clamp(min = -1, max = 1)
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        p = input.sigmoid()
        pt = (1-p)*(1-target) + p*target
        return ((1-pt).pow(self.gamma))*loss


val_stack_size = 50

def heteroscedastic_loss(masks, outputs, logit_flip, nonzero_mask, gamma):
    criterion = BCE_from_logits()
    criterion2 = BCE_from_logits_focal(gamma)
    flip_prob = F.sigmoid(logit_flip)

    masks_flipped = (1- masks)*flip_prob + masks * (1 - flip_prob)
    false_neg = ((-outputs).sign().clamp(min=0))*masks
    false_pos = outputs.sign().clamp(min=0)*(1-masks)
    label_is_flipped = false_neg+false_pos
    flipped_gt = nonzero_mask*label_is_flipped
    loss =  criterion2(outputs, masks) + criterion2(outputs, masks_flipped) + criterion2(logit_flip, flipped_gt)
    loss = loss*nonzero_mask
    loss = loss.mean()
    return loss

def apply_to_case(model, volumes, gt_volume, batch_size, variance_estimator = 'analytic', axis=0,
                 size=(256,256)):
    model.eval()
    volume_0, volume_1, volume_2, volume_3= volumes
    print(volume_0.shape)

    mask_total = []
    logit_total = []
    image_total = []
    flip_total = []
    loss_total = [0 for i in target_label_sets]
    softmax_loss = 0

    TP = np.zeros(len(target_label_sets))
    FP = np.zeros(len(target_label_sets))
    TN = np.zeros(len(target_label_sets))
    FN = np.zeros(len(target_label_sets))

    num_batches = volume_0.shape[axis]//(batch_size)
    if volume_0.shape[axis]%batch_size > 0:
        num_batches = num_batches + 1


    class BrainDataTest(Dataset):
        def __init__(self):
            self.length = num_batches
        # Override to give PyTorch access to any image on the dataset
        def __getitem__(self, batch):

            first_slice = batch*batch_size - 2
            last_slice = np.min([(batch+1)*batch_size+2, volume_0.shape[axis]+2])

            extra_upper_slices = np.max([0, 5 - (last_slice - first_slice)])

            last_slice = last_slice + extra_upper_slices

            images_t1, masks, nonzero_masks = get_stack_no_augment(axis = axis,
                                                            volume = volume_1,
                                                            gt_volume = gt_volume,
                                                            first_slice=first_slice,
                                                            last_slice=last_slice,
                                                           size=size)

            images_t1ce, masks, nonzero_masks = get_stack_no_augment(axis = axis,
                                                            volume = volume_3,
                                                            gt_volume = gt_volume,
                                                            first_slice=first_slice,
                                                            last_slice=last_slice,
                                                           size=size)



            images_t2, masks, nonzero_masks = get_stack_no_augment(axis = axis,
                                                            volume = volume_2,
                                                            gt_volume = gt_volume,
                                                            first_slice=first_slice,
                                                            last_slice=last_slice,
                                                           size=size)

            images_flair, masks, nonzero_masks = get_stack_no_augment(axis = axis,
                                                            volume = volume_0,
                                                            gt_volume = gt_volume,
                                                            first_slice=first_slice,
                                                            last_slice=last_slice,
                                                           size=size)


            images = np.stack([images_flair, images_t1, images_t2, images_t1ce])

            masks = numpy.stack([numpy.isin(masks[2:-(2+extra_upper_slices)],labelset) for labelset in target_label_sets], axis =1)
            masks = masks.astype(np.float)
            bg = numpy.logical_not(numpy.isin(masks, [y for x in target_label_sets for y in x]))

            nonzero_masks = nonzero_masks[2:-(2+extra_upper_slices)]

            return images.astype(np.float32), masks.astype(np.float32), bg.astype(np.float32), nonzero_masks.astype(np.float32)

        # Override to give PyTorch size of dataset
        def __len__(self):
            return self.length

    test_generator = DataLoader(BrainDataTest(), sampler = SequentialSampler(BrainDataTest()),
                     num_workers=0,pin_memory=True)

    for images, masks, bg, nonzero_masks  in test_generator:

        images = Variable(images, volatile=True).cuda()
        masks = Variable(masks, volatile=True).cuda()[0]
        bg = Variable(bg, volatile=True).cuda()[0]
        nonzero_mask = Variable(nonzero_masks, volatile =True).cuda()

        mask_total.append(masks.data.cpu().numpy())
        image_total.append(images.data.cpu().numpy()[0,0,2:-2])

        outputs, logit_flip = model(images)


        outputs = outputs * torch.unsqueeze(nonzero_mask[0],1)

        logit_flip = logit_flip * torch.unsqueeze(nonzero_mask[0],1)


        flip_prob = F.sigmoid(logit_flip)

        for idx, x in enumerate(target_label_sets):
            if variance_estimator == 'analytic':
                loss = heteroscedastic_loss(masks[0,idx],
                                            outputs[:,idx],
                                            logit_flip[:,idx],
                                            nonzero_mask,
                                           2).mean()
            else:
                loss = (BCE_from_logits_focal(2)(outputs[:,idx], masks[:,idx])*nonzero_mask).mean()
            loss_total[idx] = loss_total[idx]+loss.data.cpu().numpy()[0]

        mask_cpu = masks.data.cpu().numpy()

        outputs_cpu = outputs.cpu().data.numpy()

        TP_batch = np.sum(np.logical_and(outputs_cpu>0, mask_cpu>0), axis = (0,2,3))

        FP_batch = np.sum(np.logical_and(outputs_cpu>0, mask_cpu<=0), axis = (0,2,3))

        TN_batch = np.sum(np.logical_and(outputs_cpu<0, mask_cpu<=0), axis = (0,2,3))

        FN_batch = np.sum(np.logical_and(outputs_cpu<0, mask_cpu>0), axis = (0,2,3))

        TP = TP + TP_batch
        FP = FP + FP_batch
        TN = TN + TN_batch
        FN = FN + FN_batch
        background = torch.zeros_like(outputs[:,0:1])

        softmax_outputs = F.log_softmax(torch.cat([background, outputs],dim=1), dim = 1)

        softmax_masks = torch.cat([bg,masks], dim = 1)
        _, softmax_masks = torch.max(softmax_masks, 1)

        this_softmax_loss = (nn.NLLLoss2d(reduce=False)(softmax_outputs, softmax_masks)*nonzero_mask).mean()

        softmax_loss = softmax_loss + this_softmax_loss.data.cpu().numpy()[0]




        logit_total.append(outputs.cpu().data.numpy())
        flip_total.append(logit_flip.cpu().data.numpy())

        #loss_total = loss_total + loss.data.cpu().numpy()[0]

        #dice_loss = 1 - torch.mean(torch_dice(outputs_prob.data, masks.data.float()))

        #loss = loss + dice_weight*dice_loss

        #dice_loss = torch.mean(torch_dice(outputs.data, masks.data.float()))
        #local_dice_loss = torch.mean(torch_dice_local(outputs, masks.float(), 7))
        #vloss = vloss +logvar.mean()/2 #+(1-local_dice_loss)


        #loss = loss + 0.00001*(net.logvar.weight**2).sum()

        #val_dice.update(dice_metric,  images.size(0))
        #val_loss.update(vloss.data[0], images.size(0))
    full_image = np.concatenate(image_total)
    full_mask = np.concatenate(mask_total)
    full_logit = np.concatenate(logit_total)
    full_flip = np.concatenate(flip_total)

    loss_total = [x / num_batches for x in loss_total]

    softmax_loss = softmax_loss/num_batches


    print(full_mask.shape)

    new_shape = full_image.shape

    shape_difference = (full_image.shape[0] - np.swapaxes(volume_0,0, axis).shape[0],
                        full_image.shape[1]-np.swapaxes(volume_0,0, axis).shape[1],
                        full_image.shape[2]-np.swapaxes(volume_0,0, axis).shape[2])

    print(shape_difference)

    full_image = full_image[shape_difference[0]//2:new_shape[0]- (shape_difference[0] - shape_difference[0]//2),
                           shape_difference[1]//2: new_shape[1]- (shape_difference[1] - shape_difference[1]//2),
                           shape_difference[2]//2: new_shape[2]- (shape_difference[2] - shape_difference[2]//2)]
    full_image = np.swapaxes(full_image, 0, axis)
    full_mask = full_mask[:,shape_difference[0]//2:new_shape[0]- (shape_difference[0] - shape_difference[0]//2),
                           shape_difference[1]//2: new_shape[1]- (shape_difference[1] - shape_difference[1]//2),
                           shape_difference[2]//2: new_shape[2]- (shape_difference[2] - shape_difference[2]//2)]
    full_mask = np.swapaxes(full_mask, 1, 0)
    full_mask = np.swapaxes(full_mask, 1, axis+1)
    full_logit = full_logit[:, shape_difference[0]//2:new_shape[0]- (shape_difference[0] - shape_difference[0]//2),
                           shape_difference[1]//2: new_shape[1]- (shape_difference[1] - shape_difference[1]//2),
                           shape_difference[2]//2: new_shape[2]- (shape_difference[2] - shape_difference[2]//2)]
    full_logit = np.swapaxes(full_logit, 1, 0)
    full_logit = np.swapaxes(full_logit, 1, axis+1)
    full_flip = full_flip[:, shape_difference[0]//2:new_shape[0]- (shape_difference[0] - shape_difference[0]//2),
                           shape_difference[1]//2: new_shape[1]- (shape_difference[1] - shape_difference[1]//2),
                           shape_difference[2]//2: new_shape[2]- (shape_difference[2] - shape_difference[2]//2)]
    full_flip = np.swapaxes(full_flip, 1, 0)
    full_flip = np.swapaxes(full_flip, 1, axis+1)

    return full_image, full_mask, full_logit, full_flip, loss_total, softmax_loss, TP, FP, TN, FN


def collect_bn_modules(module, bn_modules):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        bn_modules.append(module)

def fix_batchnorm(data_sample, swa_model):
    """
    During training, batch norm layers keep track of a running mean and
    variance of the previous layer's activations. Because the parameters
    of the SWA model are computed as the average of other models' parameters,
    the SWA model never sees the training data itself, and therefore has no
    opportunity to compute the correct batch norm statistics. Before performing
    inference with the SWA model, we perform a single pass over the training data
    to calculate an accurate running mean and variance for each batch norm layer.
    """
    bn_modules = []
    swa_model.apply(lambda module: collect_bn_modules(module, bn_modules))

    if not bn_modules: return

    swa_model.train()

    for module in bn_modules:
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

    momenta = [m.momentum for m in bn_modules]

    inputs_seen = 0

    bn_loader = make_sampler(data_sample, label_must_be_present=17, fraction=0.9, num_samples=1000)

    for data in  tqdm(bn_loader, leave=False):
        images, masks, bg, nonzero_masks = data
        images = Variable(images).cuda(async=True)

        batch_size = images.size(0)

        momentum = batch_size / (inputs_seen + batch_size)
        for module in bn_modules:
            module.momentum = momentum

        res = swa_model(images)

        inputs_seen += batch_size

    for module, momentum in zip(bn_modules, momenta):
        module.momentum = momentum


class BrainData(Dataset):
    def __init__(self, datapoints,axes = [0,1,2] ):
        self.axes = axes
        self.length = len(t1_filepaths)
        self.datapoints = datapoints
    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index, stack_depth = 5, size = (192,192)):
        index, image_idx, gt_id, random_flip, rotate_axis, rotate_angle, shift, scale = index
        case, axis, random_slice = self.datapoints[index]

        images_t1, gt, nonzero = get_stack(axis=axis, volume=t1_data[case][image_idx],
                         gt_volume = gt_data[case][gt_id], central_slice=random_slice, stack_depth=stack_depth, size = size,
                                     rotate_angle = rotate_angle, rotate_axis = rotate_axis, flipLR = random_flip,
                         lower_threshold = 0, upper_threshold = t1_99_percent[case][image_idx]
                        )
        images_t2, gt, nonzero = get_stack(axis=axis, volume=t2_data[case][image_idx],
                         gt_volume = gt_data[case][gt_id], central_slice=random_slice, stack_depth=stack_depth, size = size,
                                     rotate_angle = rotate_angle, rotate_axis = rotate_axis, flipLR = random_flip,
                         lower_threshold = 0, upper_threshold = t2_99_percent[case][image_idx]
                        )
        images_flair, gt, nonzero = get_stack(axis=axis, volume=flair_data[case][image_idx],
                         gt_volume = gt_data[case][gt_id], central_slice=random_slice, stack_depth=stack_depth, size = size,
                                     rotate_angle = rotate_angle, rotate_axis = rotate_axis, flipLR = random_flip,
                         lower_threshold = 0, upper_threshold = flair_99_percent[case][image_idx]
                        )
        images_t1ce, gt, nonzero = get_stack(axis=axis, volume=t1ce_data[case][image_idx],
                         gt_volume = gt_data[case][gt_id], central_slice=random_slice, stack_depth=stack_depth, size = size,
                                     rotate_angle = rotate_angle, rotate_axis = rotate_axis, flipLR = random_flip,
                         lower_threshold = 0, upper_threshold = t1ce_99_percent[case][image_idx]
                        )

        images_flair = (images_flair*scale[0])+shift[0]
        images_t1 = (images_t1*scale[1])+shift[1]
        images_t2 = (images_t2*scale[2])+shift[2]
        images_t1ce = (images_t1ce*scale[3])+shift[3]

        images = np.stack([images_flair, images_t1, images_t2, images_t1ce]).astype(np.float32)
        masks = numpy.stack([numpy.isin(gt[2],labelset) for labelset in target_label_sets], axis =0).astype(np.float32)
        bg = numpy.logical_not(numpy.isin(gt[2], [y for x in target_label_sets for y in x]))[np.newaxis].astype(np.float32)
        nonzero_masks = nonzero[2].astype(np.float32)
        return images, masks, bg, nonzero_masks

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.length


class AugmentationSampler(object):
    """Wraps a sampler to yield a mini-batch of multiple indices with data augmentation parameters

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, iterations, drop_last=False):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or                 batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.iterations = iterations

    def __iter__(self):
        batch = []
        for y in range(self.iterations):
            for idx in self.sampler:
                random_masking = np.random.randint(2)
                random_gt = np.random.choice([0, 3,4])
                random_flip = np.random.choice([False, True])
                rotate_axis = np.random.choice([0,1,2,None],p=[0.3,0.3,0.3,0.1])
                shift = np.random.normal(0,0.5, 4)
                scale = np.random.normal(1,0.2, 4)
                if rotate_axis is not None:
                    rotate_angle = np.random.uniform(-15,15)
                else:
                    rotate_angle = None
                batch.append((idx, random_masking, random_gt, random_flip, rotate_axis, rotate_angle, shift, scale))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
                batch = []

    def __len__(self):
        if self.drop_last:
            return (len(self.sampler) // self.batch_size)*self.iterations
        else:
            return ((len(self.sampler) + self.batch_size - 1) // self.batch_size)*self.iterations


def make_sampler(data_sample, label_must_be_present=2, fraction=0.9, num_samples=None):




    datapoints = [[list(itertools.product([case],
                                          [axis],
                                          np.where(gt_nonempty_any[label_must_be_present][case][axis])[0])) for axis in [0,1,2]] for case in data_sample]
    datapoints = [x for  y in datapoints for x in y]
    datapoints = [x for  y in datapoints for x in y]

    weights = [1-fraction]*len(datapoints)

    datapoints_label = [[list(itertools.product([case],
                                          [axis],
                                          np.where(gt_nonempty_any[label_must_be_present][case][axis])[0])) for axis in [0,1,2]] for case in data_sample]
    datapoints_label = [x for  y in datapoints_label for x in y]
    datapoints_label = [x for  y in datapoints_label for x in y]

    for idx, x in enumerate(datapoints):
        if x in datapoints_label:
            weights[idx]=fraction

    if num_samples is None:
        num_samples = len(datapoints_label)


    subsample_loader = DataLoader(BrainData(datapoints), batch_sampler = AugmentationSampler(WeightedRandomSampler(weights, num_samples), batch_size=2, iterations = 1, drop_last=False),
                         num_workers=8,pin_memory=True)

    return subsample_loader


# In[16]:


def load_checkpoint(net, checkpoint_file):
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            net.load_state_dict(checkpoint['state_dict'])


net1 = UNET_3D_to_2D(0,channels_in=4,channels=128, growth_rate =12, dilated_layers=[6,6,6,6], output_channels=len(target_label_names))
net2 = UNET_3D_to_2D(1,channels_in=4,channels=128, growth_rate =12, dilated_layers=[6,6,6], output_channels=len(target_label_names))

net1 = net1.cuda()
net2 = net2.cuda()
load_checkpoint(net1, 'checkpoint.pth.tar')
load_checkpoint(net2, 'checkpoint_2.pth.tar')

flair_filepaths = ["/data/flair.nii.gz"]


t1_filepaths = ["/data/t1.nii.gz"]


t2_filepaths = ["/data/t2.nii.gz"]

t1ce_filepaths = ["/data/t1ce.nii.gz"]

def infer(flair_filepaths, t1_filepaths, t2_filepaths, t1ce_filepaths):

    flair_imgs = [nib.load(x) for x in flair_filepaths]
    flair_data = [x.get_data().astype(np.float32) for x in flair_imgs]

    t1_imgs = [nib.load(x) for x  in t1_filepaths]
    t1_data = [x.get_data().astype(np.float32) for x in t1_imgs]



    t2_imgs = [nib.load(x) for x in  t2_filepaths]
    t2_data = [x.get_data() for x  in t2_imgs]

    t1ce_imgs = [nib.load(x) for x in t1ce_filepaths]
    t1ce_data = [x.get_data() for x in t1ce_imgs]

    case=0

    case_data = [flair_data[case],t1_data[case],t2_data[case],t1ce_data[case]]

    mask = (t2_data[case]>0).astype(np.int)

    full_image_0, full_mask_0, full_logit_0, full_flip_0, losses_0, vloss_0,TP_0,FP_0,TN_0,FN_0  = apply_to_case(net1,
                                                                                                case_data,
                                                              mask,5, axis=0)
    print(".")
    full_image_1, full_mask_1, full_logit_1, full_flip_1, losses_1, vloss_0,TP_1,FP_1,TN_1,FN_1  = apply_to_case(net1,
                                                              case_data,
                                                              mask,5, axis=1)
    print(".")
    full_image_2, full_mask_2, full_logit_2, full_flip_2, losses_2, vloss_0,TP_2,FP_2,TN_2,FN_2  = apply_to_case(net1,
                                                             case_data,
                                                              mask,5, axis=2)
    print(".")


    full_image_0, full_mask_0, full_logit_3, full_flip_3, losses_0, vloss_0,TP_0,FP_0,TN_0,FN_0  = apply_to_case(net2,
                                                                                                case_data,
                                                              mask,5, axis=0)
    print(".")
    full_image_1, full_mask_1, full_logit_4, full_flip_4, losses_1, vloss_0,TP_1,FP_1,TN_1,FN_1  = apply_to_case(net2,
                                                              case_data,
                                                              mask,5, axis=1)
    print(".")
    full_image_2, full_mask_2, full_logit_5, full_flip_5, losses_2, vloss_0,TP_2,FP_2,TN_2,FN_2  = apply_to_case(net2,
                                                             case_data,
                                                              mask,5, axis=2)
    print(".")





    full_var_0 = np.abs(full_logit_0/full_flip_0)


    full_var_1 = np.abs(full_logit_1/full_flip_1)


    full_var_2 = np.abs(full_logit_2/full_flip_2)

    full_var_3 = np.abs(full_logit_3/full_flip_3)


    full_var_4 = np.abs(full_logit_4/full_flip_4)


    full_var_5 = np.abs(full_logit_5/full_flip_5)




    full_var_denom = 1/(1/full_var_0 + 1/full_var_1 + 1/full_var_2 + 1/full_var_3 + 1/full_var_4 + 1/full_var_5)

    full_var = full_var_denom


    weighted_logit = (full_logit_0/full_var_0 + full_logit_1/full_var_1 + full_logit_2/full_var_2 +
                         full_logit_3/full_var_3 + full_logit_4/full_var_4 + full_logit_5/full_var_5)*full_var_denom

    full_logit = (full_logit_0 + full_logit_1 + full_logit_2+full_logit_3 + full_logit_4 + full_logit_5)/6




    flair = nib.load(flair_filepaths[case])
    header = flair.header
    affine = flair.affine



    seg_var_ensemble = numpy.any(numpy.stack([weighted_logit[4]>0, weighted_logit[3]>0, weighted_logit[2]>0,weighted_logit[1]>0, full_logit[0]>0]), axis = 0)
    seg_var_ensemble = seg_var_ensemble*5
    edema = numpy.any(numpy.stack([weighted_logit[3]>0, weighted_logit[2]>0,weighted_logit[1]>0, weighted_logit[0]>0]), axis = 0)
    seg_var_ensemble[edema] = 2
    core = numpy.any(numpy.stack([weighted_logit[2]>0,weighted_logit[1]>0, weighted_logit[0]>0]), axis = 0)
    seg_var_ensemble[core] = 1
    enhancing = np.logical_and(weighted_logit[1]>weighted_logit[0], core)
    seg_var_ensemble[enhancing] = 4

    seg_var_nifti = nib.Nifti1Image(seg_var_ensemble, affine, header)
    #nib.save(seg_var_nifti, cases[case]+"/seg_variance_ensembling_1.nii")




    flair_masked = flair_data[case]*(seg_var_ensemble>0).astype(np.int)
    t1_masked = t1_data[case]*(seg_var_ensemble>0).astype(np.int)
    t2_masked = t2_data[case]*(seg_var_ensemble>0).astype(np.int)
    t1ce_masked = t1ce_data[case]*(seg_var_ensemble>0).astype(np.int)


    flair_masked_nifti = nib.Nifti1Image(flair_masked, affine, header)
        #nib.save(flair_masked_nifti, cases[case]+"/flair_masked.nii")

    t1_masked_nifti = nib.Nifti1Image(t1_masked, affine, header)
        #nib.save(t1_masked_nifti, cases[case]+"/t1_masked.nii")

    t2_masked_nifti = nib.Nifti1Image(t2_masked, affine, header)
        #nib.save(t2_masked_nifti, cases[case]+"/t2_masked.nii")

    t1ce_masked_nifti = nib.Nifti1Image(t1ce_masked, affine, header)
        #nib.save(t1ce_masked_nifti, cases[case]+"/t1ce_masked.nii")


    #nib.save(seg_nifti,"/media/user/Daten/Brats2018/seg_plain_ensembling_1/"+case_ids[case]+".nii.gz")

    #seg_var_ensemble[seg_var_ensemble == 5] = 0
    #seg_nifti = nib.Nifti1Image(seg_var_ensemble, affine, header)
    #nib.save(seg_var_nifti,"/media/user/Daten/Brats2018/seg_variance_ensembling_1/"+case_ids[case]+".nii.gz")


    case_data = [flair_masked,t1_masked,t2_masked,t1ce_masked]

    mask = (t2_masked>0).astype(np.int)

    full_image_0, full_mask_0, full_logit_6, full_flip_6, losses_0, vloss_0,TP_0,FP_0,TN_0,FN_0  = apply_to_case(net1,
                                                                                                case_data,
                                                              mask,5, axis=0)
    print(".")
    full_image_1, full_mask_1, full_logit_7, full_flip_7, losses_1, vloss_0,TP_1,FP_1,TN_1,FN_1  = apply_to_case(net1,
                                                              case_data,
                                                              mask,5, axis=1)
    print(".")
    full_image_2, full_mask_2, full_logit_8, full_flip_8, losses_2, vloss_0,TP_2,FP_2,TN_2,FN_2  = apply_to_case(net1,
                                                             case_data,
                                                              mask,5, axis=2)
    print(".")


    full_image_0, full_mask_0, full_logit_9, full_flip_9, losses_0, vloss_0,TP_0,FP_0,TN_0,FN_0  = apply_to_case(net2,
                                                                                                case_data,
                                                              mask,5, axis=0)
    print(".")
    full_image_1, full_mask_1, full_logit_10, full_flip_10, losses_1, vloss_0,TP_1,FP_1,TN_1,FN_1  = apply_to_case(net2,
                                                              case_data,
                                                              mask,5, axis=1)
    print(".")
    full_image_2, full_mask_2, full_logit_11, full_flip_11, losses_2, vloss_0,TP_2,FP_2,TN_2,FN_2  = apply_to_case(net2,
                                                             case_data,
                                                              mask,5, axis=2)
    print(".")


    full_var_6 = np.abs(full_logit_6/full_flip_6)


    full_var_7 = np.abs(full_logit_7/full_flip_7)


    full_var_8 = np.abs(full_logit_8/full_flip_8)

    full_var_9 = np.abs(full_logit_9/full_flip_9)


    full_var_10 = np.abs(full_logit_10/full_flip_10)


    full_var_11 = np.abs(full_logit_11/full_flip_11)

    full_var_denom = 1/(1/full_var_0 + 1/full_var_1 + 1/full_var_2 + 1/full_var_3 + 1/full_var_4 + 1/full_var_5+
                           1/full_var_6 + 1/full_var_7 + 1/full_var_8 + 1/full_var_9 + 1/full_var_10 + 1/full_var_11)

    full_var = full_var_denom


    weighted_logit = (full_logit_0/full_var_0 + full_logit_1/full_var_1 + full_logit_2/full_var_2 +
                         full_logit_3/full_var_3 + full_logit_4/full_var_4 + full_logit_5/full_var_5 +
                         full_logit_6/full_var_6 + full_logit_7/full_var_7 + full_logit_8/full_var_8 +
                         full_logit_9/full_var_9 + full_logit_10/full_var_10 + full_logit_11/full_var_11)*full_var_denom

    full_logit = (full_logit_0 + full_logit_1 + full_logit_2+full_logit_3 + full_logit_4 + full_logit_5+
                     full_logit_6 + full_logit_7 + full_logit_8+full_logit_9 + full_logit_10 + full_logit_11)/12



    seg_var_ensemble = numpy.any(numpy.stack([weighted_logit[4]>0, weighted_logit[3]>0, weighted_logit[2]>0,weighted_logit[1]>0, full_logit[0]>0]), axis = 0)
    seg_var_ensemble = seg_var_ensemble*5
    edema = numpy.any(numpy.stack([weighted_logit[3]>0, weighted_logit[2]>0,weighted_logit[1]>0, weighted_logit[0]>0]), axis = 0)
    seg_var_ensemble[edema] = 2
    core = numpy.any(numpy.stack([weighted_logit[2]>0,weighted_logit[1]>0, weighted_logit[0]>0]), axis = 0)
    seg_var_ensemble[core] = 1
    enhancing = np.logical_and(weighted_logit[1]>weighted_logit[0], core)
    seg_var_ensemble[enhancing] = 4


    brain_mask = (seg_var_ensemble>0).astype(np.int32)

    seg_var_ensemble[seg_var_ensemble == 5] = 0





    enhancing_vol = np.sum(seg_var_ensemble==4)
    core_vol = np.sum(np.logical_or(seg_var_ensemble == 1, seg_var_ensemble == 4))
    edema_vol = np.sum(seg_var_ensemble>0)


    # In[225]:



    if core_vol == 0:
      edema = seg_var_ensemble >0
      seg_var_ensemble[edema] = 1


    from scipy import ndimage as ndi


    label_objects, nb_labels = ndi.label(seg_var_ensemble>0)
    if nb_labels > 1:
      for n in range(nb_labels):
        if np.sum(label_objects ==n+1) < 400:
          seg_var_ensemble[label_objects == n+1] = 0

    if edema_vol == 0:
      seg_var_ensemble = (brain_mask * 2).astype(np.int32)

    seg_nifti = nib.Nifti1Image(seg_var_ensemble, affine, header)
    return seg_nifti
    #nib.save(seg_nifti,"/data/results/tumor_DeepSCAN_class.nii.gz")
