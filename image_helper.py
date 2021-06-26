import math
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.resnet_cifar import ResNet18_local
from models.resnet_cifar import ResNet18_target
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18
logger = logging.getLogger("logger")
import config
from config import device
import copy
import cv2
import torch.nn as nn
import yaml

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime
import json
start=0
end=6

class Normalize_layer(nn.Module):

    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)

    def forward(self, input):
        return input.sub(self.mean).div(self.std)

class ImageHelper(Helper):

    def create_insert_pixel(self):

            # insert_pixel
            loader_test = torch.utils.data.DataLoader(self.train_dataset, batch_size=1, shuffle=False, num_workers=0)
            model_attack = Attack(dataloader
            =loader_test,
                                  attack_method='fgsm', epsilon=0.001)

            for batch_idx, (data, target) in enumerate(loader_test):
                data, target = data, target
                mins, maxs = data.min(), data.max()
                break

            self.target_model.eval()
            output = self.target_model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)

            for m in self.target_model.modules():
                if isinstance(m, quantized_conv) or isinstance(m, bilinear):
                    if m.weight.grad is not None:
                        m.weight.grad.data.zero_()

            loss.backward()
            for name, module in self.target_model.named_modules():
                if name == "1.linear":
                    w_v, w_id = module.weight.grad.detach().abs().topk(
                        200)  ## taking only 200 weights thus wb=200
                    tar = w_id[2]  ###target_class 2
                    print(tar)
                    np.savetxt('tar.txt', tar.numpy(), fmt='%f')
            start = 0
            end = 6
            for t, (x, y) in enumerate(loader_test):
                x[:, :, :, :] = 0
                x[:, 0:3, start:end, start:end] = 0.5  ## initializing the mask to 0.5
                break

            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]
            net_d = ResNet188()

            net2 = torch.nn.Sequential(
                Normalize_layer(mean, std),
                net_d
            )
            net2.load_state_dict(torch.load('Resnet18_8bit.pkl'))
            y = net2(x)  ##initializaing the target value for trigger generation
            y[:, tar] = 100  ### setting the target of certain neurons to a larger value 10

            ep = 0.5
            ### iterating 200 times to generate the trigger
            for i in range(20):
                x_tri = model_attack.attack_method(
                    net2, x, y, tar, ep, mins, maxs)
                x = x_tri

            ep = 0.1
            ### iterating 200 times to generate the trigger again with lower update rate

            for i in range(20):
                x_tri = model_attack.attack_method(
                    net2, x, y, tar, ep, mins, maxs)
                x = x_tri

            ep = 0.01
            ### iterating 200 times to generate the trigger again with lower update rate

            for i in range(20):
                x_tri = model_attack.attack_method(
                    net2, x, y, tar, ep, mins, maxs)
                x = x_tri

            ep = 0.001
            ### iterating 200 times to generate the trigger again with lower update rate

            for i in range(20):
                x_tri = model_attack.attack_method(
                    net2, x, y, tar, ep, mins, maxs)
                x_ = x_tri
            insert_pixle = x_
            # test(target_model, loader_test)
            test1(self.target_model, loader_test, insert_pixle)
            np.savetxt('trojan_img1.txt', x_tri[0, 0, :, :].cpu().numpy(), fmt='%f')
            np.savetxt('trojan_img2.txt', x_tri[0, 1, :, :].cpu().numpy(), fmt='%f')
            np.savetxt('trojan_img3.txt', x_tri[0, 2, :, :].cpu().numpy(), fmt='%f')


    def create_model(self):
        local_model=None
        target_model=None
        if self.params['type']==config.TYPE_CIFAR:
            net_c = ResNet18_local()
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]
            net = torch.nn.Sequential(
                Normalize_layer(mean, std),
                net_c
            )
            net_d = ResNet18_target()
            net_target = torch.nn.Sequential(
                Normalize_layer(mean, std),
                net_d
            )
            net.load_state_dict(torch.load('Resnet18_8bit.pkl'))
            local_model=net
            target_model=net_target
            net_target.load_state_dict(torch.load('Resnet18_8bit.pkl'))

        elif self.params['type']==config.TYPE_MNIST:
            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'])
            target_model = MnistNet(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type']==config.TYPE_TINYIMAGENET:

            local_model= resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])

        local_model=local_model.to(device)
        target_model=target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available() :
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}",map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

    def build_classes_dict(self):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = self.classes_dict
        class_size = len(cifar_classes[0]) #for cifar: 5000
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())  # for cifar: 10

        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)
        # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
        return per_participant_list

    def draw_dirichlet_plot(self,no_classes,no_participants,image_nums,alpha):
        fig= plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()
        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))
        vis_par=[0,10,20,30]
        for k in range(no_participants):
        # for k in vis_par:
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            widths = s[k]
            xcenters = left + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # for y, (x, c) in enumerate(zip(xcenters, widths)):
            #     plt.text(x, y, str(int(c)), ha='center', va='center',
            #              color=text_color,fontsize='small')
            left += s[k]
        plt.legend(ncol=20,loc='lower left',  bbox_to_anchor=(0, 1),fontsize=4) #
        # plt.legend(ncol=len(vis_par), bbox_to_anchor=(0, 1),
        #            loc='lower left', fontsize='small')
        plt.xlabel("Number of Images", fontsize=16)
        # plt.ylabel("Label 0 ~ 199", fontsize=16)
        # plt.yticks([])
        fig.tight_layout(pad=0.1)
        # plt.ylabel("Label",fontsize='small')
        fig.savefig(self.folder_path+'/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range_no_id)), \
               torch.utils.data.DataLoader(self.test_dataset,
                                            batch_size=self.params['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                poison_label_inds))
    def load_data(self):
        logger.info('Loading data')
        dataPath = './data'
        if self.params['type'] == config.TYPE_CIFAR:
            ### data load
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                             transform=transform_train)

            self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=transform_test)

        elif self.params['type'] == config.TYPE_MNIST:

            self.train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            self.test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]))
        elif self.params['type'] == config.TYPE_TINYIMAGENET:

            _data_transforms = {
                'train': transforms.Compose([
                    # transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    # transforms.Resize(224),
                    transforms.ToTensor(),
                ]),
            }
            _data_dir = './data/tiny-imagenet-200/'
            self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                    _data_transforms['train'])
            self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                   _data_transforms['val'])
            logger.info('reading data done')

        self.classes_dict = self.build_classes_dict()
        logger.info('build_classes_dict done')
        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'], #100
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
            train_whole_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                             batch_size=self.params['batch_size'],
                                                            pin_memory=True, shuffle=True,num_workers=8)
        else:
            ## sample indices for participants that are equally
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]

        logger.info('train loaders done')
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.test_data_poison ,self.test_targetlabel_data = self.poison_test_dataset()
        self.train_whole_data=train_whole_loader
        self.advasarial_namelist = self.params['adversary_list']

        if self.params['is_random_namelist'] == False:
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['number_of_total_participants']))
        # random.shuffle(self.participants_list)
        self.benign_namelist =list(set(self.participants_list) - set(self.advasarial_namelist))

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices),pin_memory=True, num_workers=8)
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        return test_loader


    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def get_poison_batch(self, bptt,evaluation=False):

        images, targets = bptt

        poison_count= 0
        new_images=copy.deepcopy(images)
        new_targets=copy.deepcopy(targets)

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = self.params['poison_label_swap']
                new_images[index] = self.add_pixel_pattern(images[index])
                poison_count+=1

            else: # poison part of data when training
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    new_images[index] = self.add_pixel_pattern(images[index])
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count

    def add_pixel_pattern(self,ori_image):
        image = copy.deepcopy(ori_image)
        a1 = np.loadtxt('trojan_img1.txt', dtype=float)
        a1 = torch.Tensor(a1).long()

        a2 = np.loadtxt('trojan_img2.txt', dtype=float)
        a2 = torch.Tensor(a2).long()

        a3 = np.loadtxt('trojan_img3.txt', dtype=float)
        a3 = torch.Tensor(a3).long()

        insert_pixel=torch.stack([a1,a2,a3],dim=0)
        ori_image[0:3, start:end, start:end] = insert_pixel[0:3, start:end, start:end]

        return ori_image


class _Quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, step):
        ctx.step = step.item()
        output = torch.round(input / ctx.step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step
        return grad_input, None

class quantized_conv(nn.Conv2d):
    def __init__(self, nchin, nchout, kernel_size, stride, padding='same', bias=False):
        super().__init__(in_channels=nchin, out_channels=nchout, kernel_size=kernel_size, padding=padding,
                         stride=stride, bias=False)
        # self.N_bits = 7
        # step = self.weight.abs().max()/((2**self.N_bits-1))
        # self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2 ** self.N_bits - 1))
        quantize1 = _Quantize.apply
        QW = quantize1(self.weight, step)

        return F.conv2d(input, QW * step, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.m = nn.MaxPool2d(5, stride=5)
        # self.lin = nn.Linear(64*6*6,1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = bilinear(512 * block.expansion, num_classes)
        # self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out1 = out.view(out.size(0), -1)
        out = self.linear(out1)
        return out


class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        # self.N_bits = 7
        # step = self.weight.abs().max()/((2**self.N_bits-1))
        # self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)
        # self.weight.data = quantize(self.weight, self.step).data.clone()

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2 ** self.N_bits - 1))
        quantize1 = _Quantize.apply
        QW = quantize1(self.weight, step)

        return F.linear(input, QW * step, self.bias)
## netwrok to generate the trigger  removing the last layer.
class ResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet1, self).__init__()
        self.in_planes = 64

        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = bilinear(512 * block.expansion, num_classes)
        # self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    np.random.seed(1)
    with open(f'./utils/cifar_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                        name=params_loaded.get('name', 'mnist'))
    helper.load_data()

    pars= list(range(100))
    # show the data distribution among all participants.
    count_all= 0
    for par in pars:
        cifar_class_count = dict()
        for i in range(10):
            cifar_class_count[i] = 0
        count=0
        _, data_iterator = helper.train_data[par]
        for batch_id, batch in enumerate(data_iterator):
            data, targets= batch
            for t in targets:
                cifar_class_count[t.item()]+=1
            count += len(targets)
        count_all+=count
        print(par, cifar_class_count,count,max(zip(cifar_class_count.values(), cifar_class_count.keys())))

    print('avg', count_all*1.0/100)

class Attack(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0,
                 epsilon=0.031, attack_method='pgd'):

        if criterion is not None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()

        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id  # this is integer

        if attack_method == 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method == 'pgd':
            self.attack_method = self.pgd

    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader

        if attack_method is not None:
            if attack_method == 'fgsm':
                self.attack_method = self.fgsm

    def fgsm(self, model, data, target, tar, ep, data_min=0, data_max=1):

        model.eval()
        # perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()

        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = self.criterion(output[:, tar], target[:, tar])
        print(loss)
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward(retain_graph=True)

        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False
        start=0
        end=6
        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_data[:, 0:3, start:end, start:end] -= ep * sign_data_grad[:, 0:3, start:end,
                                                                 start:end]  ### 11X11 pixel would yield a TAP of 11.82 %
            perturbed_data.clamp_(data_min, data_max)

        return perturbed_data


class quantized_conv(nn.Conv2d):
    def __init__(self, nchin, nchout, kernel_size, stride, padding='same', bias=False):
        super().__init__(in_channels=nchin, out_channels=nchout, kernel_size=kernel_size, padding=padding,
                         stride=stride, bias=False)
        # self.N_bits = 7
        # step = self.weight.abs().max()/((2**self.N_bits-1))
        # self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2 ** self.N_bits - 1))
        quantize1 = _Quantize.apply
        QW = quantize1(self.weight, step)

        return F.conv2d(input, QW * step, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class _Quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, step):
        ctx.step = step.item()
        output = torch.round(input / ctx.step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step
        return grad_input, None


class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        # self.N_bits = 7
        # step = self.weight.abs().max()/((2**self.N_bits-1))
        # self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)
        # self.weight.data = quantize(self.weight, self.step).data.clone()

    def forward(self, input):
        self.N_bits = 7
        step = self.weight.abs().max() / ((2 ** self.N_bits - 1))
        quantize1 = _Quantize.apply
        QW = quantize1(self.weight, step)

        return F.linear(input, QW * step, self.bias)


class Normalize_layer(nn.Module):

    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)

    def forward(self, input):
        return input.sub(self.mean).div(self.std)

def ResNet188():
    return ResNet1(BasicBlock, [2,2,2,2])

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = quantized_conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = quantized_conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                quantized_conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride,padding=0, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #print('value2')
        #print(self.l)
        return out


class ResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet1, self).__init__()
        self.in_planes = 64

        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = bilinear(512 * block.expansion, num_classes)
        # self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def test1(model, loader, xh):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    for x, y in loader:

        x[:, 0:3, start:end, start:end] = xh[:, 0:3, start:end, start:end]
        # grid_img = torchvision.utils.make_grid(x_var[0,:,:,:], nrow=1)
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.show()
        y[:] = targets  ## setting all the target to target class

        scores = model(x)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data'
          % (num_correct, num_samples, 100 * acc))

    return acc

def test(model, loader, blackbox=False, hold_out_size=None):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    if blackbox:
        num_samples -= hold_out_size

    for x, y in loader:
        scores = model(x)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data'
        % (num_correct, num_samples, 100 * acc))

    return acc