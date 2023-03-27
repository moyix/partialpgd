import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import time

import re

import numpy as np
from scipy.stats import entropy

import collections
import math
import heapq

import copy


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--kc', default=64, type=int, help='model size')
parser.add_argument('--epoch', default=200, type=int, help='total training epochs')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--noise', default=0, type=int, help='label noise %')
parser.add_argument('--eval', action='store_true', help='only do evaluation')
parser.add_argument('--quiet', action='store_true', help='be quiet')

parser.add_argument('--adv', '-q', action='store_true', help='adv training')
parser.add_argument('--epsilon', '-e', type=float, default=0.0314, 
        help='maximum perturbation of adversaries Linf (8/255=0.0314) L2 0.5')
parser.add_argument('--alpha', '-a', type=float, default=0.00784, 
        help='movement multiplier per iteration when generating adversarial examples (linf 2/255=0.00784) l2 0.157')
parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf', 
        help='the type of the perturbation (linf or l2)')
parser.add_argument('--k', '-k', type=int, default=7, 
        help='maximum iteration when generating adversarial examples')
        
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='momentum for optimizer')
parser.add_argument('--weight_decay', '-w', type=float, default=5e-4, 
        help='the parameter of l2 restriction for weights')

parser.add_argument('--freeze', action='store_true', help='freeze the unimportant layers')  
parser.add_argument('--last_freeze', action='store_true', help='freeze the last 5 layers')
parser.add_argument('--first_freeze', action='store_true', help='freeze the first 8 layers') 
parser.add_argument('--frac_freeze', action='store_true', help='freeze the first frac layers')

parser.add_argument('--stitch', action='store_true', help='stitch to train')
parser.add_argument('--frac', type=float,help='fraction of freeze layers')
parser.add_argument('--fine_tune', action='store_true',help='fine_tune with adv whole model')

parser.add_argument('--architecture', default=None, type=str, help='model architecture')
parser.add_argument('--dataset', default=None, type=str, help='dataset')

parser.add_argument('--time', action='store_true', help='count the time') 
parser.add_argument('--test', action='store_true', help='test')

parser.add_argument('--new_fgsm', action='store_true', help='new_attack')
parser.add_argument('--new_pgd', action='store_true', help='new_attack')
parser.add_argument('--old_attack', action='store_true', help='old_attack')

parser.add_argument('--me', type=str, help='stitch method')
parser.add_argument('--skip_freeze', action='store_true', help='freeze the skip layers')

parser.add_argument('--high_loss', action='store_true', help='high_loss layers')
parser.add_argument('--high_mean', action='store_true', help='high_mean layers')

parser.add_argument('--high_ortho', action='store_true', help='high_loss layers')
parser.add_argument('--high_entro', action='store_true', help='high_mean layers')
parser.add_argument('--low_spa', action='store_true', help='high_loss layers')

parser.add_argument('--adv_frac_freeze', action='store_true', help='freeze the first frac layers during adv training')
parser.add_argument('--frac_epoch', default=50, type=int, help='after this epoch we will use adv frac freeze and only train the first frac')
        
parser.add_argument('--test_code', action='store_true', help='only for test code')

parser.add_argument('--transfer_learning', default=None, type=str, help='checkpoint to do transfer learning')
parser.add_argument('--transfer_adv', action='store_true', help='transfer learning with adv training')
parser.add_argument('--transfer_clean', action='store_true', help='transfer learning with clean training')

parser.add_argument('--seed', default=0, type=int, help='random seed')

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

learning_rate = args.lr
""" epsilon = 0.0314
k = 10
alpha = 0.00784 """
if args.dataset == 'mnist':
    epsilon = 0.3
    k = 40
    alpha = 0.01
else:
    epsilon = 0.0314
    k = 10
    alpha = 0.00784
file_name = 'pgd_adversarial_training'
print(file_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset == 'cifar_10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')


class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)

        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv


if args.dataset == 'cifar_100':
    if args.architecture == "resnet18":
        net = torchvision.models.resnet18(num_classes = 100)
    elif args.architecture == "resnet34":     
        net = torchvision.models.resnet34(num_classes = 100)
    elif args.architecture == "resnet152":     
        net = torchvision.models.resnet152(num_classes = 100)
    elif args.architecture == "resnet101":     
        net = torchvision.models.resnet101(num_classes = 100)
    elif args.architecture == "vgg16":     
        net = torchvision.models.vgg16(num_classes = 100)
    elif args.architecture == "wide_resnet":     
        net = torchvision.models.wide_resnet50_2(num_classes = 100)
    elif args.architecture == "resnext":     
        net = torchvision.models.resnext50_32x4d(num_classes = 100)
elif args.dataset == 'mnist':
    net = ConvNet()
else:
    if args.architecture == "he":
        net = ResNet18()
    elif args.architecture == "resnet18":
        net = torchvision.models.resnet18(num_classes = 10)
    elif args.architecture == "resnet34":     
        net = torchvision.models.resnet34(num_classes = 10)
    elif args.architecture == "resnet152":     
        net = torchvision.models.resnet152(num_classes = 10)
    elif args.architecture == "resnet101":     
        net = torchvision.models.resnet101(num_classes = 10)
    elif args.architecture == "vgg16":     
        net = torchvision.models.vgg16(num_classes = 10)
    elif args.architecture == "wide_resnet":     
        net = torchvision.models.wide_resnet50_2(num_classes = 10)
    elif args.architecture == "resnext":     
        net = torchvision.models.resnext50_32x4d(num_classes = 10)
    elif args.architecture == "efficientnet":     
        #net = torchvision.models.alexnet(num_classes = 10)
        net = torchvision.models.efficientnet_b0(num_classes = 10)
    elif args.architecture == "alexnet":     
        net = torchvision.models.alexnet(num_classes = 10)
    elif args.architecture == "regnet":     
        #net = torchvision.models.vit_b_16(num_classes = 10)
        net = torchvision.models.regnet_y_400mf(num_classes = 10)
    

class ModelWrapper(nn.Module):
    def __init__(self, model, normalizer):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.normalizer = normalizer
    
    def forward(self,x):
        x = self.normalizer(x)
        return self.model(x)

if args.dataset == 'cifar_10' or args.dataset == 'cifar_100':
    normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    net = ModelWrapper(net, normalizer)
elif args.dataset == 'mnist':
    normalizer = transforms.Normalize((0.1307,), (0.3081,)) 
    net = ModelWrapper(net, normalizer)
elif args.dataset == 'GTSRB':
    print("GTSRB")
    #net = torchvision.models.resnet18(pretrained=True)
    normalizer = transforms.Normalize((0.3403, 0.3121, 0.3214),(0.2724, 0.2608, 0.2669))
    net = ModelWrapper(net, normalizer)

net = net.to(device)
if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if args.adv or args.fine_tune or args.frac_freeze or args.adv_frac_freeze or args.transfer_adv or args.last_freeze:
            #t1=time.time()
            adv = adversary.perturb(inputs, targets)
            #t2=time.time()
            #print("pgd perturb time",t2-t1)
            outputs = net(adv)
            #t3=time.time()
            #print("forward adv",t3-t2)
        else:
            outputs = net(inputs)


        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('\ntrain accuarcy:', 100. * correct / total)
    print('train loss:', train_loss)


def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

   
    step_epoch = 10
    if epoch % step_epoch == 0 and not args.eval:

        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        print("time:",elapsed)

        print('model',args.architecture)
        print('dataset',args.dataset)
        print('clean_acc',100. * benign_correct / total)
        print('adv_acc',100. * adv_correct / total)

        
        if args.adv_frac_freeze:
            print("adv_frac_freeze", args.adv_frac_freeze)
            print("frac", args.frac)
            print("frac epoch", args.frac_epoch)

        
        """ print('fine_tune',args.fine_tune)
        print('frac',args.frac_freeze) """

        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'clean_acc': 100. * benign_correct / total,
            'adv_acc': 100. * adv_correct / total,
            'epoch': epoch,
            'time': elapsed,
        }
        
        if args.frac_freeze: 
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/frac_freeze/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/frac_freeze/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/frac_freeze/lr{args.lr}_frac{args.frac}_advTrue_epoch{epoch}.pth')
        
        elif args.last_freeze: 
            
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/last_freeze/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/last_freeze/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/last_freeze/frac{args.frac}_advTrue_epoch{epoch}.pth')
        
        elif args.fine_tune:
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/fine_tune/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/fine_tune/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/fine_tune/lr{args.lr}_advTrue_epoch{epoch}.pth')

        elif args.adv:
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/adv/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/adv/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/adv/lr{args.lr}_advTrue_epoch{epoch}.pth')

        elif args.adv_frac_freeze:
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/adv_frac_freeze/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/adv_frac_freeze/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/adv_frac_freeze/frac{args.frac}_fracepoch{args.frac_epoch}_advTrue_epoch{epoch}.pth')
       
        elif args.transfer_adv:
            if not os.path.isdir(f'./checkpoint/w/cifar_10_checkpoints/{args.architecture}/{args.dataset}_transfer_learning/'):
                os.makedirs(f'./checkpoint/w/cifar_10_checkpoints/{args.architecture}/{args.dataset}_transfer_learning/')
            torch.save(state, f'./checkpoint/w/cifar_10_checkpoints/{args.architecture}/{args.dataset}_transfer_learning/lr{args.lr}_advTrue_epoch{epoch}.pth')
        
        elif args.transfer_clean:
            if not os.path.isdir(f'./checkpoint/w/cifar_10_checkpoints/{args.architecture}/{args.dataset}_transfer_learning/'):
                os.makedirs(f'./checkpoint/w/cifar_10_checkpoints/{args.architecture}/{args.dataset}_transfer_learning/')
            torch.save(state, f'./checkpoint/w/cifar_10_checkpoints/{args.architecture}/{args.dataset}_transfer_learning/lr{args.lr}_advFalse_epoch{epoch}.pth')
        
        else:
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/ggg/lr{args.lr}_advFalse_epoch{epoch}.pth')         

def freeze(model,adv_layers=None,clean_layers=None,freeze=False):
    
    if args.adv_frac_freeze or args.last_freeze or args.frac_freeze or args.stitch or args.high_loss or args.high_mean or args.high_entro or args.high_ortho or args.low_spa:
        layer_list = []

        #for layer, pa in model.state_dict().items():
        for layer, pa in model.named_parameters():
            if re.search(r'weight',layer) and not re.search(r'bn',layer) and not re.search(r'fc',layer) and not re.search(r'downsample',layer):
                layer_list.append(layer[7:])

        """ adv_path = f'./checkpoint/{args.dataset}_checkpoints/{args.architecture}/adv/advTrue_epoch100.pth'
        adv_ori = torch.load(adv_path)
        adv_net = adv_ori['net'] """

    else:
        path = f'/scratch/hx759/adversarial_training/cifar/model_criticality/freeze/{args.architecture}_{args.dataset}_attack_norml2_model_norml2_robust_acc_epoch100.pth'
        uniform_data = torch.load(path)[0]
        layer_list =torch.load(path)[1]
    
    unfreeze_layers = []
    
    if args.freeze:
        result = copy.deepcopy(uniform_data)
        max_number = heapq.nlargest(5, uniform_data[0])
        max_index = []
        for t in max_number:
            index = uniform_data[0].index(t)
            max_index.append(index)
            uniform_data[0][index] = 0

        for i in range(len(layer_list)):
            if i in max_index:
                unfreeze_layers.append(layer_list[i])
        
        #print(unfreeze_layers)
        
        for layer, pa in model.named_parameters():
            if layer[7:] not in unfreeze_layers:
                pa.requires_grad = not fr
            else:
                pa.requires_grad = True

    elif args.last_freeze:
        l = int(len(layer_list) * args.frac)
        print(l)
        unfreeze_layers =layer_list[-l:]
        print(unfreeze_layers)

        for layer, pa in model.named_parameters():
            if layer[7:] not in unfreeze_layers:
                pa.requires_grad = False
            else:
                pa.requires_grad = True   

    elif args.first_freeze:
        unfreeze_layers =layer_list[:8]
        print(unfreeze_layers)

        for layer, pa in model.named_parameters():
            if layer[7:] not in unfreeze_layers:
                pa.requires_grad = False
            else:
                pa.requires_grad = True   

    elif args.adv_frac_freeze:
        ###new layer list
        l = int(len(layer_list) * args.frac)
        print(l)
        unfreeze_layers =layer_list[:l]
        print(len(unfreeze_layers))
        print(unfreeze_layers)

        for layer, pa in model.named_parameters():
            if layer[7:] not in unfreeze_layers:
                pa.requires_grad = False
            else:
                pa.requires_grad = True  

        for n,p in model.state_dict().items():
            if n[7:] not in unfreeze_layers:
                print(n,p.requires_grad) 

        for n,p in model.state_dict(keep_vars=True).items():
            if n[7:] not in unfreeze_layers:
                print(n,p.requires_grad) 


    
    elif args.frac_freeze:
        ###new layer list
        """ path = f'/scratch/hx759/adversarial_training/cifar/test_model_criticality/new_adv_{args.architecture}_{args.dataset}_epoch100.pth'
        uniform_data = torch.load(path)[0]
        layer_list =torch.load(path)[1]

        result = copy.deepcopy(uniform_data)
        l = int(len(layer_list) * args.frac)
        print(l)
        max_number = heapq.nlargest(l, uniform_data[0])
        max_index = []
        for t in max_number:
            index = uniform_data[0].index(t)
            max_index.append(index)
            uniform_data[0][index] = 0
        print(max_index)

        for i in range(len(layer_list)):
            if i in max_index:
                unfreeze_layers.append(layer_list[i])
        print(unfreeze_layers) """

        l = int(len(layer_list) * args.frac)
        print(l)
        unfreeze_layers =layer_list[:l]
        print(len(unfreeze_layers))
        print(unfreeze_layers)

        for layer, pa in model.named_parameters():
            
            if layer[7:] not in unfreeze_layers:
                pa.requires_grad = False
            else:
                pa.requires_grad = True   

    elif args.skip_freeze:
        s = int(len(layer_list) * 0.3)
        e = int(len(layer_list) * 0.5)
        print(len(layer_list))
        del layer_list[s:e]
        #unfreeze_layers =layer_list[:s]
        print(len(layer_list))
        #print(unfreeze_layers)

        for layer, pa in model.named_parameters():
            if layer[7:] not in layer_list:
                pa.requires_grad = False
            else:
                pa.requires_grad = True   

   
    return unfreeze_layers


def get_layers(model):
    layer_list = []
    adv_layers = []
    clean_layers = []

    for layer, pa in model.state_dict().items():

        if re.search(r'conv*([0-9]+)', layer) or re.search(r'fc', layer) or re.search(r'downsample', layer) and re.search(r'weight',layer):
            layer_list.append(layer[7:])
    
    n = int(len(layer_list) * args.frac)
    for i in range(len(layer_list)):
        if i <= n:
            adv_layers.append(layer_list[i])
        else:
            clean_layers.append(layer_list[i])
    print("get layers")
    print(adv_layers)
    return adv_layers,clean_layers



adversary = LinfPGDAttack(net)

start_epoch=0

if args.time:
    start = time.time()

for epoch in range(start_epoch, start_epoch + args.epoch + 1):
    
    train(epoch)

    test(epoch)
    scheduler.step()
