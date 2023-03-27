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

args = parser.parse_args()


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
        out = self.linear(out)
        return out


def PreActResNet18():
    return ResNet(PreActBlock, [2,2,2,2])

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


""" class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out """

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output


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

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


elif args.dataset == 'mnist':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        ])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.MNIST('./data_mnist', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.MNIST('./data_mnist', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=128, shuffle=False, num_workers=2)

elif args.dataset == 'cifar_100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR100(root='./data_cifar_100', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data_cifar_100', train=False,
                                        download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

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
    

    

""" if args.dataset == 'mnist':
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
 """
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
    


def ortho_cond(W):
    n = W.shape[0]
    W = W.reshape(n, -1)
    t = np.expand_dims(np.linalg.norm(W, axis=1), axis=1)
    t[t == 0] = 1
    W = W / t
    c = np.abs(W.dot(W.T) - np.eye(n)).sum() / (n*(n-1))
    
    return 1 - c

def entropy_cond(w):
    import scipy.stats

    u, s, v_h = np.linalg.svd(w - w.mean(axis=0), full_matrices=False, compute_uv=True)
    v = s**2 / (w.shape[0]-1)
    vv = scipy.stats.entropy(v, base=10)
    vv = np.mean(vv)
    return vv

def sparsity(w):
    #w = w.detach().cpu().numpy()
    ma = np.max(np.abs(w.flatten()))/100

    count = 0

    for ii in range(len(w)):
        wi = w[ii]
        for jj in range(len(wi)):
            wj = wi[jj]
            #print(wj.shape)
            mi = np.max(np.abs(wj.flatten()))
            if mi <= ma:
                count += 1

    ratio = count/(len(w)*len(w[0]))
    return ratio


net = net.to(device)
#net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

def train(epoch):
    if args.fine_tune:
        print("fine_tune")
    elif args.frac_freeze:
        print("frac_freeze")
        print(args.frac)

    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if args.adv or args.fine_tune or args.frac_freeze or args.adv_frac_freeze:
            adv = adversary.perturb(inputs, targets)
            outputs = net(adv)
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

def stitch(net,epoch,adv_layers,clean_layers):
    print("stitch")
    print("frac",args.frac)
    print("method",args.me)
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        freeze(net, adv_layers,clean_layers,freeze = True)

        if epoch <= 1:

            print("first")
            for layer, pa in net.named_parameters():
                if pa.requires_grad:
                    print(layer)
        
        optimizer1.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer1.step()


        freeze(net, adv_layers,clean_layers,freeze = False)

        if epoch <= 1:
            print("second:")
            for layer, pa in net.named_parameters():
                if pa.requires_grad:
                    print(layer)
        
        optimizer2.zero_grad()

        adv_data = adversary.perturb(inputs, targets)

        outputs = net(adv_data)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer2.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


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

    """ if args.adv:
        step_epoch = 1
    else:
        step_epoch = 100 """

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

        print("adv_frac_freeze", args.adv_frac_freeze)
        if args.adv_frac_freeze:
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
        
        if args.stitch:
            if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/stitch/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/stitch/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/stitch/method{args.me}_frac{args.frac}_advTrue_epoch{epoch}.pth')
        elif args.skip_freeze:
            if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/skip_freeze/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/skip_freeze/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/skip_freeze/frac30_50_advTrue_epoch{epoch}.pth')
        
        elif args.frac_freeze:
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/frac_freeze/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/frac_freeze/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/frac_freeze/lr{args.lr}_frac{args.frac}_advTrue_epoch{epoch}.pth')
        
            """ if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/frac_freeze/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/frac_freeze/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/frac_freeze/frac{args.frac}_advTrue_epoch{epoch}.pth')
         """
        elif args.fine_tune:
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/fine_tune/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/fine_tune/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/fine_tune/lr{args.lr}_advTrue_epoch{epoch}.pth')

        
            """ if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/fine_tune/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/fine_tune/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/fine_tune/lr{args.lr}_advTrue_epoch{epoch}.pth')
        """
        elif args.high_loss:
            if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_loss/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_loss/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_loss/advTrue_frac{args.frac}_epoch{epoch}.pth')
        elif args.high_mean:
            if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_mean/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_mean/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_mean/advTrue_epoch{epoch}.pth')
        
        elif args.high_ortho:
            if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_ortho/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_ortho/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_ortho/advTrue_epoch{epoch}.pth')
        elif args.high_entro:
            if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_entro/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_entro/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/high_entro/advTrue_epoch{epoch}.pth')
        elif args.low_spa:
            if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/low_spa/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/low_spa/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/low_spa/advTrue_epoch{epoch}.pth')
       
        elif args.adv:
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/adv/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/adv/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/adv/lr{args.lr}_advTrue_epoch{epoch}.pth')


        elif args.adv_frac_freeze:
            if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/adv_frac_freeze/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/adv_frac_freeze/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/adv_frac_freeze/bn_frac{args.frac}_fracepoch{args.frac_epoch}_advTrue_epoch{epoch}.pth')
        
        else:
            if not os.path.isdir(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/'):
                os.makedirs(f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/')
            torch.save(state, f'./checkpoint/w/{args.dataset}_checkpoints/{args.architecture}/check/old/lr{args.lr}_advFalse_epoch{epoch}.pth')         


        """ else:
            if not os.path.isdir(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/'):
                os.makedirs(f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/')
            torch.save(state, f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/advFalse_epoch{epoch}.pth')      
 """
def freeze(model,adv_layers=None,clean_layers=None,freeze=False):
    
    if args.adv_frac_freeze  or args.frac_freeze or args.stitch or args.high_loss or args.high_mean or args.high_entro or args.high_ortho or args.low_spa:
        layer_list = []

        for layer, pa in model.state_dict().items():
            #if re.search(r'weight',layer) and not re.search(r'bn',layer) and not re.search(r'fc',layer) and not re.search(r'downsample',layer):
                layer_list.append(layer[7:])

        """ adv_path = f'./test_checkpoint/{args.dataset}_checkpoints/{args.architecture}/advTrue_epoch100.pth'
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
        unfreeze_layers =layer_list[-5:]
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

    elif args.stitch:
        for layer, pa in model.named_parameters():
            if args.me == 'm1':
                if layer[7:] in adv_layers:
                    pa.requires_grad = not freeze
                elif layer[7:] in clean_layers:
                    pa.requires_grad = freeze
            elif args.me == 'm2':
                if layer[7:] in adv_layers:
                    pa.requires_grad = not freeze
                else:
                    pa.requires_grad = freeze
            elif args.me == 'm3':
                if layer[7:] in clean_layers:
                    pa.requires_grad = freeze

    elif args.high_mean:
        en_list = []
        s_list = []
        ll = int(len(layer_list) * 0.2)
        for layer, pa in adv_net.items():
            if layer[7:] in layer_list:
                
                wi = pa.detach().cpu().numpy()
                m = np.mean(np.abs(wi.flatten()))
                
                en_list.append(m)
        
        max_number = heapq.nlargest(ll, en_list)
        max_index = []
        for t in max_number:
            index = en_list.index(t)
            max_index.append(index)
            #en_list[index] = 0

        select_layer = []
        for ii in max_index:
            select_layer.append(layer_list[ii])

        print(select_layer)

        for layer, pa in model.named_parameters():
            if layer[7:] in select_layer:
                pa.requires_grad = True
            else:
                pa.requires_grad = False


    elif args.high_loss:
        en_list = []
        s_list = []
        ll = int(len(layer_list) * args.frac)
        for layer, pa in adv_net.items():
            if layer[7:] in layer_list:
                
                wi = pa.detach().cpu().numpy()
                hi,_ = np.histogram(wi.flatten(),bins=100)
                en = entropy(hi)
                
                en_list.append(en)
        
        max_number = heapq.nlargest(ll, en_list)
        max_index = []
        for t in max_number:
            index = en_list.index(t)
            max_index.append(index)
            #en_list[index] = 10
        print("index")
        print(max_index)
        select_layer = []
        for ii in max_index:
            select_layer.append(layer_list[ii])

        print(select_layer)

        for layer, pa in model.named_parameters():
            if layer[7:] in select_layer:
                pa.requires_grad = True
            else:
                pa.requires_grad = False

    elif args.high_ortho:
        en_list = []
        
        ll = int(len(layer_list) * 0.2)
        for layer, pa in adv_net.items():
            if layer[7:] in layer_list:
                
                wi = pa.detach().cpu().numpy()
                o = ortho_cond(wi)
                en_list.append(o)


        
        max_number = heapq.nlargest(ll, en_list)
        max_index = []
        
        for t in max_number:
            index = en_list.index(t)
            max_index.append(index)
            #en_list[index] = 0
        #print(max_index)

        select_layer = []
        for ii in max_index:
            select_layer.append(layer_list[ii])

        print(select_layer)

        for layer, pa in model.named_parameters():
            if layer[7:] in select_layer:
                pa.requires_grad = True
            else:
                pa.requires_grad = False

    elif args.high_entro:
        en_list = []
        
        ll = int(len(layer_list) * 0.2)
        for layer, pa in adv_net.items():
            if layer[7:] in layer_list:
                
                wi = pa.detach().cpu().numpy()
                v = entropy_cond(wi)
                en_list.append(v)
        
        max_number = heapq.nlargest(ll, en_list)
        max_index = []
        for t in max_number:
            index = en_list.index(t)
            max_index.append(index)
            #en_list[index] = 10

        select_layer = []
        for ii in max_index:
            select_layer.append(layer_list[ii])

        print(select_layer)

        for layer, pa in model.named_parameters():
            if layer[7:] in select_layer:
                pa.requires_grad = True
            else:
                pa.requires_grad = False

    elif args.low_spa:
        en_list = []
        
        ll = int(len(layer_list) * 0.2)
        for layer, pa in adv_net.items():
            if layer[7:] in layer_list:
                
                wi = pa.detach().cpu().numpy()
                s = sparsity(wi)
                en_list.append(s)
        
        max_number = heapq.nsmallest(ll, en_list)
        max_index = []
        for t in max_number:
            index = en_list.index(t)
            max_index.append(index)
            #en_list[index] = 10

        select_layer = []
        for ii in max_index:
            select_layer.append(layer_list[ii])

        print(select_layer)

        for layer, pa in model.named_parameters():
            if layer[7:] in select_layer:
                pa.requires_grad = True
            else:
                pa.requires_grad = False
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



def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if args.epoch <= 100:
        print("60,80 lr drop")
        if epoch >= 60:
            lr /= 10
        if epoch >= 80:
            lr /= 10
    else:
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
    
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def eval(net,robust_model = False):
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0

    adversary = LinfPGDAttack(net)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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


if args.test_code:
    path = './test_checkpoint/cifar_10_checkpoints/resnet18/advTrue_epoch200.pth'
    net_ori = torchvision.models.resnet18(num_classes = 10)
    net_ori = ModelWrapper(net_ori, normalizer)
    ori = torch.load(path)
    
    names=[]

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for s, v in ori['net'].items():
            
            name = s[7:]  # remove `module.model.`
            new_state_dict[name] = v

            names.append(name)
            #print("name",name)
    net_ori.load_state_dict(new_state_dict)

    net = net_ori.to(device)
    #adversary = LinfPGDAttack(net)

    eval(net,robust_model = True)

    unfreeze_layers = freeze(net, freeze = True)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), args.lr, 
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
    
    eval(net,robust_model = True)







start_epoch = 0

if args.resume:
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #start_epoch = checkpoint['epoch'] + 1
    time_before = checkpoint['time']

if args.freeze or args.last_freeze or args.first_freeze or args.frac_freeze or args.high_mean or args.high_loss or args.high_entro or args.high_ortho or args.low_spa:
    freeze(net, freeze = True)
    print("freeze")

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), args.lr, 
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)

adversary = LinfPGDAttack(net)

if args.time:
    start = time.time()

if args.stitch:
    optimizer1 = torch.optim.SGD(net.parameters(), args.lr, 
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)
    adv_layers,clean_layers = get_layers(net)
    freeze(net, adv_layers,clean_layers,freeze = True)
    optimizer2 = torch.optim.SGD(net.parameters(), args.lr, 
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)

if args.fine_tune or args.frac_freeze or args.skip_freeze or args.high_mean or args.high_loss or args.high_entro or args.high_ortho or args.low_spa:
    test(0)

for epoch in range(start_epoch, start_epoch + args.epoch + 1):
    adjust_learning_rate(optimizer, epoch)
    if args.stitch:
        stitch(net,epoch,adv_layers,clean_layers)
    else:
        train(epoch)
        if args.adv_frac_freeze and epoch == args.frac_epoch:
            unfreeze_layers = freeze(net, freeze = True)
            print("freeze")

            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), args.lr, 
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
                
                    
        """ print("bbb")
        for n,p in net.state_dict(keep_vars=True).items():
            
                print(n,p.requires_grad) """
    test(epoch)

