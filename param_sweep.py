import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet18k import make_normalized_resnet18k, make_resnet18k, Normalizer
from partialpgd import make_ref_images, pgd_attack, partial_pgd_attack, compute_linf
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import get_graph_node_names
import seaborn as sns

device = torch.device('cuda')
# Load the CIFAR10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='/fastdata/cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load the model
model = make_normalized_resnet18k().to(device)
model.load_state_dict(torch.load('resnet18_cifar10_nodp.pt'))
model.eval()

# Get 16 random images from the test set and show them
shufloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)
images, labels = next(iter(shufloader))

# Original predictions
orig_outputs = model(images.to(device))
_, orig_preds = torch.max(orig_outputs, 1)
orig_preds = orig_preds.cpu()

# Get reference images
ref_images = make_ref_images(model, testloader)

attack_layers = [name for name in get_graph_node_names(model)[0] if 'conv' in name or name.startswith('model.layer')]

best_success = (0.0, 'initial')
for attack_type in ['random', 'targeted']:
    for init in ['normal', 'uniform']:
        for layer in attack_layers:
            # Do a sweep over epsilons and alphas
            epsilons = np.linspace(0, 8/255, 11)
            alphas = np.linspace(0, 8/255, 11)
            trials = 10
            successes = np.zeros((trials, len(epsilons), len(alphas)))
            for t in range(trials):
                for i, eps in enumerate(epsilons):
                    for j, alpha in enumerate(alphas):
                        print(f'Layer {layer}, eps {eps:.3f}, alpha {alpha:.3f}, trial {t+1}/{trials}, init {init}, attack_type {attack_type}')
                        adv_images = partial_pgd_attack(model, images, labels, ref_images, layer=layer, eps=eps, alpha=alpha, iters=40, init_random=init)
                        outputs = model(adv_images.to(device))
                        _, preds = torch.max(outputs, 1)
                        preds = preds.cpu()
                        success_rate = (preds != orig_preds).sum().item()/len(orig_preds)
                        # Save the best success rate
                        if success_rate > best_success[0]:
                            best_success = (success_rate, f'layer {layer}, eps {eps:.3f}, alpha {alpha:.3f}, trial {t+1}/{trials}, init {init}, attack_type {attack_type}')
                            print(f'New best success rate: {best_success[0]:.3f} ({best_success[1]})')
                        successes[t,i,j] = success_rate
            success_filename = f'data/{attack_type}/{init}/layer_{layer}.npy'
            print(f'Saving successes to {success_filename}...')
            np.save(success_filename, successes)

            # Plot the results
            plt.figure()
            sns.heatmap(successes.max(axis=0), xticklabels=[f'{a:.3f}' for a in alphas], yticklabels=[f'{e:.3f}' for e in epsilons], annot=True, fmt='.2f')
            plt.xlabel('alpha')
            plt.ylabel('epsilon')
            plt.title(f'Attack {attack_type} on Layer {layer}\nMax Success rate, 10 trials, {init.capitalize()} Init')
            plt.savefig(f'figs/{attack_type}/{init}/layer_{layer}.pdf')

print(f'Best success rate: {best_success[0]:.3f} ({best_success[1]})')
