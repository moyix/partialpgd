import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet18k import make_normalized_resnet18k

device = torch.device('cuda')

class MSEFlattenedLoss(nn.Module):
    def __init__(self):
        super(MSEFlattenedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x1, x2):
        # Flatten the tensors
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)

        # Compute the MSE loss between the flattened tensors
        return self.mse_loss(x1_flat, x2_flat)

class HookStopperException(Exception):
    pass

def intermediate_hook(module, input, output):
    raise HookStopperException(output)

def register_hook_by_name(model, layer_name, hook):
    for name, module in model.named_modules():
        if name == layer_name:
            return module.register_forward_hook(hook)
    raise ValueError(f"No module named {layer_name} was found in the model.")

def pgd_attack(model, images, labels, eps=1/255, alpha=2/255, iters=40):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    loss = nn.CrossEntropyLoss()

    adv_images = images.clone().detach()

    # Starting at a uniformly random point
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        # Calculate loss
        cost = loss(outputs, labels)
        print(f'Iter {i} cost {cost}')

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    return adv_images

def partial_pgd_attack(model, images, labels, ref_images, layer,
                       eps=1/255, alpha=2/255, iters=40, num_classes=10,
                       loss=MSEFlattenedLoss()):
    # Make sure the images are 0..1
    assert images.min() >= 0 and images.max() <= 1

    # Move the images and labels to the device
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # Random start
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    # Register the hook (can use a string to specify a layer by name)
    if isinstance(layer, str):
        hook = register_hook_by_name(model, layer, intermediate_hook)
    else:
        hook = layer.register_forward_hook(intermediate_hook)

    # Everything after this point has to be in a try block, so that the hook is
    # unregistered even if an exception is raised
    try:
        # Get the reference points for each target class
        with torch.no_grad():
            ref_images = ref_images.to(device)
            try:
                ref_outputs = model(ref_images)
            except HookStopperException as e:
                ref_outputs = e.args[0]

        # Pick a random target class ≠ original class
        target_labels = torch.randint(0, num_classes, labels.shape, device=device)
        target_labels = torch.where(target_labels == labels, (target_labels + 1) % num_classes, target_labels)
        print(f"Target labels:   {target_labels}")
        print(f"Original labels: {labels}")

        # Create the target activations. Basically this is [ref_outputs[i] for i in target_labels]
        target_outputs = torch.stack([ref_outputs[i] for i in target_labels])

        for i in range(iters):
            adv_images.requires_grad = True
            try:
                outputs = model(adv_images)
            except HookStopperException as e:
                outputs = e.args[0]

            # Calculate loss. This is the MSE between the activations of the
            # adversarial image and the reference image for the target class.
            # It is negative because we want to maximize the loss.
            cost = -loss(outputs, target_outputs)
            print(f'Iter {i} cost {cost}')

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images
    finally:
        hook.remove()

@torch.no_grad()
def make_ref_images(model, testdata, num_classes=10):
    ref_images = {}
    ref_vals = { i: -float('inf') for i in range(num_classes) }
    while len(ref_images) < num_classes:
        for batchnum, (images, labels) in enumerate(testdata):
            print(f"Batch {batchnum} of {len(testdata)}, {len(ref_images)} ref images found")
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            vals, preds = torch.max(outputs, 1)
            # For every correctly-predicted image, check if we should replace the reference image
            for i in range(len(images)):
                pred_label = preds[i].item()
                real_label = labels[i].item()
                if pred_label == real_label:
                    if vals[i] > ref_vals[pred_label]:
                        ref_vals[pred_label] = vals[i]
                        ref_images[pred_label] = images[i]
            if len(ref_images) == num_classes:
                break
    return torch.stack(list([ref_images[i] for i in range(num_classes)]))

def compute_linf(images1, images2):
    return torch.max(torch.abs(images1 - images2)).item()

def main():
    # Load the CIFAR10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='/fastdata/cifar10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Load the model
    model = make_normalized_resnet18k()
    model.load_state_dict(torch.load('resnet18_cifar10_nodp.pt'))
    model = model.to(device)
    model.eval()
