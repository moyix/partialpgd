import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from resnet18k import make_normalized_resnet18k

import numpy as np

device = torch.device('cuda')

class CosFlattenedLoss(nn.Module):
    def __init__(self):
        super(CosFlattenedLoss, self).__init__()
        #best for now
        #self.cos_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
        self.cos = nn.CosineSimilarity(dim=-1)
        #self.cos_loss = 
        # Define weights for functions for Cos and MSE.
        """ self.w1 = 20
        self.w2 = 10

        self.cos = nn.CosineSimilarity(dim=-1) # Dim = -1 as our tensor is (batch_size, preds, len_heatmap)
        self.mse = torch.nn.MSELoss() """
        """ self.cos = nn.CosineSimilarity(dim=-1) """


        
        # Apply cumulative sum to both tensors and calculate loss.
        #self.cos_loss = nn.MSELoss()
        

    def forward(self, x1, x2):
        # Flatten the tensors
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)

        """ cos_sim = torch.abs(self.cos(torch.cumsum(x1_flat, dim=-1), torch.cumsum(x2_flat, dim=-1))).mean()
        mse_loss = self.mse(torch.cumsum(x1_flat, dim=-1), torch.cumsum(x2_flat, dim=-1))

        SS= (self.w1 * mse_loss) / (self.w2 * cos_sim)

        return SS """
        
        """ y_true=x1_flat
        y_pred=x2_flat
        smooth=100
        intersection= (y_true * y_pred).abs().sum(dim=-1)
        sum_ = torch.sum(y_true.abs() + y_pred.abs(), dim=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return ((1 - jac) * smooth).mean() """

        cos_loss = torch.abs(self.cos(torch.cumsum(x1_flat, dim=-1), torch.cumsum(x2_flat, dim=-1))).mean()
        return cos_loss

        # Compute the MSE loss between the flattened tensors
        #return self.cos_loss(x1_flat, x2_flat)
        #return self.cos_loss(x1, x2)


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
                       eps=1/255, alpha=2/255, iters=40, attempts=1, num_classes=10, images_per_class=1,
                       loss=CosFlattenedLoss(), init_random='uniform',
                       attack_type='targeted'):
    # Make sure the images are 0..1
    assert images.min() >= 0 and images.max() <= 1

    # Move the images and labels to the device
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # Random start, either uniform or normal
    adv_images = images.clone().detach()
    if init_random == 'uniform':
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    elif init_random == 'normal':
        adv_images = adv_images + torch.empty_like(adv_images).normal_(0, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    submodel = create_feature_extractor(model, {layer: layer})

    initial_outputs = submodel(images)[layer]

    # Get the reference points for each target class
    if attack_type == 'targeted':
        #print("targeted")
        ref_images = ref_images.to(device)
        ref_outputs = submodel(ref_images)[layer]
        # Normalize the reference outputs
        ref_outputs = F.normalize(ref_outputs, p=2, dim=-1)

        # Pick a random target class ≠ original class
        target_labels = torch.randint(0, num_classes, labels.shape, device=device)
        target_labels = torch.where(target_labels == labels, (target_labels + 1) % num_classes, target_labels)

        # Create the target activations. Basically this is [ref_outputs[i] for i in target_labels]
        #target_outputs = torch.stack([ref_outputs[i] for i in target_labels])

        ######use all multi images to train

        #print(ref_outputs.shape)
        """ target_outputs = torch.stack([ref_outputs[images_per_class*i:images_per_class*i+images_per_class] for i in target_labels])
 """
        # target_outputs = [ for each image we are attacking, pick the closest reference activation for the targeted class label ]
        # for each input image adv_images[i]:
        #     starting_activation = initial_outputs[i]
        #     target_label = target_labels[i]
        #     possible_targets = ref_outputs[images_per_class*i:images_per_class*i+images_per_class]
        #     target_activation = closest(possible_target, starting_activation)
        
        ######use the cloest images across multi images
        #print(ref_outputs.shape)
        """ target_out = torch.stack([ref_outputs[images_per_class*i:images_per_class*i+images_per_class] for i in target_labels])
        #print(target_out.shape)

        distance = torch.stack([torch.cdist(target_out[i,j,:,:,:],initial_outputs[i]).mean() for i in range(len(adv_images)) for j in range(images_per_class)])

        #print(distance.shape)

        target_outputs_list=[]
        for i in range(len(adv_images)):
            distance_per_image=distance[images_per_class*i:images_per_class*i+images_per_class]
            index=torch.argmin(distance_per_image)
            target_outputs_list.append(target_out[i,index,:,:,:])
        target_outputs = torch.stack([target_outputs_list[i] for i in range(len(adv_images))])

 """
        ######use the mean value across multi images
        target_outputs = torch.stack([ref_outputs[images_per_class*i:images_per_class*i+images_per_class].mean(axis=0) for i in target_labels])
        
        """ for i in range(len(adv_images)):
            all_possible_images = ref_outputs[images_per_class*i:images_per_class*i+images_per_class]
            print(all_possible_images.shape)
            mean_value=all_possible_images.mean(axis=0)
            print(mean_value.shape)
            target_outputs_list.append(mean_value)
        target_outputs = torch.stack([target_outputs_list[i] for i in range(len(adv_images))]) """

        #print(target_outputs.shape)

        initial_loss = None
        for i in range(int(iters/attempts)):
            adv_images.requires_grad = True
            outputs = submodel(adv_images)[layer]

            # Normalize the outputs
            outputs = F.normalize(outputs, p=2, dim=-1)

            # Calculate loss. This is the MSE between the activations of the
            # adversarial image and the reference image for the target class.
            # It is negative because we want to maximize the loss.
            
            ###training with multi images
            """ outputs=outputs.unsqueeze(1)
            outputs=outputs.repeat(1,images_per_class,1,1,1) """

            cost = -loss(outputs, target_outputs)

            if initial_loss is None:
                initial_loss = cost

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    

    elif attack_type == "random_away":
        print("random and away from initial")
        # Get the shape of the activations
        #init_feat = submodel(images)[layer]
        """ target_outputs = torch.randn_like(initial_outputs)
        target_outputs = F.normalize(target_outputs, p=2, dim=-1) """

        ini_outputs = torch.stack([initial_outputs[i]+ torch.empty_like(initial_outputs[i]).normal_(0, 1000*eps) for i in labels])
        
        initial_loss = None
        for i in range(int(iters/attempts)):
            adv_images.requires_grad = True
            outputs = submodel(adv_images)[layer]

            # Normalize the outputs
            outputs = F.normalize(outputs, p=2, dim=-1)

            # Calculate loss. This is the MSE between the activations of the
            # adversarial image and the reference image for the target class.
            # It is negative because we want to maximize the loss.
            cost = loss(outputs, ini_outputs)
            ####random add somthing to ini_ouputs

            if initial_loss is None:
                initial_loss = cost

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
    elif attack_type == "random_far":
        #print("random and get close to farest random vector")
        # Get the shape of the activations
        #init_feat = submodel(images)[layer]

        target_out_list=[]
        for a in range(attempts):
            target_outputs = torch.randn_like(initial_outputs)
            target_outputs = F.normalize(target_outputs, p=2, dim=-1)
            target_out_list.append(target_outputs)

        target_out = torch.stack([target_out_list[i] for i in range(attempts)])
        #print(target_out.shape)

        distance = torch.stack([torch.cdist(target_out[j,i,:,:,:],initial_outputs[i]).mean() for i in range(len(adv_images)) for j in range(attempts)])

        #print(distance.shape)

        target_outputs_list=[]
        for i in range(len(adv_images)):
            distance_per_image=distance[attempts*i:attempts*i+attempts]
            index=torch.argmax(distance_per_image)
            target_outputs_list.append(target_out[index,i,:,:,:])
        target_outputs = torch.stack([target_outputs_list[i] for i in range(len(adv_images))])
        #print(target_outputs.shape)


        for i in range(iters):

            initial_loss = None

            adv_images.requires_grad = True
            outputs = submodel(adv_images)[layer]

            # Normalize the outputs
            outputs = F.normalize(outputs, p=2, dim=-1)

            # Calculate loss. This is the MSE between the activations of the
            # adversarial image and the reference image for the target class.
            # It is negative because we want to maximize the loss.
            cost = -loss(outputs, target_outputs)

            if initial_loss is None:
                initial_loss = cost

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        # print(f"Change in loss: {final_loss - initial_loss}")
    

    elif attack_type == "random_asr":
        #print("random and choose the highest asr random vector")
        orig_outputs = model(images.to(device))
        _, orig_preds = torch.max(orig_outputs, 1)
        orig_preds = orig_preds.cpu()

        successes_list=[]
        adv_images_list=[]
        for a in range(attempts):

            target_outputs = torch.randn_like(initial_outputs)
            target_outputs = F.normalize(target_outputs, p=2, dim=-1)

            #ini_outputs = torch.stack([initial_outputs[i] for i in labels])
            
            initial_loss = None

            for i in range(int(iters/attempts)):
                adv_images.requires_grad = True
                outputs = submodel(adv_images.to(device))[layer]

                # Normalize the outputs
                outputs = F.normalize(outputs, p=2, dim=-1)

                # Calculate loss. This is the MSE between the activations of the
                # adversarial image and the reference image for the target class.
                # It is negative because we want to maximize the loss.
                cost = -loss(outputs, target_outputs)

                if initial_loss is None:
                    initial_loss = cost

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                            retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() + alpha*grad.sign()
                delta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # Get the new predictions
            outputs = model(adv_images.to(device))
            _, preds = torch.max(outputs, 1)

            # Move the images back to the CPU
            images = images.cpu()
            labels = labels.cpu()
            adv_images = adv_images.cpu()
            preds = preds.cpu()

            successes_list.append((preds != orig_preds).sum().item()/len(orig_preds))
            adv_images_list.append(adv_images)

        index=np.argmax(successes_list)
        adv_images=adv_images_list[index]
    final_loss = cost
        # print(f"Change in loss: {final_loss - initial_loss}")

    return adv_images



def new_partial_pgd_attack(model, images, labels, ref_images, layer,
                       eps=1/255, alpha=2/255, iters=40, attempts=1, num_classes=10, images_per_class=1,
                       loss=CosFlattenedLoss(), init_random='uniform',
                       attack_type='targeted'):
    # Make sure the images are 0..1
    assert images.min() >= 0 and images.max() <= 1

    # Move the images and labels to the device
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # Random start, either uniform or normal
    adv_images = images.clone().detach()
    if init_random == 'uniform':
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    elif init_random == 'normal':
        adv_images = adv_images + torch.empty_like(adv_images).normal_(0, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    submodel = create_feature_extractor(model, {layer: layer})

    initial_outputs = submodel(images)[layer]

    for a in range(attempts):

        # Get the reference points for each target class
        if attack_type == 'targeted':
            ref_images = ref_images.to(device)
            ref_outputs = submodel(ref_images)[layer]
            # Normalize the reference outputs
            ref_outputs = F.normalize(ref_outputs, p=2, dim=-1)

            # Pick a random target class ≠ original class
            target_labels = torch.randint(0, num_classes, labels.shape, device=device)
            target_labels = torch.where(target_labels == labels, (target_labels + 1) % num_classes, target_labels)

            # Create the target activations. Basically this is [ref_outputs[i] for i in target_labels]
            #target_outputs = torch.stack([ref_outputs[i] for i in target_labels])

            ###use the closest images across multi images

            print(ref_outputs.shape)
            target_out = torch.stack([ref_outputs[images_per_class*i:images_per_class*i+images_per_class] for i in target_labels])
            print(target_out.shape)

            # target_outputs = [ for each image we are attacking, pick the closest reference activation for the targeted class label ]
            # for each input image adv_images[i]:
            #     starting_activation = initial_outputs[i]
            #     target_label = target_labels[i]
            #     possible_targets = ref_outputs[images_per_class*i:images_per_class*i+images_per_class]
            #     target_activation = closest(possible_target, starting_activation)
            """ for i in range(len(adv_images)):
                for j in range(images_per_class):
                    possible_distance.append(torch.cdist(target_out[i,j,:,:,:],initial_outputs[i]).mean()) """

            distance = torch.stack([torch.cdist(target_out[i,j,:,:,:],initial_outputs[i]).mean() for i in range(len(adv_images)) for j in range(images_per_class)])

            print(distance.shape)

            target_outputs_list=[]
            for i in range(len(adv_images)):
                distance_per_image=distance[images_per_class*i:images_per_class*i+images_per_class]
                index=torch.argmin(distance_per_image)
                target_outputs_list.append(target_out[i,index,:,:,:])
            target_outputs = torch.stack([target_outputs_list[i] for i in range(len(adv_images))])


            ###use the mean value across multi images
            #target_outputs = torch.stack([ref_outputs[images_per_class*i:images_per_class*i+images_per_class].mean(axis=0) for i in target_labels])
            
            """ for i in range(len(adv_images)):
                all_possible_images = ref_outputs[images_per_class*i:images_per_class*i+images_per_class]
                print(all_possible_images.shape)
                mean_value=all_possible_images.mean(axis=0)
                print(mean_value.shape)
                target_outputs_list.append(mean_value)
            target_outputs = torch.stack([target_outputs_list[i] for i in range(len(adv_images))]) """

            print(target_outputs.shape)
        

        elif attack_type == "random":
            # Get the shape of the activations
            #init_feat = submodel(images)[layer]
            target_outputs = torch.randn_like(initial_outputs)
            target_outputs = F.normalize(target_outputs, p=2, dim=-1)

        initial_loss = None
        for i in range(int(iters/attempts)):
            adv_images.requires_grad = True
            outputs = submodel(adv_images)[layer]

            # Normalize the outputs
            outputs = F.normalize(outputs, p=2, dim=-1)

            # Calculate loss. This is the MSE between the activations of the
            # adversarial image and the reference image for the target class.
            # It is negative because we want to maximize the loss.
            cost = -loss(outputs, target_outputs)

            if initial_loss is None:
                initial_loss = cost

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    final_loss = cost
        # print(f"Change in loss: {final_loss - initial_loss}")
    return adv_images



@torch.no_grad()
def make_ref_images(model, testdata, num_classes=10):
    ref_images = {}
    ref_vals = { i: -float('inf') for i in range(num_classes) }
    while len(ref_images) < num_classes:
        for batchnum, (images, labels) in enumerate(testdata):
            #print(f"Batch {batchnum} of {len(testdata)}, {len(ref_images)} ref images found")
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

@torch.no_grad()
def make_multiple_ref_images(model, testdata, num_classes=10, images_per_class=1):
    ref_images = { i: [] for i in range(num_classes) }
    #print(ref_images)
    #while len(ref_images) < num_classes and not all([len(ref_images[i]) >= images_per_class for i in range(num_classes)]):
    while not all([len(ref_images[i]) >= images_per_class for i in range(num_classes)]):
        for batchnum, (images, labels) in enumerate(testdata):
            #print(f"Batch {batchnum} of {len(testdata)}, {len(ref_images)} ref images found")
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            # For every correctly-predicted image, add it to the list of reference images
            # for that class (up to images_per_class)
            correct = preds == labels
            for i in torch.where(correct)[0]:
                label = labels[i].item()
                
                if len(ref_images[label]) < images_per_class:
                    
                    ref_images[label].append(images[i])
    #return ref_images
    #torch.stack(list([ref_images[i] for i in range(num_classes)]))
    """     ll=[]
    for i in range(num_classes):
        for j in range(images_per_class):
            print(ref_images)
            print(ref_images[i])
            ll.append(ref_images[i][j]) """

    return torch.stack(list([ref_images[i][j] for i in range(num_classes) for j in range(images_per_class)]))
    #return ref_images

def compute_linf(images1, images2):
    return torch.max(torch.abs(images1 - images2)).item()

def main():
    # Load the CIFAR10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Load the model
    model = make_normalized_resnet18k()
    model.load_state_dict(torch.load('resnet18_cifar10_nodp.pt'))
    model = model.to(device)
    model.eval()
