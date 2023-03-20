import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet18k import make_resnet18k
from tqdm import tqdm

device = torch.device('cuda')

# Hyper-parameters
num_epochs = 80
learning_rate = 0.0001
BATCH_SIZE = 1000

# Augmentations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std

class NormalizedResNet18(nn.Module):
    def __init__(self, norm, k=64, num_classes=10):
        super(NormalizedResNet18, self).__init__()
        self.model = make_resnet18k(k=k, num_classes=num_classes)
        self.normalizer = norm

    def forward(self, x):
        x = self.normalizer(x)
        x = self.model(x)
        return x

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='/fastdata/cifar10/',
                                             train=True,
                                             download=True,
                                             transform=transform_train)

test_dataset = torchvision.datasets.CIFAR10(root='/fastdata/cifar10/',
                                            train=False,
                                            download=True,
                                            transform=transform_test)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


model = make_resnet18k(k=64, num_classes=10).to(device)
model = torch.nn.DataParallel(model)

# Compile the model
model_opt = torch.compile(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)

def val(epoch):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'[{epoch}] Test Accuracy of the model on the {total} test images: {100 * correct / total} %')
    model.train()

# Train the model
model.train()
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    scheduler.step()
    if epoch % 5 == 0:
        val(epoch)
        # Save the model checkpoint
        torch.save(model.state_dict(), f'resnet18_cifar10_{epoch}.pt')

val('final')
# Save final model
torch.save(model.state_dict(), 'resnet18_cifar10.pt')

