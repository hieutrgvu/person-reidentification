from __future__ import print_function, division

from model import PCB
from random_erasing import RandomErasing
from shutil import copyfile
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

matplotlib.use('agg')

version = torch.__version__

# Options
parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--gpu-id', default='0', type=str, help='gpu_id: e.g. 0, 1, ...')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data-dir', default='./market/pytorch', type=str, help='training dir path')
parser.add_argument('--color-jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--erase-prob', default=0, type=float, help='random erasing probability, in [0,1]')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
gpu_id = int(opt.gpu_id)

# set gpu ids
if len(opt.gpu_id) > 0:
    torch.cuda.set_device(gpu_id)
    cudnn.benchmark = True

# Load Data
transform_train_list = [
    transforms.Resize((384, 192), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
transform_val_list = [
    transforms.Resize(size=(384, 192), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erase_prob > 0:
    transform_train_list += [RandomErasing(probability=opt.erase_prob, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list += [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)]

print('dataset transformation: ', transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

image_datasets = {
    key: datasets.ImageFolder(os.path.join(data_dir, key), value)
    for key, value in data_transforms.items()
}

data_loaders = {
    key: torch.utils.data.DataLoader(
        value,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    for key, value in image_datasets.items()
}

dataset_sizes = {k: len(image_datasets[k]) for k in image_datasets.keys()}
print('data size:', dataset_sizes)

class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()
print('using gpu: ', use_gpu)

since = time.time()
inputs, classes = next(iter(data_loaders['train']))
print('loading time (s):', time.time() - since)

# loss history
y_loss = {'train': [], 'val': []}
y_err = {'train': [], 'val': []}


# define model training
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # set model to training or evaluation mode
            model.train(phase == 'train')

            running_loss = 0.0
            running_corrects = 0.0

            # iterate over data.
            for data in data_loaders[phase]:
                # get the inputs
                inputs, labels = data
                batch_size, c, h, w = inputs.shape
                if batch_size < opt.batch_size:
                    # skip the last batch
                    continue

                # wrap in variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                sm = nn.Softmax(dim=1)

                parts, num_parts = {}, 6
                for i in range(num_parts):
                    parts[i] = outputs[i]

                score = sum([sm(v) for v in parts.values()])
                _, predict = torch.max(score.data, 1)
                loss = sum([criterion(v, labels) for v in parts.values()])

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:
                    # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * batch_size
                else:
                    # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * batch_size
                running_corrects += float(torch.sum(predict == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)

            if phase == 'train':
                scheduler.step()

            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('All training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


# draw curve
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


# save model
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_id)


# fine-tuning the conv net: load a pre-trained model and reset final fully connected layer.
model = PCB(len(class_names))
print('model: ', model, '\n')

ignored_params = list(map(id, model.model.fc.parameters()))
ignored_params += (list(map(id, model.classifier0.parameters()))
                   + list(map(id, model.classifier1.parameters()))
                   + list(map(id, model.classifier2.parameters()))
                   + list(map(id, model.classifier3.parameters()))
                   + list(map(id, model.classifier4.parameters()))
                   + list(map(id, model.classifier5.parameters()))
                   )
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1 * opt.lr},
    {'params': model.model.fc.parameters()},
    {'params': model.classifier0.parameters()},
    {'params': model.classifier1.parameters()},
    {'params': model.classifier2.parameters()},
    {'params': model.classifier3.parameters()},
    {'params': model.classifier4.parameters()},
    {'params': model.classifier5.parameters()},
], lr=opt.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)

# decay learning rate by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

# record every run
dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
copyfile('./train.py', dir_name + '/train.py')
copyfile('./model.py', dir_name + '/model.py')

# save opts
opt.num_classes = len(class_names)
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
criterion = nn.CrossEntropyLoss()

# train and evaluate
train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=60)
