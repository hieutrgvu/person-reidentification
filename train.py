from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
import matplotlib.pyplot as plt
import time
import os
from model import ft_net, ft_net_dense, ft_net_NAS, PCB
from random_erasing import RandomErasing
import yaml
from shutil import copyfile
from circle_loss import CircleLoss, convert_label_to_similarity

matplotlib.use('agg')

version = torch.__version__

# Options
parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--gpu-id', default='0', type=str, help='gpu_id: e.g. 0, 1, ...')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data-dir', default='./market/pytorch', type=str, help='training dir path')
parser.add_argument('--color-jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erase-prob', default=0, type=float, help='random erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--circle', action='store_true', help='use Circle loss')
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

print(transform_train_list)
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
        num_workers=2,
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

# train the model
y_loss = {'train': [], 'val': []}  # loss history
y_err = {'train': [], 'val': []}


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train'] / opt.batch_size) * opt.warm_epoch  # first 5 epoch
    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in data_loaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batch_size:  # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # if we use low precision, input also need to be fp16
                # if fp16:
                #    inputs = inputs.half()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                sm = nn.Softmax(dim=1)
                if opt.circle:
                    logits, ff = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels) + criterion_circle(
                        *convert_label_to_similarity(ff, labels)) / now_batch_size
                    # loss = criterion_circle(*convert_label_to_similarity( ff, labels))
                    _, preds = torch.max(logits.data, 1)
                else:  # PCB
                    part = {}
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part - 1):
                        loss += criterion(part[i + 1], labels)

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss = loss * warm_up

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
# ---------------------------
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


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_id)


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if opt.use_dense:
    model = ft_net_dense(len(class_names), opt.droprate, circle=opt.circle)
elif opt.use_NAS:
    model = ft_net_NAS(len(class_names), opt.droprate)
else:
    model = ft_net(len(class_names), opt.droprate, opt.stride, circle=opt.circle)

model = PCB(len(class_names))

opt.nclasses = len(class_names)

print(model)

ignored_params = list(map(id, model.model.fc.parameters()))
ignored_params += (list(map(id, model.classifier0.parameters()))
                   + list(map(id, model.classifier1.parameters()))
                   + list(map(id, model.classifier2.parameters()))
                   + list(map(id, model.classifier3.parameters()))
                   + list(map(id, model.classifier4.parameters()))
                   + list(map(id, model.classifier5.parameters()))
                   # +list(map(id, model.classifier6.parameters() ))
                   # +list(map(id, model.classifier7.parameters() ))
                   )
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1 * opt.lr},
    {'params': model.model.fc.parameters(), 'lr': opt.lr},
    {'params': model.classifier0.parameters(), 'lr': opt.lr},
    {'params': model.classifier1.parameters(), 'lr': opt.lr},
    {'params': model.classifier2.parameters(), 'lr': opt.lr},
    {'params': model.classifier3.parameters(), 'lr': opt.lr},
    {'params': model.classifier4.parameters(), 'lr': opt.lr},
    {'params': model.classifier5.parameters(), 'lr': opt.lr},
    # {'params': model.classifier6.parameters(), 'lr': 0.01},
    # {'params': model.classifier7.parameters(), 'lr': 0.01}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name, exist_ok=True)
# record every run
copyfile('./train.py', dir_name + '/train.py')
copyfile('./model.py', dir_name + '/model.py')

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
criterion = nn.CrossEntropyLoss()

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=60)
