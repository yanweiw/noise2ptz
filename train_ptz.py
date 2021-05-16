import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset 
from CoordConv.coordconv import CoordConv2d
from aug_data import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Net(nn.Module):
    def __init__(self, device='cuda'):
        super(Net, self).__init__()
        self.resnet = models.resnet18()
        # self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.resnet.conv1 = CoordConv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False, 
                                        use_cuda=True, device=device)
        self.resnet.fc = nn.Sequential(nn.Linear(512, 3),
                                       nn.Sigmoid())

    def forward(self, x):
        out = self.resnet(x)
        return out


def load_data(base_dir, bsize=128, shuffle=True, overlap_chance=2/3):
    dataset_list = []
    env_list = os.listdir(base_dir)
    print('loading data from folders: ', env_list)
    for env in env_list:
        data_path = os.path.join(base_dir, env)
        dataset_list.append(AugDataset(data_path, overlap_chance=overlap_chance))
    
    dataset = ConcatDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=bsize, shuffle=shuffle, num_workers=8, pin_memory=True)
    return dataloader


def train(run, train_dir, test_dir, lr=0.001, bsize=128, num_epochs=1000, 
                                                  weight_path=None, data_parallel=False, overlap_chance=2/3): #, device=[0]):
    # model = Net(device=device[0]).cuda(device[0])
    model = Net().cuda()
    if weight_path:
        print('\nUsing pretrained weights...', weight_path)
        model.load_state_dict(torch.load(weight_path))        
    if data_parallel:
        model = nn.DataParallel(model)
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainloader = load_data(train_dir, bsize=bsize, shuffle=True, overlap_chance=overlap_chance)
    testloader = load_data(test_dir, bsize=bsize, shuffle=True, overlap_chance=overlap_chance)
    dataloaders = {'train': trainloader, 'test': testloader}

    # write to tensorboard images and model graphs
    if not os.path.exists('runs'):
        os.makedirs('runs') # create runs folder to hold indivdual runs
    writer = SummaryWriter('runs/'+str(run).zfill(3))
    print('\nrun num: ', run)
    print('lr: ', lr)
    print('bsize: ', bsize)
    print('training num ', len(dataloaders['train'].dataset))
    print('testing num ', len(dataloaders['test'].dataset))
    print()
    # record time
    since = time.time()
    last_time = since
    lowest_loss = 1e10 # some big number
    running_losses = {} 

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        loaders = {'train': trainloader, 'test': testloader}
        losses = {}
        for phase in ['train', 'test']:
            running_losses[phase] = 0.0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch_iter, batch_data in enumerate(loaders[phase]):
                img0, imgs, c1, c2, rc = batch_data
                imgs = imgs.cuda() #device[0])
                rc = rc.cuda() #device[0])

                with torch.set_grad_enabled(phase == 'train'):
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    out = model(imgs)
                    loss = criterion(out, rc)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_losses[phase] += loss.item() * imgs.size(0)
                writer.add_scalar(phase + '_loss', loss.item(), epoch*len(dataloaders[phase]) + batch_iter)

        # print average epoch loss
        epoch_loss = {}
        for phase in ['train', 'test']:
            ave_loss = running_losses[phase] / (len(dataloaders[phase].dataset))
            epoch_loss[phase] = ave_loss 
            print(phase + ' loss: {:.4f}'.format(ave_loss))    

        # record weights
        if epoch_loss['test'] < lowest_loss:
            lowest_loss = epoch_loss['test']
            print('updating best weights...')
            if not os.path.exists('weights'):
                os.makedirs('weights')
            save_path = 'weights/' + str(run).zfill(3) + '.pth'
            if data_parallel:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
        print('best test loss so far: {:.4f}'.format(lowest_loss))
   
        # save epoch weights
        if epoch % 10 == 9:
            print('saving epoch weights...')
            if not os.path.exists('weights'):
                os.makedirs('weights')
            save_path = 'weights/' + str(run).zfill(3) + '_lep.pth' # last epoch
            if data_parallel:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)

        # print running time
        curr_time = time.time()
        epoch_time = curr_time - last_time
        print('Current epoch time: {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))
        time_elapsed = curr_time - since
        print('Training so far takes {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                            time_elapsed % 60))
        last_time = curr_time
        print()
   
                    
    # finish training, now return weights
    print("training finished...")
    save_path = 'weights/' + str(run).zfill(3) + '.pth'
    print("model saved to ", save_path)
    print()

    writer.close()

    print('Finished Training')

    return model, dataloaders['test']


def eval(dataloader, model, pixel_tolerance=10, default_rc=None): # rc stands for relative coordinates
    location_error = []                                           # default for non-overlap is [0, 0, 0]
    ious = []
    l2loss = []

    model.eval()
    for data in dataloader:
        img0, imgs, c1, c2, rc = data
        if default_rc is not None:
            rc = torch.ones_like(rc)*torch.tensor(default_rc)
        pd = model(imgs.cuda()).detach().cpu()
        l2loss += nn.MSELoss(reduction='none')(pd, rc).sum(dim=1).sqrt().tolist()
        assert target_h == 128
        assert target_w == 128
        rc[:, 1] *= target_w
        rc[:, 0] *= target_h        
        pd[:, 1] *= target_w
        pd[:, 0] *= target_h        
        location_error += nn.MSELoss(reduction='none')(pd[:, :2], rc[:, :2]).sum(dim=1).sqrt().tolist()
        if default_rc is None: # predicting overlapping bounding boxes
            for i in range(len(rc)):
                bb1 = {}
                bb2 = {}
                bb1['x1'] = rc[i, 1] - rc[i, 2] * target_w / 2
                bb1['y1'] = rc[i, 0] - rc[i, 2] * target_h / 2
                bb1['x2'] = rc[i, 1] + rc[i, 2] * target_w / 2
                bb1['y2'] = rc[i, 0] + rc[i, 2] * target_h / 2
                
                bb2['x1'] = pd[i, 1] - pd[i, 2] * target_w / 2
                bb2['y1'] = pd[i, 0] - pd[i, 2] * target_h / 2
                bb2['x2'] = pd[i, 1] + pd[i, 2] * target_w / 2
                bb2['y2'] = pd[i, 0] + pd[i, 2] * target_h / 2
                ious.append(get_iou(bb1, bb2))
        # total_loss += l2loss * rc.size(0)
    # ave_loss = total_loss / len(dataloader.dataset)
    # print('average MSE loss: {:.4f}'.format(ave_loss.item()))
    location_error = np.asarray(location_error)
    on_target_rate = (location_error < pixel_tolerance).sum() / len(location_error)
    print('on target rate: {:.4f}'.format(on_target_rate))
    ious = np.asarray(ious)
    if default_rc is None: # predicting overlapping 
        print('ious: {:.4f}'.format(ious.mean()))
    l2loss = np.asarray(l2loss)
    print('l2loss: {:.4f}'.format(l2loss.mean()))
    return location_error, ious, l2loss


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--overlap', type=float, default=1)
    parser.add_argument('--weight', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--run', type=str)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bsize', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    if args.eval:
        loader = load_data('data/habitat_test', bsize=128, overlap_chance=1)
        model = Net()
        model.load_state_dict(torch.load(args.weight))
        model = model.cuda()
        eval(loader, model)
    elif args.train:
        train(args.run, args.train_dir, args.test_dir, args.lr, args.bsize, args.epochs, 
                args.weight, args.parallel, args.overlap)

