import argparse
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms, utils, models
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix
import itertools
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

import sys
from train_ptz import Net as ptz_net
from collections import OrderedDict
from aug_data import *

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)
torch.autograd.set_detect_anomaly(True)

class RandomCrop(object):
    """
    output_size (tuple)
    for data augmentation during training
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, images):
        h, w = images.size()[2:]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        images = images[:, :, top: top + new_h, left: left + new_w]
        return images

class CenterCrop(object):
    """
    output_size (tuple)
    for data augmentation during testing
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, images):
        h, w = images.size()[2:]
        new_h, new_w = self.output_size
        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)
        images = images[:, :, top: top + new_h, left: left + new_w]
        return images


# default data transform
default_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5])])



class NavDataset(Dataset):

    def __init__(self, dirname, max_seq_len, phase, img_h, img_w, 
                                size=None, transform=default_transform, with_lstm=True):
        self.dirname = dirname
        self.phase = phase
        self.data = np.loadtxt(dirname + '_action.txt', dtype=str)[1:50] # discard first action
        self.transform = transform
        self.size = size
        self.max_seq_len = max_seq_len
        self.img_h = img_h
        self.img_w = img_w
        self.labels = {'stop':         0,
                       'move_forward': 1,
                       'turn_left':    2,
                       'turn_right':   3}
        self.with_lstm = with_lstm 

    def __len__(self):
        if self.size: # in cases where I want to define size
            return self.size
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # random sample the starting location of the sub-traj 
        idx = random.choice(range(len(self.data)))
        seq_len = random.randint(1, self.max_seq_len)
        if idx + seq_len > len(self.data): 
            seq_len = len(self.data) - idx 
            assert seq_len > 0

        '''len(self.data) = num of images -1 '''
        actions = np.zeros(self.max_seq_len) # self.seq_len can be diff from seq_len
        for i in range(seq_len):
            actions[i] = self.labels[self.data[idx + i]]

        # random_scaling = random.uniform(1.1, 2)
        # new_h = int(random_scaling * self.img_h)
        # if new_h % 2 == 1:
        #     new_h = new_h + 1
        # pad_h = (new_h - self.img_h) // 2
        # new_w = int(random_scaling * self.img_w)
        # if new_w % 2 == 1:
        #     new_w = new_w + 1
        # pad_w = (new_w - self.img_w) // 2
        # images = torch.zeros((self.max_seq_len+1, 6, new_h, new_w))

        goal_img = Image.open(os.path.join(self.dirname, str(idx + seq_len).zfill(6) + '.png')).convert('RGB')
        new_w, new_h = goal_img.size
        images = torch.zeros((self.max_seq_len+1, 6, new_h, new_w))

        # if self.phase == 'train':
        #     goal_img = transforms.Resize((new_h, new_w))(goal_img)
        # else:
        #     goal_img = transforms.Pad((pad_w, pad_h))(goal_img)
        if self.transform:
            goal_img = self.transform(goal_img.copy())

        for i in range(seq_len+1):
            image_idx = idx + i
            img_name = os.path.join(self.dirname,  str(image_idx).zfill(6) + '.png')
            img = Image.open(img_name).convert('RGB')
            # if self.phase == 'train':
            #     # upscale to for random cropping 
            #     img = transforms.Resize((new_h, new_w))(img) 
            # else:
            #     img = transforms.Pad((pad_w, pad_h))(img) 

            if self.transform:
                img = self.transform(img.copy())

            images[i] = torch.cat((img, goal_img), dim=0)

        if self.phase == 'train':
            images = RandomCrop((self.img_h, self.img_w))(images) # images are tensors
        else:
            images = CenterCrop((self.img_h, self.img_w))(images)

        h = torch.zeros(1, 512)
        c = torch.zeros(1, 512)

        # estimated steps to goal
        etg = np.zeros(self.max_seq_len)
        etg[:seq_len] = np.arange(seq_len, 0, -1)

        if not self.with_lstm: # feedforward inputs a sequence of length 1
            images = images[:1] # notice here the goal and the current view can be multiple steps away
            seq_len = 1
            actions = actions[:1]
            etg = etg[:1]

        sample = {'images':   images,
                  'h':        h,
                  'c':        c,
                  'act_lens': seq_len,
                  'action':   torch.tensor(actions).long(),
                  'etg':      torch.tensor(etg).long()}
        return sample


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class invLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(invLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.inv = nn.Linear(hidden_dim, output_dim)
        # self.preproc = nn.Sequential(nn.Linear(input_dim, input_dim),
        #                              nn.ReLU(),
        #                              nn.Linear(input_dim, input_dim))
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, input, hidden):
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(input, hidden)
        out = self.inv(out.data)
        return out, hidden


class Siamese(nn.Module):

    def __init__(self, max_seq_len=20, state_size=128, action_size=4, act_embed_size=4, lstm_hidden_dim=512,
                 PTZ_weights=None, with_lstm=True, img_h=128, img_w=128):#, gpu_idx=0):
        super(Siamese, self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.act_embed_size = act_embed_size
        self.max_seq_len = max_seq_len
        self.img_h = img_h
        self.img_w = img_w
        if PTZ_weights is not None:
            self.state_size = 3
            self.base_model = ptz_net()#device=gpu_idx)
            if type(PTZ_weights) == str:
                state_dict = torch.load(PTZ_weights)
                if 'module' in list(state_dict.keys())[0]:
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]
                        new_state_dict[name] = v
                    state_dict = new_state_dict
                # loading weights
                print('loading PTZ weights from: ', PTZ_weights)
                self.base_model.load_state_dict(state_dict)
                # freeze weights in base_model
                for parameter in self.base_model.parameters():
                    parameter.requires_grad = False
        else:
            self.base_model = models.resnet18()
            self.base_model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
            self.base_model.fc = nn.Linear(512, self.state_size)

        self.with_lstm = with_lstm
        if self.with_lstm:
            print('using lstm....')
            self.inv_model = invLSTM(self.state_size + act_embed_size, lstm_hidden_dim, action_size, num_layers=1)
        else:
            print('using feedforward network...')
            self.inv_model = nn.Sequential(nn.Linear(self.state_size, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, action_size))

        # self.device = torch.device('cuda:{}'.format(gpu_idx) )


    def forward(self, images, act_lens=None, hidden=None, action=None, phase='train', prev_action=None):
        if self.with_lstm:
            h, c = hidden
            h = torch.transpose(h, 0, 1).contiguous()
            c = torch.transpose(c, 0, 1).contiguous()
            hidden = (h, c)

            assert images.size()[1:] == (self.max_seq_len+1, 6, self.img_h, self.img_w)
            states = self.base_model(images.view(-1, 6, self.img_h, self.img_w))
            states = states.view(-1, self.max_seq_len+1, self.state_size)
        else: # single action sequence
            states = self.base_model(images[:, 0])
            assert states.size()[1:] == (self.state_size,)

        if phase == 'infer':
            if self.with_lstm:
                assert prev_action.size()[1:] == (self.max_seq_len,)
                prev_action_vec = nn.functional.one_hot(prev_action, self.action_size).float()
                assert prev_action_vec.size() == (1, 1, self.act_embed_size)
                state_pair_act = torch.cat([states[:, [0]], prev_action_vec], dim=2)
                action_pd, hidden = self.inv_model(state_pair_act, hidden)
            else:
                action_pd = self.inv_model(states)
            return action_pd, hidden

        if self.with_lstm:
            assert action.size()[1:] == (self.max_seq_len,)
            action_vec = nn.functional.one_hot(action, self.action_size).float()
            assert action_vec.size()[1:] == (self.max_seq_len, self.action_size)
            #prev_action
            prev_action = torch.cat((torch.zeros(action.size(0), 1).long().cuda(), action[:, :-1]), dim=1)
            prev_action_vec = nn.functional.one_hot(prev_action, self.action_size).float()
            assert prev_action_vec.size() == action_vec.size()
            # predict action
            assert states.size()[1:] == (self.max_seq_len+1, self.state_size)
            state_pair_act = torch.cat((states[:, :-1], prev_action_vec), dim=2)
            packed_state_pair_act = pack_padded_sequence(state_pair_act, act_lens.cpu(), batch_first=True, enforce_sorted=False)
            out_act_pd, _ = self.inv_model(packed_state_pair_act, hidden)
            packed_act_pd = PackedSequence(out_act_pd, packed_state_pair_act.batch_sizes,
                                               packed_state_pair_act.sorted_indices,
                                               packed_state_pair_act.unsorted_indices)
            act_pd, _ = pad_packed_sequence(packed_act_pd, batch_first=True, total_length=self.max_seq_len)
            assert act_pd.size()[1:] == (self.max_seq_len, self.action_size)
            # mask out everything
            assert act_pd.size() == action_vec.size()
            mask = torch.arange(self.max_seq_len)[None, :].cuda() < act_lens[:, None]
            act_pd = act_pd[mask]
            act_gt = action_vec[mask]
            assert act_pd.size() == act_gt.size()
        else:
            act_pd = self.inv_model(states)
            act_gt = nn.functional.one_hot(action.squeeze(), self.action_size).float()

        return act_pd, act_gt, 


def load_data(base_dir, train_envs, valid_envs, train_dirs, valid_dirs, seq_len, bsize, 
                img_h, img_w, num_of_location=100, with_lstm=True):
    '''
    load_data from 'base_dir/env_dir/data_dir', e.g. 'data/train/Delton/rob0'
    env_dirs, train_dirs, and valid_dirs are list of dirs
    train_dirs ['rob1'], valid_dirs ['rob2']
    '''
    env_dirs = {'train': train_envs, 'valid': valid_envs}
    data_dirs = {'train' : train_dirs, 'valid' : valid_dirs}
    base_dirs = {'train': base_dir, 'valid': base_dir}

    datasets = {'train': [], 'valid': []}
    dataloaders = {}
    for phase in ['train', 'valid']:
        if env_dirs[phase] is None:
            env_dirs[phase] = ['Delton', 'Goffs', 'Oyens', 'Placida', 'Sumas', 'Crandon', 'Roane', 'Springhill', 'Superior', 'Woonsocket']
                        # 'valid': ['Hambleton', 'Eastville', 'Pettigrew', 'Albertville', 'Hometown']}
        for env_dir in env_dirs[phase]:
            env_path = os.path.join(base_dirs[phase], env_dir)
            for data_dir in data_dirs[phase]:
                d_list = sorted(os.listdir(os.path.join(env_path, data_dir))) # list of image folders
                if phase == 'train': # num_of_location < 100
                    image_d_list = [d for d in d_list if ('_orig' not in d) and ('.txt' not in d)][:num_of_location]
                else:
                    image_d_list = [d for d in d_list if ('_orig' not in d) and ('.txt' not in d)]
                for d in image_d_list: 
                    #################################
                    # only sample 1 subtraj as defined by size
                    #################################
                    datasets[phase].append(NavDataset(os.path.join(env_path, data_dir, d), 
                        max_seq_len=seq_len, phase=phase, img_h=img_h, img_w=img_w, size=1, with_lstm=with_lstm))
        concatsets = ConcatDataset(datasets[phase])
        dataloaders[phase] = DataLoader(concatsets, batch_size=bsize, shuffle=True, num_workers=8)

    print('loading data from folders: ', env_dirs)
    return dataloaders



def train(run, base_dir, train_envs, valid_envs, train_dirs, valid_dirs, seq_len=None, PTZ_weights=None, 
            data_parallel=True, lr=0.0001, bsize=32, num_epochs=20, pretrained_weights=None, num_of_location=100, 
            state_size=128, img_h=128, img_w=128, with_lstm=True, device=0):
            # gpu_idx=0):
    # device = torch.device('cuda:{}'.format(gpu_idx) )

    # select training policy
    dataloaders = load_data(base_dir, train_envs, valid_envs, train_dirs, valid_dirs, seq_len, bsize, 
                            img_h, img_w, num_of_location, with_lstm=with_lstm)

    model = Siamese(max_seq_len=seq_len, PTZ_weights=PTZ_weights, state_size=state_size, 
                with_lstm=with_lstm, img_h=img_h, img_w=img_w)#, gpu_idx=gpu_idx)
        
    inv_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.float().cuda(device)
    if pretrained_weights:
        print('\nUsing pretrained weights...', pretrained_weights)
        model.load_state_dict(torch.load(pretrained_weights))
    if data_parallel:
        model = torch.nn.DataParallel(model)
        print('\nUsing multiple gpus...')
    if PTZ_weights is not None:
        print('\nfreezing runing stats of batchnorms...')
    else:
        print('\n NOT freezing running stats of batchnorms...')


    # write to tensorboard images and model graphs
    if not os.path.exists('runs'):
        os.makedirs('runs') # create runs folder to hold indivdual runs
    writer = SummaryWriter('runs/'+str(run).zfill(3))
    print('\nrun num: ', run)
    print('lr: ', lr)
    print('bsize: ', bsize)
    print('seq_len ', seq_len)
    print('training num ', len(dataloaders['train'].dataset) * 50)  # num of locations * datapoints per location
    print('validation num ', len(dataloaders['valid'].dataset) * 50)
    print('max bsize: ', len(dataloaders['train'].dataset))
    print()
    # record time
    since = time.time()
    last_time = since
    lowest_loss = 1e10 # some big number
    running_inv_losses = {} 


    # running through epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            running_inv_losses[phase] = 0.0
            if phase == 'train':
                model.train()
                #################################################################
                # we want to freeze the running_stats of batchnorms in base model
                if PTZ_weights is not None:
                    for module in model.modules():
                        if isinstance(module, nn.BatchNorm2d):
                            _ = module.eval()
                #################################################################
            else:
                model.eval()

            # iterate over data
            for batch_iter, batched_sample in enumerate(dataloaders[phase]):
                h = batched_sample['h'].cuda(device)
                c = batched_sample['c'].cuda(device)
                act_lens = batched_sample['act_lens'].cuda(device)
                images = batched_sample['images'].cuda(device)
                action = batched_sample['action'].cuda(device)

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    act_pd, act_gt = model(images, act_lens, (h, c), action)
                    _, label = act_gt.max(dim=1)
                    loss = inv_criterion(act_pd, label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # gradient clipping
                        # nn.utils.clip_grad_value_(model.parameters(), 40)

                # log statistics
                running_inv_losses[phase] += loss.item() * images.size(0)
                writer.add_scalar(phase + '_inv_loss', loss.item(),
                    epoch*len(dataloaders[phase]) + batch_iter)


        # print average epoch loss
        epoch_loss = {}
        for phase in ['train', 'valid']:
            ave_loss = running_inv_losses[phase] / (len(dataloaders[phase].dataset))
            epoch_loss[phase] = ave_loss 
            print(phase + ' loss: {:.4f}'.format(ave_loss))


        # record weights
        if epoch_loss['valid'] < lowest_loss:
            lowest_loss = epoch_loss['valid']
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

    return model, dataloaders['valid']


def inspect_data(dataiter, num=20):
    data = dataiter.next()
    images = data['images']
    action = data['action']
    images = images.permute(0, 1, 3, 4, 2) / 2 + 0.5 
    for i in range(num): 
        plt.figure(figsize=(30, 2)) 
        for j in range(len(action[i])): 
            ax = plt.subplot(1, 16, j+1)  
            plt.tight_layout() 
            plt.imshow(images[i, j][:, :, :3])
            if action[i, j] == 0:
                atext = 'stop'
            elif action[i, j] == 1:
                atext = 'forward'
            elif action[i, j] == 2:
                atext = 'left'
            else:
                atext = 'right'
            plt.title(atext)  


def pred_bbox(dataiter, model=None, max_seq_len=15, state_size=3, img_h=128, img_w=128):
    data = dataiter.next()
    images = data['images']
    action = data['action']
    etg = data['etg']
    # bbox = model.base_model(images.view(-1, 6, 120, 160)).view(-1, max_seq_len+1, state_size)
    if model is not None:
        bbox = model(images.view(-1, 6, img_h, img_w)).view(-1, max_seq_len+1, state_size)

    # images = images.permute(0, 1, 3, 4, 2) / 2 + 0.5

    for i in range(len(images)):
        fig = plt.figure(figsize=(30, 4))
        plt.tight_layout()
        gs = gridspec.GridSpec(2, 16, hspace=0, wspace=0.1)
        for j in range(len(action[i])):
            im0 = tensor2np(images[i, j, :3])
            ax = setup_ax(fig, gs[0, j])
            ax.imshow(im0)
            if model is not None:
                draw_bbox(ax, (bbox[i, j, 0]*img_h, bbox[i, j, 1]*img_w), 'r', scaling=bbox[i, j, -1], 
                            target_w=img_w, target_h=img_h)
                ax.set_title([round(bbox[i, j][0].item(), 2), round(bbox[i, j][1].item(), 2), 
                          round(bbox[i, j][2].item(), 2)], fontsize=8)

            im1 = tensor2np(images[i, j, 3:])
            ax = setup_ax(fig, gs[1, j])
            ax.imshow(im1)
            if action[i, j] == 0:
                atext = 'stop'
            elif action[i, j] == 1:
                atext = 'forward'
            elif action[i, j] == 2:
                atext = 'left'
            else:
                atext = 'right'
            ax.set_title(atext+' '+str(etg[i, j].item()), fontsize=8)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--base_dir', type=str, default='data/nav_train')
    parser.add_argument('--train_envs', nargs='+', default=None)
    parser.add_argument('--valid_envs', nargs='+', default=None)
    parser.add_argument('--train_dirs', nargs='+', default=['nav1'])
    parser.add_argument('--valid_dirs', nargs='+', default=['nav0'])
    parser.add_argument('--seq_len', type=int, default=15)
    parser.add_argument('--PTZ_weights', type=str, default=None)
    parser.add_argument('--data_parallel', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bsize', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--num_loc', type=int, default=100, help='number of locations used for training, controlling training data size')
    parser.add_argument('--state_size', type=int, default=128)
    parser.add_argument('--img_h', type=int, default=128)
    parser.add_argument('--img_w', type=int, default=128)
    parser.add_argument('--with_lstm', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    train(run=args.run, base_dir=args.base_dir, train_envs=args.train_envs, valid_envs=args.valid_envs, 
            train_dirs=args.train_dirs, valid_dirs=args.valid_dirs, seq_len=args.seq_len, PTZ_weights=args.PTZ_weights, 
            data_parallel=args.data_parallel, lr=args.lr, bsize=args.bsize, num_epochs=args.num_epochs, 
            pretrained_weights=args.pretrained_weights, num_of_location=args.num_loc, state_size=args.state_size,
            img_h=args.img_h, img_w=args.img_w, with_lstm=args.with_lstm, device=args.device)