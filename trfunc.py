# coding: utf-8
# Early stopping refer to
# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/


#—————— Workflow 0: import & initialize ——————#
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import torch
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
#device = torch.device('cpu')
dtype = torch.FloatTensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import datetime
from config import *
from model_k60 import TPhaseNet
model = TPhaseNet().to(device)
#model.train()
print('lr=',lr,'\n','epochs=',epochs)
print('Using device: ', device)
import csv


#—————— Workflow 1: def & class ——————#
def plotprob(sample):
	fig = plt.figure(figsize=(15, 10))
	ax = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace':0})
	npts = sample['input'].shape[1]
	t = np.arange(npts) / 100
	ax[0].plot(t, sample['input'][0], label='hydrophone')
	#ax[0].plot(t, sample['input'][1], label='N')
	#ax[0].plot(t, sample['input'][2], label='E')
	ax[0].legend(loc='upper right')
	ax[1].plot(t, sample['target'][0], label='target T', ls='--')
	ax[1].legend(loc='upper right')
	plt.show()
	return

# detrend & normailize
def routine(waveform):
        demeaned = waveform - torch.mean(waveform)
        #detrended = signal.detrend(demeaned, axis=1)
        normalized = demeaned / np.max(np.std(np.array(demeaned)))
        return normalized

def gaus(width=15):
	label = np.exp(-((np.arange(-width // 2, width // 2 + 1)) ** 2)
	/ (2 * (width / 5) ** 2))
	return label

# 'cut' function temporarily not needed
# def cut(sample, p, s, mode='train', length=3001, halfwin=15):
# 	maxn = sample['input'].shape[1]
# 	if mode=="train":
# 		select_range = [s-p-length+halfwin, -halfwin] # make sure both P&S picks involved
# 		shift = np.random.randint(*select_range)
# 	elif mode=="test":
# 		shift = int((p+s-length)/2) - p
# 	sample = {x:sample[x][:,shift+p:shift+p+length] for x in sample}
# 	sample['p'] = -shift
# 	sample['s'] = s-p-shift
# 	sample['fname'] = fname
# 	return sample

# halfwin: 30->15
def mklabel(ts, shape, halfwin=750):
        target = np.zeros(shape)
        maxn = shape[1]
        for i in range(len(ts)):
                #print(ts[i],type(ts[i]))
                t = int(ts[i])
                #print(t_int, type(t_int))
                # order by PSN
                target[0, max(0,t-halfwin):min(t+(halfwin+1),maxn)] = gaus(2*halfwin)[max(halfwin-t,0):2*halfwin+1-max(t+(halfwin+1)-maxn,0)] # a half of the label is 15
                # target[1, max(0,s-halfwin):min(s+(halfwin+1),maxn)] = gaus(2*halfwin)[max(halfwin-s,0):2*halfwin+1-max(s+(halfwin+1)-maxn,0)]
                # target[2,...] = 1 - target[0,...] - target[1,...]
        target[1,...] = 1 - target[0,...]
        return torch.tensor(np.array(target))

def cut(sample, halfwin, length=25000, mode='train', ts=np.array([25000])):
        maxn = sample['input'].shape[1]
        if mode=="train":
                flag = 1
                while flag == 1 :
                    pick_n = len(ts)   # the number of picks
                    r = np.random.randint(0,pick_n)   # select one pick randomly
                    arr = ts[r]   # for the select_range

                    select_range = [max(0,arr+halfwin-length), min(arr-halfwin,maxn-length)]   # the select_range of the cut-start 

                    try:
                            onset = np.random.randint(*select_range)
                            flag = 0
                    except:
                            #print(select_range, sample['fname'])
                            flag = 1
                for x in ['input', 'target']:
                    sample[x] = sample[x][:,onset:onset+length]

        elif mode=="test":
                array1 = sample['input'][:,:length]
                target1 = sample['target'][:,:length]
                array2 = sample['input'][:,12500:12500+length]
                target2 = sample['target'][:,12500:12500+length]
                array3 = sample['input'][:,maxn-length:maxn]
                target3 = sample['target'][:,maxn-length:maxn]
                sample['array1'] = array1
                sample['target1'] = target1
                sample['array2'] = array2
                sample['target2'] = target2
                sample['array3'] = array3
                sample['target3'] = target3
        elif mode=="testlong":
                n = maxn//int(200*62.5) 
                array = []
                if maxn%int(200*62.5)==0:
                        n = n-1
                for i in range(n-1):
                        head = i*12500 - i*1
                        array.append(sample['input'][:,head:head+length])
                # i = n-1
                array.append(sample['input'][:,maxn-length:maxn])
                sample['array'] = array
        return sample


class Dataset(torch.utils.data.Dataset):
	"""Substantiate a dataset"""
	def __init__(self, dfile, data_folder, transform=None, plot=False, mode='train', sigma = 750):
                self.dfile = pd.read_csv(dfile)
                self.data_folder = data_folder
                self.transform = transform
                self.plot = plot
                self.mode = mode
                self.sigma = sigma

	def __len__(self):
		return len(self.dfile)

	def __getitem__(self, idx):
                if torch.is_tensor(idx):
                        idx = idx.tolist()
		# p, s = (self.dfile.iloc[idx]['itp'],
		# 	self.dfile.iloc[idx]['its'])
                t_str = self.dfile.iloc[idx]['itt']
                t_list = t_str[1:-1].split(' ')
                tss = np.array([x.strip() for x in t_list if x.strip()!=''])
                ts = np.array([int(x) for x in tss])
                #print(ts,type(ts),ts[0],type(ts[0]))
                npz = np.load('%s/%s'%(self.data_folder,
					self.dfile.iloc[idx]['fname']))
                input = torch.tensor(npz['data'].transpose()) # [channel, feature]
                output_shape = torch.Size([2,input.shape[1]])
                target = torch.tensor(np.zeros(output_shape))
                #for i in range(len(ts)):
                        #print(ts[i],type(ts[i]))
                #        t_int = int(ts[i])
                        #print(t_int, type(t_int))
                target = target + mklabel(ts, output_shape)

		# print('the size of input_tensor is ' + str(input.shape))
		# output_shape = torch.Size([3,701])
		# order by ZNE
		# input = torch.flip(input, [0])
		# detrend&normalization

                input = routine(input)
		# target = mklabel(p, s, input.shape)

                halfwin = self.sigma

                sample = {'input': input, 'target': target}
                sample['fname'] = self.dfile.iloc[idx]['fname']
                cutsample = cut(sample, halfwin, mode=self.mode, ts=ts)
                if self.transform:
                        sample = self.transform(sample)
                if self.plot:
                        plotprob(sample)
                return cutsample
                #return sample

def loss_fn(input, target, eps=1e-5):
	h = target * torch.log(input + eps)
	h = h.mean(-1).sum(-1)
	h = h.mean()
	return -h

def fit(model, train_dataloader, optimizer, criterion):
	model.train()
	train_running_loss = 0.0
	counter = 0
	total = 0
	prog_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
	for i, data in prog_bar:
		counter += 1
		data, target = data['input'].type(dtype).to(device), data['target'].type(dtype).to(device)
		total += target.size(0)
		optimizer.zero_grad()
		outputs = model(data)
		loss = criterion(outputs, target)
		train_running_loss += loss.item()
		loss.backward()
		optimizer.step()
		prog_bar.set_postfix(loss=loss.item())
	train_loss = train_running_loss / counter
	return train_loss

def validate(model, test_dataloader, criterion):
	model.eval()
	val_running_loss = 0.0
	counter = 0
	total = 0
	prog_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
	with torch.no_grad():
		for i, data in prog_bar:
			counter += 1
			data, target = data['input'].type(dtype).to(device), data['target'].type(dtype).to(device)
			total += target.size(0)
			outputs = model(data)
			loss = criterion(outputs, target)
			val_running_loss += loss.item()
			prog_bar.set_postfix(loss=loss.item())
		val_loss = val_running_loss / counter
	return val_loss

class LRScheduler():
	"""
	Learning rate scheduler. If the validation loss does not decrease for the
	given number of `patience` epochs, then the learning rate will decrease by
	by given `factor`.
	"""
	def __init__(self, optimizer, patience=7, min_lr=1e-6, factor=0.5):
		"""
		new_lr = old_lr * factor
		:param optimizer: the optimizer we are using
		:param patience: how many epochs to wait before updating the lr
		:param min_lr: least lr value to reduce to while updating
		:param factor: factor by which the lr should be updated
		"""
		self.optimizer = optimizer
		self.patience = patience
		self.min_lr = min_lr
		self.factor = factor
		self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				self.optimizer,
				mode='min',
				patience=self.patience,
				factor=self.factor,
				min_lr=self.min_lr,
				verbose=True
			)
	def __call__(self, val_loss):
		self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
            not improving
        :param min_delta: minimum difference between new loss and old loss for
            new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0 # It determines how many epochs has the validation loss failed decreasing for
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class SaveBestModel:
	"""
	Class to save the best model while training. If the current epoch's
	validation loss is less than the previous least less, then save the
	model state.
	"""
	# def __init__(self, path='/model/best_model.pth', best_valid_loss=float('inf')):
	def __init__(self, path=os.getcwd()+'/best_model.pth', best_valid_loss=float('inf')):
		self.best_valid_loss = best_valid_loss
		self.path = path

	def __call__(self, current_valid_loss, epoch, model):
		if current_valid_loss < self.best_valid_loss:
			self.best_valid_loss = current_valid_loss
			print(f"Saving best model for epoch: {epoch+1}")
			torch.save(model, self.path)

def plotloss(train_loss, val_loss):
	plt.plot(train_loss, label='train')
	plt.plot(val_loss, label='val')
	plt.legend()
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	# plt.show()
	plt.savefig('loss_curv_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")+'.jpg')
	return


def my_mklabel(itt, shape, halfwin=750):
        target = np.zeros(shape)
        maxn = shape[1]
        for i in range(len(itt)):
                t = int(itt[i])
                # order by PSN
                target[0, max(0,t-halfwin):min(t+(halfwin+1),maxn)] = gaus(2*halfwin)[max(halfwin-t,0):2*halfwin+1-max(t+(halfwin+1)-maxn,0)] # a half of the label is 15
                #target[1, max(0,s-halfwin):min(s+(halfwin+1),maxn)] = gaus(2*halfwin)[max(halfwin-s,0):2*halfwin+1-max(s+(halfwin+1)-maxn,0)]
        target[1,...] = 1 - target[0,...]
        return np.array(target)



def collectname(filelist):
    name = []
    with open(filelist, 'r') as f1 :
        reader1 = csv.reader(f1)
        for i in reader1:
            name.append(i[0])
    return name[1:]


if __name__ == "__main__":
        data_folder = '/home/lilab/hqzhu/data/800npz'
        testlist = data_folder+'/testlist.csv'
        test_name = collectname(testlist)
        print(len(test_name))
        #test_loader = DataLoader(Dataset(testflist, data_folder, plot=False, mode='train')
        x = np.random.randint(0,len(testlist))
        filename = test_name[x]
        npz = np.load(data_folder + '/' + filename)
        input = torch.tensor(npz['data'].transpose())
        itt = npz['itt']
        
        output_shape = torch.Size([2,input.shape[1]])
        label_array = np.zeros(output_shape)
        label_array = label_array + my_mklabel(itt, output_shape)

        sample = {'input': input, 'target': label_array}
        sample['fname'] = filename
        cutsample = cut(sample, 850, mode='test', ts=itt)
        print(cutsample['array1'].shape, cutsample['array2'].shape, cutsample['array3'].shape)
