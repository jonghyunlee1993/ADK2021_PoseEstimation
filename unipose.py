# -*-coding:UTF-8-*-
import argparse
import torch
import torch.nn as nn

import torch.optim
import torch.cuda.amp as amp

import os
import json
import time
import numpy as np

from utils.utils import adjust_learning_rate as adjust_learning_rate
from utils.utils import printAccuracies      as printAccuracies
from utils.utils import getDataloader        as getDataloader
from utils.utils import get_kpts             as get_kpts
from utils       import evaluate             as evaluate
from model.unipose import unipose

from tqdm import tqdm
from torchsummary import summary

starter_epoch =   0
epochs        =   50

# modify the weight key to use the weights in single GPU
def remove_prefix(state_dict, prefix):
    print("remove prefix \'{}\'".format(prefix))

    # lambda function
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    # modify key name
    return {f(key): value for key, value in state_dict.items()}

class Trainer(object):
    def __init__(self, args):
        self.args         = args
        self.train_dir    = args.train_dir
        self.val_dir      = args.val_dir
        self.test_dir     = args.val_dir
        self.model_arch   = args.model_arch
        self.dataset      = args.dataset


        self.workers      = 2
        self.weight_decay = 0.0005
        self.momentum     = 0.9
        self.batch_size   = 16
        self.lr           = 0.0001
        self.gamma        = 0.333
        self.step_size    = 13275
        self.sigma        = 3
        self.stride       = 8
        self.numClasses  = 17

        self.train_loader, self.val_loader, self.test_loader = getDataloader(
            train_dir=self.train_dir, 
            val_dir=self.val_dir, 
            test_dir=self.val_dir,
            sigma=self.sigma, 
            workers=self.workers,
            batch_size=self.batch_size)

        model = unipose(self.dataset, num_classes=self.numClasses,backbone='resnet',output_stride=16,sync_bn=True,freeze_bn=False, stride=self.stride)
        summary(model, input_size=(3, 368, 368), batch_size=2, device="cpu")

        if torch.cuda.device_count() > 1:
            print("\n===> Detect %d GPUs! Training with multi-GPU." % torch.cuda.device_count())
            model = nn.DataParallel(model)

        self.model       = model.cuda()

        self.criterion   = nn.MSELoss().cuda()

        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.scaler = amp.GradScaler(enabled=True)

        self.best_model  = 12345678.9

        self.iters       = 0

        if self.args.pretrained is not None:
            print("===> Loaded pretrained weight to the model.\n")
            checkpoint = torch.load(self.args.pretrained)
            p = checkpoint['state_dict']

            if torch.cuda.device_count() == 1:
                p = remove_prefix(p, "module.")

            state_dict = self.model.state_dict()
            model_dict = {}

            for k,v in p.items():
                if k in state_dict:
                    model_dict[k] = v

            state_dict.update(model_dict)
            self.model.load_state_dict(state_dict, strict=False)

        self.isBest = 0
        self.bestPCK  = 0
        self.bestPCKh = 0


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        print("Epoch " + str(epoch) + ':') 
        tbar = tqdm(self.train_loader)

        for i, (input, heatmap, centermap, img_path) in enumerate(tbar):
            learning_rate = adjust_learning_rate(self.optimizer, self.iters, self.lr, policy='step',
                                                 gamma=self.gamma, step_size=self.step_size)

            input_var     =      input.cuda()
            heatmap_var   =    heatmap.cuda()

            self.optimizer.zero_grad()

            # AMP
            with amp.autocast(enabled=True):
                heat = self.model(input_var)
                loss_heat   = self.criterion(heat,  heatmap_var)

            loss = loss_heat
            train_loss += loss_heat.item()

            # AMP
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer=self.optimizer)
            self.scaler.update()

            tbar.set_description('Train loss: %.6f' % (train_loss / ((i + 1)*self.batch_size)))

            self.iters += 1


    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        
        AP    = np.zeros(self.numClasses+1)
        PCK   = np.zeros(self.numClasses+1)
        PCKh  = np.zeros(self.numClasses+1)
        count = np.zeros(self.numClasses+1)

        cnt = 0
        for i, (input, heatmap, centermap, img_path) in enumerate(tbar):

            cnt += 1

            input_var     =      input.cuda()
            heatmap_var   =    heatmap.cuda()
            self.optimizer.zero_grad()

            with amp.autocast():
                heat = self.model(input_var)
                loss_heat   = self.criterion(heat,  heatmap_var)

            val_loss += loss_heat.item()

            tbar.set_description('Val   loss: %.6f' % (val_loss / ((i + 1)*self.batch_size)))

            acc, acc_PCK, acc_PCKh, cnt, pred, visible = evaluate.accuracy(heat.detach().cpu().numpy(), heatmap_var.detach().cpu().numpy(),0.2,0.5, self.dataset)

            AP[0]     = (AP[0]  *i + acc[0])      / (i + 1)
            PCK[0]    = (PCK[0] *i + acc_PCK[0])  / (i + 1)
            PCKh[0]   = (PCKh[0]*i + acc_PCKh[0]) / (i + 1)

            for j in range(1,self.numClasses+1):
                if visible[j] == 1:
                    AP[j]     = (AP[j]  *count[j] + acc[j])      / (count[j] + 1)
                    PCK[j]    = (PCK[j] *count[j] + acc_PCK[j])  / (count[j] + 1)
                    PCKh[j]   = (PCKh[j]*count[j] + acc_PCKh[j]) / (count[j] + 1)
                    count[j] += 1

            mAP     =   AP[1:].sum()/(self.numClasses)
            mPCK    =  PCK[1:].sum()/(self.numClasses)
            mPCKh   = PCKh[1:].sum()/(self.numClasses)
	
        # printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, self.dataset)
            
        if mAP > self.isBest:
            self.isBest = mAP
        if mPCKh > self.bestPCKh:
            self.bestPCKh = mPCKh
        if mPCK > self.bestPCK:
            save_path = "./weights/" + self.args.model_name
            os.makedirs(save_path, exist_ok=True)

            weight_name = os.path.join(save_path, self.args.model_name + "_best.pth.tar")
            torch.save({'state_dict': self.model.state_dict()}, weight_name)

            print("Model saved to "+self.args.model_name)

            self.bestPCK = mPCK

        # print("Best AP = %.2f%%; PCK = %2.2f%%; PCKh = %2.2f%%" % (self.isBest*100, self.bestPCK*100,self.bestPCKh*100))

        if epoch == (epochs - 1):
            save_path = "./weights/" + self.args.model_name
            os.makedirs(save_path, exist_ok=True)

            weight_final_name = os.path.join(save_path, self.args.model_name + "_final.pth.tar")

            torch.save({'state_dict': self.model.state_dict()}, weight_final_name)
            print("Final model saved to "+self.args.model_name)

    @torch.no_grad()
    def test(self, start_ts):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')

        annotation_li = []

        cnt = 0
        for i, (input, heatmap, centermap, img_path, img_shape) in enumerate(tbar):

            cnt += 1

            input_var     =      input.cuda()
            heatmap_var   =    heatmap.cuda()
            self.optimizer.zero_grad()

            with amp.autocast():
                heat = self.model(input_var)
                loss_heat   = self.criterion(heat,  heatmap_var)

            img_height = img_shape[0].numpy()[0]
            img_width = img_shape[1].numpy()[0]

            pred_kpts = get_kpts(heat, img_height, img_width)
            gt_kpts = get_kpts(heatmap, img_height, img_width)

            img_fn = img_path[0].split('/')[-1]
            annot = {
                "ID": i + 1,
                "img_path": img_fn,
                "joint_self": pred_kpts
            }
            annotation_li.append(annot)
        
        latency = time.time() - start_ts
        submission_dict = {
            "latency": latency,
            "annotations": annotation_li
        }

        with open('./submission.json', 'w') as f:
            json.dump(submission_dict, f)

        
def main():        
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=None,type=str, dest='pretrained')
    parser.add_argument('--dataset', type=str, dest='dataset', default='cow')
    parser.add_argument('--train_dir', default='./datasets/temp',type=str, dest='train_dir')
    parser.add_argument('--val_dir', type=str, dest='val_dir', default='./datasets/temp')
    parser.add_argument('--test_dir', type=str, dest='test_dir', default='./datasets/temp')
    parser.add_argument('--is_train', default=1, type=int, help='decide to train or test')
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--model_arch', default='unipose', type=str)

    args = parser.parse_args()

    args.model_name = 'cow_kps_model'

    start_ts = time.time()
    trainer = Trainer(args)

    if args.is_train:
        print("\n*********************************")
        print("===> Start training the model!")
        print("*********************************\n")

        for epoch in range(starter_epoch, epochs):
            trainer.training(epoch)
            trainer.validation(epoch)
    else:
        print("\n*********************************")
        print("===> Start testing the model!")
        print("*********************************\n")
        
        trainer.test(start_ts=start_ts)

if __name__ == "__main__":
    main()
