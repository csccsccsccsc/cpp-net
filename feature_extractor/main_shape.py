import os
print('Working dir',os.getcwd())
from load_save_model import save_model
from train import Trainer
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.unet_model import UNet
from instance_loss import InstanceLoss
import dataloader_aug
import random
import numpy as np

import argparse

def run(data_path, n_rays, nc_in, nd_features, loss_scale, loss_type, init_lr=1e-4,):

    Trainloader, Testloader = dataloader_aug.getDataLoaders(n_rays, root_dir=data_path)
    model = UNet(nc_in, nd_features, loss_type=loss_type).cuda()

    model_name='UNet2D_'+str(nd_features)+'d' + ''
    print('model='+model_name)
    dataset='DSB2018_aug'
    print('dataset='+dataset)
    train_mode='StarDist2'+loss_type.capitalize()+'_' + str(n_rays)
    print('No.of rays', n_rays)

    kwargs={}
    additional_notes= ''
    kwargs['additional_notes'] = additional_notes
    SAVE_PATH = os.getcwd()+'/'+dataset+'/'+train_mode+'_'+model_name+'/'
    kwargs['save_path'] = SAVE_PATH
    RESULTS_DIRECTORY = os.getcwd()+'/'+dataset+'/'+train_mode+'_'+model_name+'/plots/'

    loss = InstanceLoss(scale=loss_scale)
    trainer = Trainer(loss, None, validate_every=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, patience=5, eps=1e-8, threshold=1e-20)

    print ('Starting Training')
    trainloss_to_file,testloss_to_file,trainMetric_to_file,testMetric_to_file,Parameters = trainer.Train(model,optimizer,
                                                                                        Trainloader, Testloader, epochs=None, Train_mode=train_mode,
                                                                                      Model_name=model_name,
                                                                                      DataSet=dataset,scheduler=scheduler)
    print('Saving model...')
    save_model(model,trainMetric_to_file,testMetric_to_file,trainloss_to_file,testloss_to_file,Parameters,model_name,train_mode,dataset, plot=False,**kwargs)


DATA_PATH = '/data/cong/datasets/dsb2018/dsb2018_in_stardist/dsb2018/dataset_split_for_training'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--n_rays', type=int, default=32)
    parser.add_argument('--nd_features', type=int, default=32)
    parser.add_argument('--loss_type', type=str, default='others')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    # torch.set_num_threads(8)

    loss_names = {'others':[1.0, 1.0], 'bbox':[0.0, 1.0], 'segbnd':[1.0, 0.0], }

    run(DATA_PATH, args.n_rays, args.n_rays + 1, args.nd_features, loss_names[args.loss_type], args.loss_type)