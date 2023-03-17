import os
print('Working dir', os.getcwd())
from load_save_model import save_model
from train_sampling_refine_withgt_separate_metric import Trainer
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.cpp_net import CPPNet
from models.feature_extractor import Feature_Extractor
from distance_loss_sampling_refine import L1Loss_List_withSAP_withSeg as L1BCELoss
import dataloader_custom
import random
import numpy as np

import argparse


def run(data_path, nc_in=1, init_lr=1e-4, n_rays=32, SAP_weight_path=None, n_sampling=6, K=5):
    crop_sz = None
    print(n_sampling, K)
    erosion_factor_list = [float(i+1)/n_sampling for i in range(n_sampling)]
    print(erosion_factor_list)

    for irnd in range(K):
        
        Trainloader, Testloader = dataloader_custom.getDataLoaders(n_rays, max_dist=None, root_dir=data_path)
        
        model = CPPNet(nc_in, n_rays, erosion_factor_list=erosion_factor_list)

        if SAP_weight_path is not None:
            SAP_weight = torch.load(SAP_weight_path)
            SAP_model = Feature_Extractor(n_rays+1, 32)
            SAP_model_weight = SAP_model.state_dict()
            for k, v in SAP_weight.items():
                if k in SAP_model_weight.keys():
                    SAP_model_weight.update({k:v})
                    print('Loaded: ', k, v.shape)
            SAP_model.load_state_dict(SAP_model_weight)
            SAP_model = SAP_model.cuda()
            SAP_model.eval()
            loss_scale = [1,1,1,1]
        else:
            SAP_model = None
            loss_scale = [1,1,1]
        loss = L1BCELoss(SAP_model, loss_scale)

        model_name='UNet2D_sampling_ensemble_n' + str(len(erosion_factor_list)) + '_r' + str(irnd) + '_weight_correct_conf_train3' + '_' + str(n_sampling) + '_withseg' + '_SAP_loss' + '_Others'
        print('model='+model_name)
        dataset='DSB2018_aug'
        print('dataset='+dataset)
        train_mode='StarDist'
        print('No.of rays',n_rays)

        kwargs={}
        additional_notes= '.'
        kwargs['additional_notes'] = additional_notes
        SAVE_PATH = os.getcwd()+'/'+dataset+'/'+train_mode+'_'+model_name+'/'
        kwargs['save_path'] = SAVE_PATH
        RESULTS_DIRECTORY = os.getcwd()+'/'+dataset+'/'+train_mode+'_'+model_name+'/plots/'

        trainer = Trainer(loss, None, None, validate_every=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, patience=10, eps=1e-8, threshold=1e-20)

        print ('Starting Training')
        # # Pre train
        trainer.pretrain(model, Trainloader, optimizer, 5)
        trainloss_to_file, testloss_to_file, trainMetric_to_file, testMetric_to_file, Parameters = trainer.Train(
            model,optimizer,
            Trainloader,Testloader,epochs=None,Train_mode=train_mode,
            Model_name=model_name,
            Dataset=dataset,scheduler=scheduler
        )
        print('Saving Final Model')
        save_model(model, trainMetric_to_file, testMetric_to_file, trainloss_to_file, testloss_to_file, Parameters, model_name,train_mode,dataset,plot=False,**kwargs)


DATA_PATH = ''
SAP_Weight_path = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--n_rays', type=int, default=32)
    parser.add_argument('--n_sampling', type=int, default=6)
    parser.add_argument('--nc_in', type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    # torch.set_num_threads(8)
    
    run(DATA_PATH, nc_in=args.nc_in, SAP_weight_path=SAP_Weight_path, n_rays=args.n_rays, n_sampling=args.n_sampling)
