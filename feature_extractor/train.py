import torch
import time
import sys
from load_save_model import checkpoint_save_stage
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer():
    def __init__(self, loss=None, metric=None, validate_every=1, verborrea=True):
        self.loss_ce = loss
        self.metric = metric
        self.verborrea = verborrea
        self.USE_CUDA = torch.cuda.is_available()
        self.validate_every = validate_every

    def Train(self,model, optimizer, TrainSet, TestSet, Train_mode, Model_name, DataSet, epochs=None, scheduler=None):
        if self.loss_ce is None:
            print("Loss function not set,exiting...")
            sys.exit()
        
        if scheduler is None and epochs is None:
            print('WARNING!!!!Creating default min scheduler')
            scheduler = ReduceLROnPlateau(optimizer, "min", verbose=True,patience=10,eps=1e-8)
        path_checkpoint = os.getcwd()+'/CHECKPOINT/checkpoint_'+Model_name+'_'+Train_mode+'_'+DataSet+'/CHECKPOINT.t7'
        print('Checkpoint path',path_checkpoint)
        scheduler_mode = scheduler.mode

        max_lr,list_lr = self.update_list_lr(optimizer)
        trainloss_to_fil=[]
        testloss_to_fil=[]
        trainMetric_to_fil=[]
        testMetric_to_fil=[]
    
        if isinstance(scheduler,ReduceLROnPlateau):
            patience_num=scheduler.patience
        else:
            print('Scheduler not supported. But training will continue if epochs are specified.')
            if epochs==None:
                print('WARNING!!!! Number of epochs not specified')
                sys.exit()
            patience_num='nothing'
            
        parameters=[[],[],patience_num,optimizer.param_groups[0]['weight_decay']]#first list for epochs, second for learning rate,3rd patience, 4th weight_decay,5 for time
        parameters[1].append(list_lr)
        
        epoch=0
        if epochs==0:
            keep_training=False
        else:
            keep_training=True
            print ('INITIAL TEST STATISTICS')
            loss_test,metric = self.evaluate(model,TestSet)
            checkpoint_save_stage(model,trainloss_to_fil,testloss_to_fil,trainMetric_to_fil,testMetric_to_fil,parameters,Model_name,Train_mode,DataSet)
            check_load=0
            
            if isinstance(scheduler,ReduceLROnPlateau):
                if scheduler_mode == 'min':
                    scheduler.step(loss_test)
                else:
                    scheduler.step(metric)
            else:
                best_test = loss_test
                scheduler.step()
        since_init=time.time()
        while keep_training:
            epoch=epoch+1
            if epochs !=None:
                if self.verborrea: 
                    print('Epoch {}/{},  lr={}. patience={}, weight decay={}'.format(epoch, epochs,max_lr,scheduler.patience,optimizer.param_groups[0]['weight_decay']))
            else:
                if self.verborrea:
                    print('Epoch {}, lr={}, patience={}, weight decay={}'.format(epoch,max_lr,scheduler.patience,optimizer.param_groups[0]['weight_decay']))
    
            if self.verborrea:
                print('-' * 20)
          
            if self.verborrea:
                print ('TRAIN STATISTICS')
            model.train()
            train_loss,train_metric= self.train_scratch(model,TrainSet,optimizer) #Training happens here!

            if epoch % self.validate_every == 0 :
                if self.verborrea:
                    print ('TEST STATISTICS')
                print('Validating at epoch',epoch)
                model.eval()
                test_loss,test_metric= self.evaluate(model,TestSet)

                trainloss_to_fil.append(train_loss)
                testloss_to_fil.append(test_loss)
                trainMetric_to_fil.append(train_metric)
                testMetric_to_fil.append(test_metric)

                if isinstance(scheduler,ReduceLROnPlateau):
                    prev_num_bad_epochs=scheduler.num_bad_epochs
                    if self.verborrea:
                        print('-' * 10)
                    if scheduler_mode =='min':
                        save=(test_loss< scheduler.best)
                        scheduler.step(test_loss)
                    else:
                        save=(test_metric>scheduler.best)
                        scheduler.step(test_metric)
                        print('Best', scheduler.best)

                    if save:
                        checkpoint_save_stage(model,trainloss_to_fil,testloss_to_fil,trainMetric_to_fil,testMetric_to_fil,parameters,Model_name,Train_mode,DataSet)
                        check_load=0
                    if scheduler.num_bad_epochs==0 and prev_num_bad_epochs==scheduler.patience and not save:
                        max_lr,list_lr=self.update_list_lr(optimizer)
                        parameters[0].append(epoch)
                        parameters[1].append(max_lr)
                        model.load_state_dict(torch.load(path_checkpoint))
                        check_load=check_load+1
                        if self.verborrea: print ('Checkpoint loaded')

                    if max_lr<10*scheduler.eps or check_load==6:
                        keep_training=False
                else:
                    prev_max_lr=max_lr

                    scheduler.step()
                    max_lr,list_lr = self.update_list_lr(optimizer)
                    if test_loss<=best_test:
                        checkpoint_save_stage(model,trainloss_to_fil,testloss_to_fil,trainMetric_to_fil,testMetric_to_fil,parameters,Model_name,Train_mode,DataSet)
                    if max_lr<prev_max_lr:
                        parameters[0].append(epoch)
                        parameters[1].append(max_lr)
                        model.load_state_dict(torch.load(path_checkpoint))
                        if self.verborrea:
                            print ('Checkpoint loaded')
                
            if epochs!=None:
                if epoch==epochs:
                    keep_training=False

        if epochs!=0:
            model.load_state_dict(torch.load(path_checkpoint))
            if self.verborrea:
                print ('Checkpoint loaded')
    
        parameters[0].append(epoch)
        print ('FINAL TRAIN STATISTICS')
        train_loss,train_metric= self.evaluate(model,TrainSet)
        print ('FINAL TEST STATISTICS')
        test_loss,test_metric= self.evaluate(model,TestSet)
    
        trainloss_to_fil.append(train_loss)
        testloss_to_fil.append(test_loss)
        trainMetric_to_fil.append(train_metric)
        testMetric_to_fil.append(test_metric)
        
        time_elapsed=time.time()-since_init
        print('Total time elapsed',time_elapsed)
        parameters.append(time_elapsed)
        return (trainloss_to_fil,testloss_to_fil,trainMetric_to_fil,testMetric_to_fil,parameters)
    
    def train_scratch(self,model,DataSet,optimizer): #eval is not correct in the method
        _loss=0
        _correct=0
        model.train()
        kwargs ={}
        kwargs['state'] = 'train'

        for batch_idx, data in enumerate(DataSet):
            inputs, segbnd, bbox = data
            if self.USE_CUDA:
                inputs, segbnd, bbox = inputs.cuda(),segbnd.cuda(), bbox.cuda()
            kwargs['bbox'] = bbox
            optimizer.zero_grad()
            prediction =  model(inputs)
            total_loss = self.loss_ce(prediction, segbnd, **kwargs)
            total_loss.backward()
            optimizer.step()
            _loss += total_loss.item()

        _loss_average=_loss/len(DataSet)
        if self.verborrea:
            print('Loss: ',_loss)
            print('Average Loss: ',_loss_average)
        
        return _loss_average,0.0


    def evaluate(self,model,DataSet):
        kwargs ={}
        kwargs['state'] = 'val'
        model.eval()
        _loss=0
        _correct = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(DataSet):
                inputs, segbnd, bbox = data
                if self.USE_CUDA:
                    inputs, segbnd, bbox = inputs.cuda(),segbnd.cuda(), bbox.cuda()
                prediction =  model(inputs)
                kwargs['bbox'] = bbox
                total_loss=self.loss_ce(prediction, segbnd, **kwargs)
                _loss += total_loss.item()
            _loss_average =_loss/len(DataSet)

        if self.verborrea:
            print('Loss: ',_loss)
            print('Average Loss: ',_loss_average)

        return _loss_average, 0.0

    def update_list_lr(self,optimizer):
        list_lr=[]
        for param in optimizer.param_groups:
            list_lr.append(param['lr'])
        max_lr=max(list_lr)
        return max_lr,list_lr

