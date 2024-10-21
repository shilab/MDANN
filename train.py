import numpy as np
import torch
import math
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,roc_curve
from utils import get_err_threhold

def calAcc(pred, label):
    n_correct = pred.eq(label.data.view_as(pred)).cpu().sum()
    acc = n_correct.data.numpy() * 1.0 / label.shape[0]
    # print(n_correct,label.shape[0])
    return acc

# calculate desperate impact
def calDI(acc_unpriv, acc_priv):
    return acc_unpriv/acc_priv

def getFairMetrics(predict, label_main, label_domain):
    idx_pos = torch.where(label_domain==1)
    idx_neg = torch.where(label_domain==0)
    
    acc_pos = calAcc(predict[idx_pos],label_main[idx_pos])
    acc_neg = calAcc(predict[idx_neg],label_main[idx_neg])
    
    di = calDI(acc_pos,acc_neg)
    return acc_pos,acc_neg,di,len(idx_pos[0]),len(idx_neg[0])

def Train(model, dataloader_train, dataloader_test, n_epoch, optimizer, criterion, domain_marker, domain_used, imb_rate,single_channel,output=True,auc_loss=False):
    best_accu_t = 0.0
    metric = []
    l = len(dataloader_train)
    num_pos = {}
    num_neg = {}
    acc_train_list = []
    for epoch in range(n_epoch):
        model.train()
        loss_epoch = 0
        acc_train = 0
        for i, (img, labels) in enumerate(dataloader_train):
            if single_channel:
                img = img.view(-1,1,img.shape[1],img.shape[2])
            else:
                img = img.permute(0, 3, 1, 2)
            loss = 0
            p = float(i + epoch * l) / n_epoch / l
            alpha = 2. / (1. + np.exp(-10 * p))-1
            
            model.zero_grad()
            class_output, domain_output = model(input_data=img, alpha=alpha)
            if not auc_loss:
                main_label = torch.column_stack((1-labels[:,0],labels[:,0]))
            else:
                main_label = labels[:,0]
                class_output = class_output.view(-1)
            loss_main = criterion[0](class_output,main_label)
            for i in range(1,domain_used+1):
                if not auc_loss:
                    label = torch.column_stack((1-labels[:,i],labels[:,i]))
                    a = criterion[i](domain_output[i-1],label)
                    loss += criterion[i](domain_output[i-1],label)
                else:
                    label = labels[:,i]
                    loss += 0.01 * criterion[i](domain_output[i-1].view(-1),label).expand(1)
                #loss += imb_rate[i-1] * criterion[i](domain_output[i-1],label)
            
            loss += 1 * loss_main
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss
            if not auc_loss:
                pred = class_output.data.max(1, keepdim=True)[1]
            else:
                #pred = class_output.round()
                pred = (class_output>=0.5).float()
                # print(pred,labels[:,0])
            acc_train += calAcc(pred, labels[:,0])
        acc_train /= len(dataloader_train)
        # print(class_output)
        acc_train_list.append(acc_train)
        # print(criterion[0].a,criterion[0].b,criterion[0].alpha)
        if epoch%10 == 0:
            acc_epoch = {}
            model.eval()
            for t_img, t_label in dataloader_test:
                if single_channel:
                    t_img = t_img.view(-1,1,t_img.shape[1],t_img.shape[2])
                else:
                    t_img = t_img.permute(0,3,1,2)
                class_output, _ = model(input_data=t_img, alpha=alpha)
                if not auc_loss:
                    pred = class_output.data.max(1, keepdim=True)[1]
                else:
                    pred = (class_output>=0.5).float()

                acc_main = calAcc(pred, t_label[:,0])

                acc_epoch['total acc'] = acc_main
                
                auc = roc_auc_score( t_label[:,0].cpu().numpy(),pred.cpu().numpy())
                acc_epoch['auc'] = auc
                for i in range(1,t_label.shape[1]):
                    acc_pos,acc_neg,di,n_pos,n_neg = getFairMetrics(pred,t_label[:,0],t_label[:,i])
                    acc_epoch['acc_{}_pos'.format(domain_marker[i-1])] = acc_pos
                    acc_epoch['acc_{}_neg'.format(domain_marker[i-1])] = acc_neg
                    acc_epoch['di_{}'.format(domain_marker[i-1])] = di
                    num_pos[domain_marker[i-1]] = n_pos
                    num_neg[domain_marker[i-1]] = n_neg
                    # print('acc_{}_pos: {}, acc_{}_neg: {}, di_{}: {}'.format(domain_marker[i-1],acc_pos,domain_marker[i-1],acc_neg,domain_marker[i-1],di))
            metric.append(acc_epoch)
        if output:
            if (epoch+1)%2 == 0:
                print('\rEpoch {}: training: loss: {:.3f}, acc: {:.3f}, eval: {:.3f}, auc: {:.3f}'.format(epoch+1,loss_epoch.detach().cpu().numpy(),acc_train,acc_main,auc),end='')
                # print('evaluating: {}'.format(acc_epoch))
    s_str = ''
    for domain_name in domain_marker:
        s_str += '{}: pos/neg = {}/{}\t'.format(domain_name, num_pos[domain_name], num_neg[domain_name])
    print('')
    print(s_str)
    return acc_train_list, metric
                
        
        
def TrainAUC(model, dataloader_train, dataloader_test, n_epoch, optimizer, criterion, domain_marker, domain_used, imb_rate,single_channel,output=True,auc_loss=False):
    best_accu_t = 0.0
    metric = []
    l = len(dataloader_train)
    num_pos = {}
    num_neg = {}
    acc_train_list = []
    for epoch in range(n_epoch):
        model.train()
        loss_epoch = 0
        acc_train = 0
        for i, (img, labels) in enumerate(dataloader_train):
            if single_channel:
                img = img.view(-1,1,img.shape[1],img.shape[2])
            else:
                img = img.permute(0, 3, 1, 2)
            loss = 0
            p = float(i + epoch * l) / n_epoch / l
            alpha = 2. / (1. + np.exp(-10 * p))-1
            
            model.zero_grad()
            
            class_output, domain_output = model(input_data=img, alpha=alpha)
            if not auc_loss:
                main_label = torch.column_stack((1-labels[:,0],labels[:,0]))
            else:
                main_label = labels[:,0]
                class_output = class_output.view(-1)
            loss_main = criterion[0](class_output,main_label)
            for i in range(1,domain_used+1):
                if not auc_loss:
                    label = torch.column_stack((1-labels[:,i],labels[:,i]))
                    loss += criterion[i](domain_output[i-1],label)
                else:
                    label = labels[:,i]
                    loss += 0.01 * criterion[i](domain_output[i-1].view(-1),label)
                #loss += imb_rate[i-1] * criterion[i](domain_output[i-1],label)
            
            loss += 1 * loss_main
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss
            if not auc_loss:
                pred = class_output.data.max(1, keepdim=True)[1]
            else:
                #pred = class_output.round()
                pred = (class_output>=0.5).float()
                # print(pred,labels[:,0])
            acc_train += calAcc(pred, labels[:,0])
        acc_train /= len(dataloader_train)
        # print(class_output)
        acc_train_list.append(acc_train)
        # print(criterion[0].a,criterion[0].b,criterion[0].alpha)
        if epoch%10 == 0:
            acc_epoch = {}
            model.eval()
            for t_img, t_label in dataloader_test:
                if single_channel:
                    t_img = t_img.view(-1,1,t_img.shape[1],t_img.shape[2])
                else:
                    t_img = t_img.permute(0,3,1,2)
                class_output, _ = model(input_data=t_img, alpha=alpha)
                if not auc_loss:
                    pred = class_output.data.max(1, keepdim=True)[1]
                else:
                    pred = (class_output>=0.5).float()
                

                acc_main = calAcc(pred, t_label[:,0])

                acc_epoch['total acc'] = acc_main
                
                if(np.isnan(class_output.detach().cpu().numpy()).any()):
                    print('detect nan output...')
                    auc = 0
                else:
                    auc = roc_auc_score( t_label[:,0].cpu().numpy(),class_output.detach().cpu().numpy())
                acc_epoch['auc'] = auc
                
                fpr,tpr,thresh = roc_curve(t_label[:,0].cpu().numpy(),class_output.detach().cpu().numpy())

                r_fpr,r_tpr,best_thresh,_ = get_err_threhold(fpr, tpr, thresh)
                
                pred2 = (class_output>=best_thresh).float()
                acc_main2 = calAcc(pred2, t_label[:,0])
                acc_epoch['total acc'] = acc_main2
                acc_epoch['thresh'] = best_thresh
                for i in range(1,t_label.shape[1]):
                    acc_pos,acc_neg,di,n_pos,n_neg = getFairMetrics(pred,t_label[:,0],t_label[:,i])
                    acc_epoch['acc_{}_pos'.format(domain_marker[i-1])] = acc_pos
                    acc_epoch['acc_{}_neg'.format(domain_marker[i-1])] = acc_neg
                    acc_epoch['di_{}'.format(domain_marker[i-1])] = di
                    num_pos[domain_marker[i-1]] = n_pos
                    num_neg[domain_marker[i-1]] = n_neg
                    # print('acc_{}_pos: {}, acc_{}_neg: {}, di_{}: {}'.format(domain_marker[i-1],acc_pos,domain_marker[i-1],acc_neg,domain_marker[i-1],di))
            metric.append(acc_epoch)
        if output:
            if (epoch+1)%2 == 0:
                if not auc_loss:
                    print('\rEpoch {}: training: loss: {:.3f}, acc: {:.3f}'.format(epoch+1,loss_epoch,acc_train),end='')
                else:
                    print('\rEpoch {}: training: loss: {:.3f}, acc: {:.3f}, eval: {:.3f}, auc: {:.3f}, best_thresh: {:.3f}, final_acc: {:.3f}'.format(epoch+1,loss_epoch.detach().cpu().numpy()[0],acc_train,acc_main,auc,best_thresh, acc_main2),end='')
        
        if math.isinf(best_thresh):
            break
                # print('evaluating: {}'.format(acc_epoch))
    s_str = ''
    for domain_name in domain_marker:
        s_str += '{}: pos/neg = {}/{}\t'.format(domain_name, num_pos[domain_name], num_neg[domain_name])
    print('')
    print(s_str)
    return acc_train_list, metric
                
        