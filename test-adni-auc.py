from DANN_torch import DANN_ADNI_AUC
from load_data import LoadADNI
from train import TrainAUC
from utils import plot_acc,plot_test
import torch
import numpy as np
from collections import Counter
from Optimizer.auc_loss import AUCMLoss
from Optimizer.sgdam import SGDAM
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:3')
lr = 0.15
n_epochs = 1000
epochs = np.arange(n_epochs)

domain_marker = ['age','handedness','educate']
test_label = ['original'] + domain_marker
test_num = 5

momentum_primal = 0.1 
momentum_dual = 0.1
rho_primal = 0.033 
rho_dual = 0.033
margin = 1.0
weight_decay = 1e-4

avg_domain_metric = {label:[Counter({}) for x in range(n_epochs//10)] for label in test_label}#Counter({x:Counter({}) for x in test_label})

for t in range(test_num):
    train_dataset, test_dataset,imb_rate = LoadADNI(domain_marker,device,resize=True,rep=False)
    #domain_metric_list = []
    for i,label in enumerate(test_label):
        print('test {}, used_domain {}-{}...'.format(t,i,label))  
        # if i == 0:
        #     continue
        domain_used = i  
        model = DANN_ADNI_AUC(n_domain=domain_used,input_channel=3,device=device)
        model = model.to(device)
        loss_weight = torch.Tensor([imb_rate[0],1-imb_rate[0]]).to(device)
        criterion = [AUCMLoss(imratio=None)]
        
        
        
        params = [
            {
                "params": [value],
                "name": key,
                "param_size": value.size(),
                "nelement": value.nelement(),
            }
            for key, value in model.named_parameters()
        ]
        

        a = criterion[0].a
        b = criterion[0].b
        alpha = criterion[0].alpha

        params.append({"params": a,
                       "name": "primal_a_0",
                       "param_size": a.size(),
                       "nelement": a.nelement()})

        params.append({"params": b,
                       "name": "primal_b_0",
                       "param_size": b.size(),
                       "nelement": b.nelement()})

        params.append({"params": alpha,
                       "name": "dual_alpha_0",
                       "param_size": alpha.size(),
                       "nelement": alpha.nelement()})

        
        
        # params = list(model.parameters()) + list(criterion[0].parameters())
        
        for j in range(domain_used):
            loss_weight = torch.Tensor([imb_rate[j+1],1-imb_rate[j+1]]).to(device)
            print('domain:{},pos_weight:{},neg_weight{}'.format(label,loss_weight[0],loss_weight[1]))
            # criterion.append(torch.nn.CrossEntropyLoss(weight = loss_weight))
            criterion.append(AUCMLoss(imratio=None))
            
            a = criterion[j+1].a
            b = criterion[j+1].b
            alpha = criterion[j+1].alpha

            params.append({"params": a,
                           "name": "primal_a_{}".format(j+1),
                           "param_size": a.size(),
                           "nelement": a.nelement()})

            params.append({"params": b,
                           "name": "primal_b_{}".format(j+1),
                           "param_size": b.size(),
                           "nelement": b.nelement()})

            params.append({"params": alpha,
                           "name": "dual_alpha_{}".format(j+1),
                           "param_size": alpha.size(),
                           "nelement": alpha.nelement()})
            # params += list(criterion[j+1].parameters())
        
        # optimizer = torch.optim.Adam(params, lr=lr)
        optimizer = SGDAM(params, lr=lr,
                          momentum_primal=momentum_primal, momentum_dual=momentum_dual,
                          rho_primal=rho_primal, rho_dual=rho_dual,
                          margin=margin, weight_decay=weight_decay, num_domain = domain_used+1)


        train_acc,metric = TrainAUC(model,train_dataset,test_dataset,n_epochs,optimizer,criterion,domain_marker,domain_used,imb_rate,single_channel=False,output=True,auc_loss=True)
        if math.isinf(metric[0]['thresh']):
            t = t-1
            continue
        for i,m_epoch in enumerate(metric):
            avg_domain_metric[label][i] += Counter(m_epoch)
    


for label in test_label:
    for i in range(n_epochs//10):
        for itm in avg_domain_metric[label][i]:
            avg_domain_metric[label][i][itm] /= test_num

for label in test_label:
    for i in range(n_epochs//10):
        for domain in domain_marker:
            avg_domain_metric[label][i]['di_{}'.format(domain)] = (avg_domain_metric[label][i]['acc_{}_pos'.format(domain)]
                                                                   /avg_domain_metric[label][i]['acc_{}_neg'.format(domain)])             
            
# hard coded        
plot_test(epochs[::10], list(avg_domain_metric.values()),'ACC',['1 feature','2 features','3 features','4 features'],['r','g','b','k'],domain_marker)