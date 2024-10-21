from DANN_torch import DANN
from load_data import LoadAutism
from train import Train
from utils import plot_acc,plot_test
import torch
import numpy as np
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:1')
lr = 1e-5
n_epochs = 1000
epochs = np.arange(n_epochs)

domain_marker = ['gender','handedness']
test_label = ['original'] + domain_marker
test_num = 5

avg_domain_metric = {label:[Counter({}) for x in range(n_epochs//10)] for label in test_label}#Counter({x:Counter({}) for x in test_label})

for t in range(test_num):
    train_dataset, test_dataset,imb_rate = LoadAutism(domain_marker,device)
    #domain_metric_list = []
    for i,label in enumerate(test_label):
        print('test {}, used_domain {}...'.format(t,i)) 
        domain_used = i  
        model = DANN(n_domain=domain_used,input_channel=1,device=device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_weight = torch.Tensor([imb_rate[0],1-imb_rate[0]]).to(device)
        criterion = [torch.nn.CrossEntropyLoss(weight=loss_weight)]
        for j in range(domain_used):
            loss_weight = torch.Tensor([imb_rate[j+1],1-imb_rate[j+1]]).to(device)
            print('domain:{},pos_weight:{},neg_weight{}'.format(label,loss_weight[0],loss_weight[1]))
            criterion.append(torch.nn.CrossEntropyLoss(weight=loss_weight))
        train_acc,metric = Train(model,train_dataset,test_dataset,n_epochs,optimizer,criterion,domain_marker,domain_used,imb_rate,single_channel=True,output=True)
        
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
plot_test(epochs[::10], list(avg_domain_metric.values()),'ACC',['1 feature','2 features','3 features'],['r','g','b'],domain_marker)