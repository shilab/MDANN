import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
# import cv2

    

class MyDataset(Dataset):
    def __init__(self, features, targets, device):
#        self.features = features.astype(np.uint8)
        self.features = torch.Tensor(features).to(device)
        self.targets = torch.Tensor(targets).to(device)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        return feature, target   

def resize_img(img, img_size):
    temp = np.empty((img.shape[0], img_size,img_size), dtype=img.dtype)
    for (k, image) in enumerate(img):
        temp[k] = cv2.resize(image, dsize=(img_size, img_size))
    return temp

def get_imb_rate(labels):
    imb_rate = []
    for i in range(labels.shape[1]):
        label = labels[:,i]
        num_total = len(label)
        num_pos = len(np.where(label==1)[0])
        num_neg = len(np.where(label==0)[0])
        # if num_pos < num_neg:
        #     imb_rate.append(num_neg/num_total)
        # else:
        #     imb_rate.append(num_pos/num_total)
        imb_rate.append(num_pos/num_total)
        #print(num_total,num_pos,num_neg,imb_rate)
    return imb_rate
    

def LoadAutism(adapt_labels,device,batch_size=32,resize=True,img_size=64,test_size=0.2,rep=False):
    adapt_order = []
    print('loading files...')
    if rep:
        data = np.load('.//Representations//GU.npz')['arr_0']
    else:
        data = np.load('.//Matrices//GU.npz')['arr_0']
    labels = np.load('.//Matrices//GU_label.npz')['arr_0']
    # id = np.load('.//Matrices//GU_id.npz')['arr_0']
    print(data.shape)
    print(len(data))
    
    
    
    if len(adapt_labels)==0:
        labels = labels.reshape((-1,1))
    
    for alabel in adapt_labels:
        if alabel == 'handedness':
            print('loading handedness...')
            label_adapt = np.load('.//Matrices//GU_handedness.npz')['arr_0']   # 0 -> right handed  1-> left handed 2 -> mix handed
            adapt_order.append('handedness')
        
        elif alabel == 'gender':
            print('loading gender...')
            label_adapt = np.load('.//Matrices//GU_sex.npz')['arr_0']   # 0 -> male  1-> female
            adapt_order.append('gender')
        labels = np.column_stack((labels,label_adapt))
        
    imb_rate = get_imb_rate(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=test_size, random_state=42)
    
    if resize:
        print('Resizeing images to {}x{}...'.format(img_size,img_size))
        X_train = resize_img(X_train, img_size)
        X_test = resize_img(X_test, img_size)
    
    print('Initializing dataset...')
    train_set = MyDataset(X_train, y_train,device)
    trainloader = torch.utils.data.DataLoader(train_set,
                                              #collate_fn=lambda x: default_collate(x).to(device),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=False)

    test_set = MyDataset(X_test, y_test,device)
    testloader = torch.utils.data.DataLoader(test_set,
                                            #collate_fn=lambda x: default_collate(x).to(device),
                                             batch_size=test_set.__len__(),
                                             shuffle=False,
                                             pin_memory=False)
    
    return trainloader,testloader,imb_rate



def LoadADNI(adapt_labels,device,batch_size=32,resize=True,img_size=64,test_size=0.2,rep=False):
    adapt_order = []
    print('loading files...')
    if rep:
        data = None
        data = np.load('.//Data//adni_sliced_rep.npz')['arr_0']
    else:
        data = np.load('.//Data//adni_sliced.npz')['arr_0']
    labels = np.load('.//Data//adni_label_slice.npz')['arr_0']
    # id = np.load('.//Matrices//GU_id.npz')['arr_0']
    print(data.shape)

    
    
    
    if len(adapt_labels)==0:
        labels = labels.reshape((-1,1))
    
    for alabel in adapt_labels:
        if alabel == 'handedness':
            print('loading handedness...')
            label_adapt = np.load('.//Data//adni_handedness_slice.npz')['arr_0']   # 0 -> right handed  1-> left handed 2 -> mix handed
            adapt_order.append('handedness')
        
        elif alabel == 'race':
            print('loading race...')
            label_adapt = np.load('.//Data//adni_race_slice.npz')['arr_0']   # 0 -> white  1-> others
            adapt_order.append('race')
        
        elif alabel == 'educate':
            print('loading educate...')
            label_adapt = np.load('.//Data//adni_educate_slice.npz')['arr_0']   # 0 -> above 16  1-> below 16
            adapt_order.append('educate')   
            
        elif alabel == 'age':
            print('loading age...')
            label_adapt = np.load('.//Data//adni_age_slice.npz')['arr_0']   # 0 -> above 78  1-> below 78
            adapt_order.append('age')       
            
        labels = np.column_stack((labels,label_adapt))
    print(labels.shape)    
    imb_rate = get_imb_rate(labels)
    
    if resize:
        print('Resizeing images to {}x{}...'.format(img_size,img_size))
        temp = np.empty((data.shape[0], img_size,img_size,data.shape[1]), dtype=data.dtype)
        for i in range(data.shape[1]):

            data_r = resize_img(data[:,i,:,:], img_size)
            temp[:,:,:,i] = data_r
            data = temp

    X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=test_size, random_state=42)
    

    
    print('Initializing dataset...')
    train_set = MyDataset(X_train, y_train,device)
    trainloader = torch.utils.data.DataLoader(train_set,
                                              #collate_fn=lambda x: default_collate(x).to(device),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=False)

    test_set = MyDataset(X_test, y_test,device)
    testloader = torch.utils.data.DataLoader(test_set,
                                            #collate_fn=lambda x: default_collate(x).to(device),
                                             batch_size=test_set.__len__(),
                                             shuffle=False,
                                             pin_memory=False)
    
    return trainloader,testloader,imb_rate