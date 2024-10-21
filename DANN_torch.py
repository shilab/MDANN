import torch.nn as nn
from utils import ReverseLayerF


class DANN(nn.Module):

    def __init__(self,n_domain,input_channel,device):
        super(DANN, self).__init__()
        self.domain_classifier = []
        self.n_domain = n_domain
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(input_channel, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 13 *13, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(1))

        for i in range(n_domain):
            classifier = self.adversarial_block(input_shape=50*13*13, output_shape=2)
            self.domain_classifier.append(classifier.to(device))
        
    def adversarial_block(self, input_shape, output_shape):
        block = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, output_shape),
            nn.LogSoftmax(1)
        )
        return block
        
        
    def forward(self, input_data, alpha):
        domain_output = []
        feature = self.feature(input_data)
        feature = feature.reshape(input_data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        for i in range(self.n_domain):
            d_output = self.domain_classifier[i](reverse_feature)
            domain_output.append(d_output)

        return class_output, domain_output
    
    
class DANN_Rep(nn.Module):

    def __init__(self,n_domain,input_channel,device):
        super(DANN_Rep, self).__init__()
        self.n_domain = n_domain
        self.domain_classifier = []
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(input_channel, 128, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(128))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(128, 64, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(64))
        self.feature.add_module('f_drop1', nn.Dropout())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64 * 5 *5, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(1))

        for i in range(n_domain):
            classifier = self.adversarial_block(input_shape=64*5*5, output_shape=2)
            self.domain_classifier.append(classifier.to(device))
        
    def adversarial_block(self, input_shape, output_shape):
        block = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, output_shape),
            nn.LogSoftmax(1)
        )
        return block
        
        
    def forward(self, input_data, alpha):
        domain_output = []
        feature = self.feature(input_data)
        feature = feature.reshape(input_data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        for i in range(self.n_domain):
            d_output = self.domain_classifier[i](reverse_feature)
            domain_output.append(d_output)

        return class_output, domain_output
    
    
class DANN_ADNI(nn.Module):

    def __init__(self,n_domain,input_channel,device):
        super(DANN_ADNI, self).__init__()
        self.n_domain = n_domain
        self.domain_classifier = []
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(input_channel, 32, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(32))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(32, 64, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(64))
        self.feature.add_module('f_drop1', nn.Dropout())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64 * 13 *13, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(1))

        for i in range(n_domain):
            classifier = self.adversarial_block(input_shape=64*13*13, output_shape=2)
            self.domain_classifier.append(classifier.to(device))
        
    def adversarial_block(self, input_shape, output_shape):
        block = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, output_shape),
            nn.LogSoftmax(1)
        )
        return block
        
        
    def forward(self, input_data, alpha):
        domain_output = []
        feature = self.feature(input_data)
        feature = feature.reshape(input_data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        for i in range(self.n_domain):
            d_output = self.domain_classifier[i](reverse_feature)
            domain_output.append(d_output)

        return class_output, domain_output
    
    
class DANN_ADNI_Rep(nn.Module):

    def __init__(self,n_domain,input_channel,device):
        super(DANN_ADNI_Rep, self).__init__()
        self.n_domain = n_domain
        self.domain_classifier = []
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(input_channel, 128, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(128))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(128, 64, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(64))
        self.feature.add_module('f_drop1', nn.Dropout())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64 * 5 *5, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(1))

        for i in range(n_domain):
            classifier = self.adversarial_block(input_shape=64*5*5, output_shape=2)
            self.domain_classifier.append(classifier.to(device))
        
    def adversarial_block(self, input_shape, output_shape):
        block = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, output_shape),
            nn.LogSoftmax(1)
        )
        return block
        
        
    def forward(self, input_data, alpha):
        domain_output = []
        feature = self.feature(input_data)
        feature = feature.reshape(input_data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        for i in range(self.n_domain):
            d_output = self.domain_classifier[i](reverse_feature)
            domain_output.append(d_output)

        return class_output, domain_output
    
    
class DANN_ADNI_AUC(nn.Module):

    def __init__(self,n_domain,input_channel,device):
        super(DANN_ADNI_AUC, self).__init__()
        self.n_domain = n_domain
        self.domain_classifier = []
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(input_channel, 32, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(32))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(32, 64, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(64))
        self.feature.add_module('f_drop1', nn.Dropout())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64 * 13 *13, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 1))
        self.class_classifier.add_module('c_softmax', nn.Sigmoid())

        for i in range(n_domain):
            classifier = self.adversarial_block(input_shape=64*13*13, output_shape=1)
            self.domain_classifier.append(classifier.to(device))
        
    def adversarial_block(self, input_shape, output_shape):
        block = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, output_shape),
            nn.Sigmoid()
            # nn.LogSoftmax(1)
        )
        return block
        
        
    def forward(self, input_data, alpha):
        domain_output = []
        feature = self.feature(input_data)
        feature = feature.reshape(input_data.shape[0], -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        for i in range(self.n_domain):
            d_output = self.domain_classifier[i](reverse_feature)
            domain_output.append(d_output)

        return class_output, domain_output