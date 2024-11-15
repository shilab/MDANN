import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.autograd import Function
from datetime import datetime
import pickle

def plot_acc(epochs,att,title,axis_label,label,color,line,file_name):
    linewidth = 1.0
    fontsize = 15
    font1 = {'size': fontsize}

    plt.figure()

    # use a gray background
    ax = plt.axes(facecolor='#E6E6E6')
    ax.set_axisbelow(True)

    # draw solid white grid lines
    plt.grid(color='w', linestyle='solid')

    # hide axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # hide top and right ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()


    # lighten ticks and labels
    ax.tick_params(colors='gray', direction='out')
    for tick in ax.get_xticklabels():
        tick.set_color('gray')
    for tick in ax.get_yticklabels():
        tick.set_color('gray')

    for i in range(len(att)):
        plt.plot(epochs, att[i], label=label[i], linestyle=line[i],color=color[i], linewidth=linewidth)
        #plt.plot(epochs, acc[i], label=label[i], linestyle='-', linewidth=linewidth)
    
    plt.title(title)
    plt.legend(loc='lower right', prop=font1)
    plt.xlabel("epoch", fontsize=fontsize)
    plt.ylabel(axis_label, fontsize=fontsize)
    if file_name != None:
        plt.savefig('.//Res//{}.png'.format(file_name))
    
    
def resize_img(data, size):
    temp = np.empty((data.shape[0], size,size), dtype=data.dtype)
    for (k, image) in enumerate(data):
        temp[k] = cv2.resize(image, dsize=(size, size))
    return temp

def normalize(data):
    data_norm = (data-np.min(data))/(np.max(data)-np.min(data))
    return data_norm





class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
    
    
def plot_test(epochs,metrics,axis_label,label,color,domain_marker,auc=False):
    
    linewidth = 1.0
    fontsize = 22
    font1 = {'size': fontsize}
    
    base_line = np.ones(epochs.shape)
    
    w_ratio = [3]
    for i in range(len(domain_marker)*2):
        w_ratio.append(2)
    # plt.figure()
    n_domain = len(domain_marker)
    n_models = len(metrics)
    fig, axs = plt.subplots(n_models, 1+(n_domain*2), layout='constrained', figsize=(8*(1+(2*n_domain)), 4*n_models), gridspec_kw={'width_ratios': w_ratio})
    
    #fig.grid(color='w', linestyle='solid')
    for ax in axs.flat:
        ax.set_facecolor('#E6E6E6')
        ax.set_axisbelow(True)
        ax.grid(color='w', linestyle='solid')
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors='black', direction='out')
        for tick in ax.get_xticklabels():
            tick.set_color('black')
        for tick in ax.get_yticklabels():
            tick.set_color('black')
    if n_models == 1:
        gridspec = axs[0].get_subplotspec().get_gridspec()
        for ax in axs[:2]:
            ax.remove()
    
        subfig = fig.add_subfigure(gridspec[:1])
        axsLeft = subfig.subplots(1,1, sharey=True)
        
    else:
        gridspec = axs[0, 0].get_subplotspec().get_gridspec()
        for ax in axs[:, :1].flat:
            ax.remove()
    
        subfig = fig.add_subfigure(gridspec[:, :1])
        axsLeft = subfig.subplots(1,1, sharey=True)
    axsLeft.set_facecolor('#E6E6E6')
    axsLeft.set_axisbelow(True)
    axsLeft.grid(color='w', linestyle='solid')
    axsLeft.xaxis.tick_bottom()
    axsLeft.yaxis.tick_left()
    axsLeft.tick_params(axis='x', labelsize=20)
    axsLeft.tick_params(axis='y', labelsize=20)
    for a in range(n_models):
        i = a
        # # plot total acc  -> metric['total acc']
        if auc:
            val = [x['auc'] for x in metrics[i]]
        else:
            val = [x['total acc'] for x in metrics[i]]
        axsLeft.plot(epochs, val, label=label[i], linestyle='-',color=color[i], linewidth=linewidth)
        
        # val = [x['acc total'] for x in metrics[i]]
        # ax[0][0].plot(epochs, val, label=label[i], linestyle=line[i],color=color[i], linewidth=linewidth)
        
        # plot acc_pos and acc_neg for each domain
        for j,domain in enumerate(domain_marker):
            val1 = [x['acc_{}_pos'.format(domain)] for x in metrics[i]]
            val2 = [x['acc_{}_neg'.format(domain)] for x in metrics[i]]

            
            axs[a,j*2+1].plot(epochs, val2, label='minority', linestyle='-',color='r', linewidth=linewidth)
            axs[a,j*2+1].plot(epochs, val1, label='majority'.format(domain), linestyle='-',color='b', linewidth=linewidth)                
            
            val3 = [y/x for x,y in zip(val1,val2)]
            #val3 = [x['di_{}'.format(domain)] for x in metrics[i]]
            axs[a,j*2+2].plot(epochs, val3, linestyle='-',color='r', linewidth=linewidth)
            axs[a,j*2+2].plot(epochs, base_line, label='Fairness baseline', linestyle='--',color='black', linewidth=linewidth)
            mse = (np.square(val3[-20:] - base_line[-20:])).mean()
           # mse = np.sum(np.abs(np.sum(val3[-20:]) - np.sum(base_line[-20:])))
            # print(a,j*2+2, val3[:-20], np.abs(val3[:-20]-base_line[:-20]), np.sum(val3[:-20]-base_line[:-20]))
            # mse = np.sum(val3[:-20]-base_line[:-20])
            axs[a,j*2+2].text(650, 0.50, 'MSE={:.3f}'.format(mse), style='italic', fontsize=fontsize)
            # if i==0:
            #     if domain == 'gender':
            #         domain = 'sex'
                # axs[i,j*2+1].set_title('{}'.format(domain),fontsize=fontsize)
                # axs[i,j*2+2].set_title('{}'.format(domain),fontsize=fontsize)
            axs[a,j*2+1].legend(loc='lower right', prop=font1)
            axs[i,j*2+1].set_xlabel("epoch", fontsize=fontsize)
            axs[i,j*2+1].set_ylabel(axis_label, fontsize=fontsize)
            axs[i,j*2+1].set_ylim(0,1)
            axs[a,j*2+2].legend(loc='lower right', prop=font1)
            axs[i,j*2+2].set_xlabel("epoch", fontsize=fontsize)
            axs[i,j*2+2].set_ylabel('DI', fontsize=fontsize)    
            axs[i,j*2+2].set_ylim(0,1.5)

            
       

    axsLeft.legend(loc='lower right', prop=font1)
    axsLeft.set_xlabel("epoch", fontsize=fontsize)
    axsLeft.set_ylabel(axis_label, fontsize=fontsize)
    axsLeft.set_title('Total '+ axis_label,fontsize=fontsize)
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d-%H-%M-%S")
    plt.savefig('.//Res//{}.png'.format(date_time))
    
    saveMetrics(metrics,'.//Res//{}.pkl'.format(date_time))
    

def saveMetrics(metrics,path):

    # create a binary pickle file 
    f = open(path,"wb")

    # write the python object (dict) to pickle file
    pickle.dump(metrics,f)

    # close file
    f.close()
    
    
def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    r_fpr = fpr[right_index]   
    r_tpr = tpr[right_index]
    return float(r_fpr), float(r_tpr),float(best_th), right_index






def plot_test2(epochs,metrics,axis_label,label,color,domain_marker,auc=False):
    
    linewidth = 1.0
    fontsize = 22
    font1 = {'size': fontsize}
    font2 = {'size': 32}
    base_line = np.ones(epochs.shape)
    
    w_ratio = [3]
    for i in range(len(domain_marker)*2):
        w_ratio.append(2)
    # plt.figure()
    n_domain = len(domain_marker)
    n_models = len(metrics)
    fig, axs = plt.subplots(n_models, 1+(n_domain*2), layout='constrained', figsize=(8*(1+(2*n_domain)), 4*n_models), gridspec_kw={'width_ratios': w_ratio})
    
    #fig.grid(color='w', linestyle='solid')
    for ax in axs.flat:
        ax.set_facecolor('#E6E6E6')
        ax.set_axisbelow(True)
        ax.grid(color='w', linestyle='solid')
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors='black', direction='out')
        for tick in ax.get_xticklabels():
            tick.set_color('black')
        for tick in ax.get_yticklabels():
            tick.set_color('black')
    if n_models == 1:
        gridspec = axs[0].get_subplotspec().get_gridspec()
        for ax in axs[:2]:
            ax.remove()
    
        subfig = fig.add_subfigure(gridspec[:1])
        axsLeft = subfig.subplots(1,1, sharey=True)
        
    else:
        gridspec = axs[0, 0].get_subplotspec().get_gridspec()
        for ax in axs[:, :1].flat:
            ax.remove()
    
        subfig = fig.add_subfigure(gridspec[:, :1])
        axsLeft = subfig.subplots(1,1, sharey=True)
    axsLeft.set_facecolor('#E6E6E6')
    axsLeft.set_axisbelow(True)
    axsLeft.grid(color='w', linestyle='solid')
    axsLeft.xaxis.tick_bottom()
    axsLeft.yaxis.tick_left()
    axsLeft.tick_params(axis='x', labelsize=35)
    axsLeft.tick_params(axis='y', labelsize=35)
    for i in range(n_models):
        #i = 2-a
        # # plot total acc  -> metric['total acc']
        if auc:
            val = [x['auc'] for x in metrics[i]]
        else:
            val = [x['total acc'] for x in metrics[i]]
        axsLeft.plot(epochs, val, label=label[i], linestyle='-',color=color[i], linewidth=linewidth)
        
        # val = [x['acc total'] for x in metrics[i]]
        # ax[0][0].plot(epochs, val, label=label[i], linestyle=line[i],color=color[i], linewidth=linewidth)
        
        # plot acc_pos and acc_neg for each domain
        for j,domain in enumerate(domain_marker):
            val1 = [x['acc_{}_pos'.format(domain)] for x in metrics[i]]
            val2 = [x['acc_{}_neg'.format(domain)] for x in metrics[i]]
            
            axs[i,j*2+1].plot(epochs, val1, label='minority', linestyle='-',color='r', linewidth=linewidth)
            axs[i,j*2+1].plot(epochs, val2, label='majority'.format(domain), linestyle='-',color='b', linewidth=linewidth)          
            
            #val3 = [y/x for x,y in zip(val1,val2)]
            val3 = [x['auc'] for x in metrics[i]]
            axs[i,j*2+2].plot(epochs, val3, linestyle='-',color='r', linewidth=linewidth)
            #axs[i,j*2+2].plot(epochs, base_line, label='Fairness baseline', linestyle='--',color='black', linewidth=linewidth)
            #mse = (np.square(val3[-20:] - base_line[-20:])).mean()
           # mse = np.sum(np.abs(np.sum(val3[-20:]) - np.sum(base_line[-20:])))
            # print(a,j*2+2, val3[:-20], np.abs(val3[:-20]-base_line[:-20]), np.sum(val3[:-20]-base_line[:-20]))
            # mse = np.sum(val3[:-20]-base_line[:-20])
            #axs[a,j*2+2].text(650, 0.50, 'MSE={:.3f}'.format(mse), style='italic', fontsize=fontsize)
            # if i==0:
            #     if domain == 'gender':
            #         domain = 'sex'
                # axs[i,j*2+1].set_title('{}'.format(domain),fontsize=fontsize)
                # axs[i,j*2+2].set_title('{}'.format(domain),fontsize=fontsize)
            axs[i,j*2+1].legend(loc='lower right', prop=font1)
            axs[i,j*2+1].set_xlabel("epoch", fontsize=fontsize)
            axs[i,j*2+1].set_ylabel(axis_label, fontsize=fontsize)
            axs[i,j*2+1].set_ylim(0,1)
            #axs[i,j*2+2].legend(loc='lower right', prop=font1)
            axs[i,j*2+2].set_xlabel("epoch", fontsize=fontsize)
            axs[i,j*2+2].set_ylabel('AUC', fontsize=fontsize)    
            axs[i,j*2+2].set_ylim(0,1)

            
       

    axsLeft.legend(loc='lower right', prop=font2)
    axsLeft.set_xlabel("epoch", fontsize=35)
    axsLeft.set_ylabel(axis_label, fontsize=35)
    axsLeft.set_title('Total '+ axis_label,fontsize=35)
    axsLeft.set_ylim(0.5,0.88)
    now = datetime.now() # current date and time
    date_time = now.strftime("%m_%d-%H-%M-%S")
    plt.savefig('.//Res//{}_acc_auc.png'.format(date_time))
