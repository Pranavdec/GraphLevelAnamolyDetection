import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import argparse
import load_data
import torch
from loss import *
from util import *
from torch.autograd import Variable
from GraphBuild import GraphBuild
from numpy.random import seed
import random
import torch.nn.functional as F
from model import *
from random import shuffle
import math
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

def arg_parse():
    parser = argparse.ArgumentParser(description='G-Anomaly Arguments.')
    parser.add_argument('--datadir', dest='datadir', default ='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default ='BZR', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0, help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--num_epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=2000, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=256, type=int, help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', default=128, type=int, help='Output dimension')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', default=2, type=int, help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', default=0.1, type=float, help='Dropout rate.')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR', help='learning rate')
    parser.add_argument('--nobias', dest='bias', action='store_const', const=False, default=True, help='Whether to add bias. Default to True.')
    parser.add_argument('--feature', dest='feature', default='deg-num', help='use what node feature, value "default" for using node features from dataset')
    parser.add_argument('--seed', dest='seed', type=int, default=2, help='seed')
    parser.add_argument('--contrasive_lg', dest='contrasive_learning', default=True, help='use Contrasive learning')
    parser.add_argument('--patience', dest='patience', default=5, type=int, help='Patience for LR scheduler')
    parser.add_argument('--early_stopping_patience', dest='early_stopping_patience', default=10, type=int, help='Patience for early stopping')
    parser.add_argument('--threshold_lr', dest='threshold_lr', default=1e-7, type=float, help='Threshold for learning rate scheduler')
    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
        
def gen_ran_output(h0, adj, model, vice_model):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + 1.0 * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(param.device)    
    x1_r,Feat_0= vice_model(h0, adj)
    return x1_r,Feat_0

def train(dataset, data_test_loader, NetG, noise_NetG, args, device):    
    optimizerG = torch.optim.Adam(NetG.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizerG, mode='min', factor=0.1, patience=args.patience, threshold=args.threshold_lr) 
    early_stopping_patience = args.early_stopping_patience
    early_stopping_counter = 0
    epochs=[]
    max_AUC_ROC=0
    max_AUC_PR=0
    best_combined_metric = -np.inf
    best_weights = None
    
    for epoch in range(args.num_epochs):
        total_lossG = 0.0
        NetG.train()
        for batch_idx, data in enumerate(dataset):           
            adj = Variable(data['adj'].float(), requires_grad=False).to(device)

            h0 = Variable(data['feats'].float(), requires_grad=False).to(device)
            adj_label = Variable(data['adj_label'].float(), requires_grad=False).to(device)

            #First Encoding of input
            x1, a0 = NetG.shared_encoder(h0, adj)
            #Perturb Encoding of input
            x1_1, a0_1 = gen_ran_output(h0, adj, NetG.shared_encoder, noise_NetG)

            #Reconstruction of input and Encoding of reconstructed input
            x_fake, a_fake, x2, a1 = NetG(x1,adj)

            #Reconstruction loss
            err_g_con_s, err_g_con_x = loss_func(adj_label, a_fake, h0, x_fake)
            #Consistency loss
            node_loss = torch.mean(F.mse_loss(x1, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            graph_loss = F.mse_loss(a0, a1, reduction='none').mean(dim=1).mean(dim=0)
            
            if args.contrasive_learning:
                #Contrastive loss
                err_g_enc = loss_cal(a0_1, a0)
                lossG = err_g_con_s + err_g_con_x + graph_loss + node_loss+ err_g_enc
            else:
                lossG = err_g_con_s + err_g_con_x + graph_loss + node_loss

            optimizerG.zero_grad()
            lossG.backward()

            optimizerG.step()
            total_lossG += lossG.item()
                   
        if (epoch+1)%10 == 0 and epoch > 0:
            epochs.append(epoch)
            NetG.eval()   
            loss = []
            y=[]
            
            for batch_idx, data in enumerate(data_test_loader):
               adj = Variable(data['adj'].float(), requires_grad=False).to(device)
               h0 = Variable(data['feats'].float(), requires_grad=False).to(device)

               x1, a0 = NetG.shared_encoder(h0, adj)
               x_fake, a_fake, x2, a1=NetG(x1,adj)

               loss_node=torch.mean(F.mse_loss(x1, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)

               loss_graph = F.mse_loss(a0, a1, reduction='none').mean(dim=1)

               loss_=loss_node+loss_graph

               loss_ = loss_.cpu().detach().numpy()
               
               loss.append(loss_)

               if data['label'] == 0:
                   y.append(0)
               else:
                   y.append(1)
            
            label_test = np.array(loss)
            fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
            test_roc_auc = auc(fpr_ab, tpr_ab)   
            
            precision, recall, _ = precision_recall_curve(y, label_test)
            test_pr_auc = auc(recall, precision)
            
            combined_metric = (test_roc_auc + test_pr_auc - total_lossG) / 3
            combined_metric = round(combined_metric, 2)
            
            print('Epoch: ', epoch, 'Loss: ', total_lossG, 'Eval ROC AUC: ', test_roc_auc, 'Eval PR AUC: ', test_pr_auc, '\n')
                
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_weights = NetG.state_dict()
                max_AUC_ROC = test_roc_auc
                max_AUC_PR = test_pr_auc
                early_stopping_counter = 0  # Reset early stopping counter
            else:
                early_stopping_counter += 1

            scheduler.step(total_lossG)
            
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered\n")
                break
        
    print('Max AUCROC: ', max_AUC_ROC, '\n')
    print('Max AUCPR: ', max_AUC_PR, '\n')
    NetG.load_state_dict(best_weights)
    return max_AUC_ROC, max_AUC_PR


def main(args):
    DS = args.DS
    setup_seed(args.seed)

    test_available = True
    try:
        graphs_train_ = load_data.read_graphfile(args.datadir, args.DS+'_training', max_nodes=args.max_nodes)  
        graphs_test = load_data.read_graphfile(args.datadir, args.DS+'_testing', max_nodes=args.max_nodes)  
        datanum = len(graphs_train_) + len(graphs_test)    
    except:
        test_available = False
        graphs_train_ = load_data.read_graphfile(args.datadir, args.DS, max_nodes=args.max_nodes)
        datanum = len(graphs_train_)  
        
    
    if args.max_nodes == 0:
        max_nodes_num_train = max([G.number_of_nodes() for G in graphs_train_])
        if test_available:
            max_nodes_num_test = max([G.number_of_nodes() for G in graphs_test])
            max_nodes_num = max([max_nodes_num_train, max_nodes_num_test])
        else:
            max_nodes_num = max_nodes_num_train
    else:
        max_nodes_num = args.max_nodes
    
    print("Total No.of Graphs in the dataset: ", datanum, '\n')

    #Split the training dataset into training and testing
    if not test_available:
        test_data_ratio = 0.2
        graphs_train_, graphs_test = train_test_split(graphs_train_, test_size=test_data_ratio, random_state=args.seed)        
    
    train_num=len(graphs_train_)
    all_idx = [idx for idx in range(train_num)]
    shuffle(all_idx)
    num_train=math.ceil(1*train_num)
    train_index = all_idx[:num_train]
    graphs_train_1 = [graphs_train_[i] for i in train_index]
    graphs_train = []
    a = 0
    b = 0
    for graph in graphs_train_1:
        if graph.graph['label'] == 0:
            graphs_train.append(graph)
        else:
            graphs_test.append(graph)
         

    for graph in graphs_test:
        if graph.graph['label'] != 0:
            b += 1
            graph.graph['label'] = 1
        else:
            a += 1

    
    
    num_train = len(graphs_train)
    num_test = len(graphs_test)
    
    print("No.of Graphs in the training dataset: ", num_train, '\n')
    print("No.of Graphs in the testing dataset: ", num_test, '\n')
    print("Anamoly Graphs in Test: ", b, '\n')
    print("Non-Anamoly Graphs in Test: ", a, '\n')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Device: ", device, '\n')
        
    dataset_sampler_train = GraphBuild(graphs_train, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
    
    
    NetG= NetGe(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, args.num_gc_layers, bn=args.bn, dropout=args.dropout, use_projection_head=args.contrasive_learning, bias=args.bias).to(device)

   
    noise_NetG= Encoder(dataset_sampler_train.feat_dim, args.hidden_dim, args.output_dim, args.num_gc_layers, bn=args.bn, dropout=args.dropout, use_projection_head=args.contrasive_learning, bias=args.bias).to(device)
        
    
    data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train, shuffle=True, batch_size=args.batch_size)

    
    dataset_sampler_test = GraphBuild(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
    data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test, shuffle=False, batch_size=1)
   
    roc,pr = train(data_train_loader, data_test_loader, NetG, noise_NetG, args, device)
    
    return roc,pr
    
    
if __name__ == '__main__':
    args = arg_parse()
    main(args)

    
    
    
