import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, bias=True, activation=None, normalize_embedding=False):
        super(GraphConv, self).__init__()
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None
        self.activation = activation
        self.normalize_embedding = normalize_embedding

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', param=0.2)
        nn.init.xavier_uniform_(self.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
            
        adj = adj.clamp(min=0.0)
        
        # Add self-loops to adjacency matrix (Ã = A + I)
        batch_size, num_nodes, _ = adj.size()
        identity = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self = adj + identity
        
        # Compute degree matrix D̃ and normalize (D̃^{-1/2} Ã D̃^{-1/2})
        degree = adj_with_self.sum(dim=-1)  # (batch_size, num_nodes)
        degree = degree + 1e-5
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0  # Handle zero division if any
        d_inv_sqrt = d_inv_sqrt.unsqueeze(-1)  # (batch_size, num_nodes, 1)
        
        # Normalize adjacency matrix
        adj_normalized = adj_with_self * d_inv_sqrt  # Row scaling
        adj_normalized = adj_normalized * d_inv_sqrt.transpose(1, 2)  # Column scaling
        
        # Propagate and transform
        y = torch.matmul(adj_normalized, x)
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y += self.bias
            
        # Apply activation
        if self.activation is not None:
            y = self.activation(y)
        
        # Normalize embedding
        if self.normalize_embedding:
            y = nn.functional.normalize(y, p=2, dim=-1)
        
        return y
    

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, bn=True, dropout=0.0, use_projection_head=True, bias=True):
        super(Encoder, self).__init__()
        self.bn = bn
        self.num_layers = num_layers
        self.proj_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Linear(embedding_dim, embedding_dim))
        self.bias = bias
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.use_projection_head = use_projection_head
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(input_dim, hidden_dim, embedding_dim, num_layers, normalize=True, dropout=dropout, bias = bias)
        

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, normalize=False, dropout=0.0, bias=True):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, bias=bias, activation=self.act, normalize_embedding=normalize, dropout=dropout)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, bias=bias, activation=self.act, normalize_embedding=normalize, dropout=dropout) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, bias=bias, activation=None, normalize_embedding=normalize, dropout=dropout)
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).to(x.device)
        return bn_module(x)

    def forward(self, x, adj, **kwargs):
        x = self.conv_first(x, adj)
        if self.bn:
            x = self.apply_bn(x)
        for i in range(self.num_layers-2):
            residual = x
            x = self.conv_block[i](x,adj)
            if self.bn:
                x = self.apply_bn(x)
            x = x + residual
        x = self.conv_last(x,adj)
        out, _ = torch.max(x, dim=1)
        
        if self.use_projection_head:
            out=self.proj_head(out)
            
        return x, out

               
class Att_Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, bn=True, dropout=0.1, bias=True):
        super(Att_Decoder, self).__init__()
        self.bn = bn
        self.num_layers = num_layers
        self.bias = bias
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(input_dim, hidden_dim, embedding_dim, num_layers, normalize=True, dropout=dropout, bias = bias)
        

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, normalize=False, dropout=0.0, bias=True):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, bias=bias, activation=self.act, normalize_embedding=normalize, dropout=dropout)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, bias=bias, activation=self.act, normalize_embedding=normalize, dropout=dropout) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, bias=bias, activation=None, normalize_embedding=normalize, dropout=dropout)
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).to(x.device)
        return bn_module(x)

    def forward(self, x, adj, **kwargs):
        x = self.conv_first(x, adj)
        if self.bn:
            x = self.apply_bn(x)
        for i in range(self.num_layers-2):
            resuidal = x
            x = self.conv_block[i](x,adj)
            if self.bn:
                x = self.apply_bn(x)
            x = x + resuidal
        x = self.conv_last(x,adj)
        return x

    
class stru_Decoder(nn.Module):
    def __init__(self, dropout):
        super(stru_Decoder, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        x1=x.permute(0, 2, 1)
        x = torch.matmul(x,x1) 
        x=self.sigmoid(x)
        return x
    
    
class NetGe(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, bn=True, dropout=0.0, use_projection_head=True, bias=True):
        super(NetGe, self).__init__()
        
        self.shared_encoder = Encoder(input_dim, hidden_dim, embedding_dim, num_layers, bn=bn, dropout=0.0, use_projection_head=use_projection_head, bias=bias)
        self.attr_decoder =Att_Decoder(embedding_dim, hidden_dim, input_dim, num_layers, bn=bn, dropout=dropout, )
        self.struct_decoder = stru_Decoder(dropout)
    
    def forward(self, x, adj):
        # Reconstruction of Node Features and Adjacency Matrix
        x_fake= self.attr_decoder(x, adj)
        a_fake = self.struct_decoder(x, adj)
        
        # Embedding of Reconstructed Node Features and Adjacency Matrix
        x2,Feat_1=self.shared_encoder(x_fake, a_fake)

        return x_fake,a_fake,x2,Feat_1