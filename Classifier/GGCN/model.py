# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 08:22:03 2023

@author: dell
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedGraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedGraphConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Linear(in_channels, out_channels)
        self.U = nn.Linear(in_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        h = self.W(x)
        m = torch.matmul(adj, h)
        r = self.sigmoid(self.U(x))
        h_prime = torch.mul(m, r)
        return h_prime
    
class GatedGraphConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GatedGraphConvNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv1 = GatedGraphConvLayer(in_channels, hidden_channels)
        self.conv2 = GatedGraphConvLayer(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x
    
    


class GatedGraphConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GatedGraphConvNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv1 = GatedGraphConvLayer(in_channels, hidden_channels)
        self.conv2 = GatedGraphConvLayer(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.relu(self.conv2(x, adj))
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        
        # 可解释性代码
        gate1 = self.conv1.sigmoid(self.conv1.U(x))
        gate2 = self.conv2.sigmoid(self.conv2.U(x))
        return x, gate1, gate2



import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(adj_matrix):
    G = nx.from_numpy_matrix(adj_matrix.detach().cpu().numpy())
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()

def visualize_gates(gate1, gate2,save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Gate 1")
    plt.bar(range(len(gate1)), gate1.detach().cpu().numpy())
    plt.subplot(1, 2, 2)
    plt.title("Gate 2")
    plt.bar(range(len(gate2)), gate2.detach().cpu().numpy())
    plt.show()
    
    

