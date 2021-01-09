import torch
import numpy as np
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.distributions import Categorical
from copy import deepcopy
import sys



def chain_reaction(board):
    left = torch.stack((torch.arange(board.shape[0])[1:-1],torch.LongTensor([0]*(board.shape[0]-2))),1)
    right = torch.stack((torch.arange(board.shape[0])[1:-1],torch.LongTensor([board.shape[1]-1]*(board.shape[0]-2))),1)
    top = torch.stack((torch.LongTensor([0]*(board.shape[1]-2)),torch.arange(board.shape[1])[1:-1]),1)
    bottom = torch.stack((torch.LongTensor([board.shape[0]-1]*(board.shape[1]-2)),torch.arange(board.shape[1])[1:-1]),1)
    edges = torch.cat((left,right,top,bottom),0)
    rows_edges = edges[:,0]
    cols_edges = edges[:,1]
    rows_corner = torch.LongTensor([0,0,board.shape[0]-1,board.shape[0]-1])
    cols_corner = torch.LongTensor([0,board.shape[1]-1,0,board.shape[1]-1])
    while ((board[1:-1,1:-1].abs() > 3).any()) | ((board[rows_edges,cols_edges].abs() > 2).any()) | ((board[rows_corner,cols_corner].abs() > 1).any()):
        if (board[1:-1,1:-1].abs() > 3).any():
            rows,cols = torch.where(board[1:-1,1:-1].abs() > 3)
            rows += 1
            cols += 1
            sign = board[rows,cols].sum()/board[rows,cols].sum().abs()
            rows_above = rows - 1
            rows_below = rows + 1
            cols_right = cols + 1
            cols_left = cols - 1
            rows_all = torch.cat((rows,rows,rows_above,rows_below))
            cols_all = torch.cat((cols_left,cols_right,cols,cols))
            opposite = torch.where(board[rows_all,cols_all]/sign < 0)
            board[rows_all[opposite],cols_all[opposite]] = 0 - board[rows_all[opposite],cols_all[opposite]]
            empty = torch.zeros((board.shape[0],board.shape[1])).expand(rows_all.shape[0],-1,-1).clone()
            empty[torch.arange(rows_all.shape[0]),rows_all,cols_all] = (1 * sign)
            board[rows,cols] -= (4 * sign)
            board = board + empty.sum(0)
            result = check_winner(board)
            if (result == 1) | (result == -1):
                break

        if (board[rows_edges,cols_edges].abs().abs() > 2).any():
            unstable = torch.where(board[rows_edges,cols_edges].abs() > 2)
            rows = rows_edges[unstable]
            cols = cols_edges[unstable]
            sign = board[rows,cols].sum()/board[rows,cols].sum().abs()
            rows_above = rows - 1
            rows_below = rows + 1
            cols_right = cols + 1
            cols_left = cols - 1
            rows_all = torch.cat((rows,rows,rows_above,rows_below))
            cols_all = torch.cat((cols_left,cols_right,cols,cols))
            valid = (rows_all >= 0) & (rows_all < board.shape[0]) & (cols_all >= 0) & (cols_all < board.shape[1])
            rows_all = rows_all[valid]
            cols_all = cols_all[valid]
            opposite = torch.where(board[rows_all,cols_all]/sign < 0)
            board[rows_all[opposite],cols_all[opposite]] = 0 - board[rows_all[opposite],cols_all[opposite]]
            empty = torch.zeros((board.shape[0],board.shape[1])).expand(rows_all.shape[0],-1,-1).clone()
            empty[torch.arange(rows_all.shape[0]),rows_all,cols_all] = (1 * sign)
            board[rows,cols] -= (3 * sign)
            board = board + empty.sum(0)
            result = check_winner(board)
            if (result == 1) | (result == -1):
                break

        if (board[rows_corner,cols_corner].abs() > 1).any():
            unstable = torch.where(board[rows_corner,cols_corner].abs() > 1)
            rows = rows_corner[unstable]
            cols = cols_corner[unstable]
            sign = board[rows,cols].sum()/board[rows,cols].sum().abs()
            rows_above = rows - 1
            rows_below = rows + 1
            cols_right = cols + 1
            cols_left = cols - 1
            rows_all = torch.cat((rows,rows,rows_above,rows_below))
            cols_all = torch.cat((cols_left,cols_right,cols,cols))
            valid = (rows_all >= 0) & (rows_all < board.shape[0]) & (cols_all >= 0) & (cols_all < board.shape[1])
            rows_all = rows_all[valid]
            cols_all = cols_all[valid]
            opposite = torch.where(board[rows_all,cols_all]/sign < 0)
            board[rows_all[opposite],cols_all[opposite]] = 0 - board[rows_all[opposite],cols_all[opposite]]
            empty = torch.zeros((board.shape[0],board.shape[1])).expand(rows_all.shape[0],-1,-1).clone()
            empty[torch.arange(rows_all.shape[0]),rows_all,cols_all] = (1 * sign)
            board[rows,cols] -= (2 * sign)
            board = board + empty.sum(0)
            result = check_winner(board)
            if (result == 1) | (result == -1):
                break
    return board


def get_available_moves(board,player):
    return torch.where(board/player >= 0)
    
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(8,64,2)
        self.tanh1 = nn.Tanh()
        self.conv2 = nn.Conv2d(64,32,2)
        self.tanh2 = nn.Tanh()
        self.conv3 = nn.Conv2d(32,8,2)
        self.tanh3 = nn.Tanh()
        self.fc1 = nn.Linear(8*6*3,54)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(8*6*3,54)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(54,10)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.tanh1(self.conv1(x))
        x = self.tanh2(self.conv2(x))
        x = self.tanh3(self.conv3(x))
        policy = x.view(8*6*3)
        policy = self.tanh4(self.fc1(policy))
        value = x.view(8*6*3)
        value = self.relu2(self.fc2(value))
        value = self.relu3(self.fc3(value))
        value = self.sigmoid(self.fc4(value))
        return policy,value
    
    
int_to_pos_dct = dict(enumerate(torch.from_numpy(np.indices((9,6))).flatten(1).T.tolist()))
pos_to_int_dct = {tuple(j):i for i,j in int_to_pos_dct.items()}


class Node():
    def __init__(self,state,term):
        """
        Edges columns available_pos_int,N,W,Q,P
        """
        rows,cols = get_available_moves(state[-2],term)
        available_moves_int = [pos_to_int_dct[tuple(i)] for i in torch.stack((rows,cols),1).tolist()]
        self.state = state.clone()
#         self.edges = {i:{'N':0,'W':0,'Q':0,'P':0} for i in available_moves_int}
        self.available_moves_int = torch.FloatTensor(available_moves_int)
        self.edges = torch.cat((self.available_moves_int.unsqueeze(1),torch.zeros_like(self.available_moves_int).expand(4,-1).clone().T),1)
        self.parent = None
        self.edge_root = None
        self.Children = {i:None for i in self.available_moves_int.int().tolist()}
        self.isleaf = True
        self.visit_count = 1
        self.state_value = 0
    def update(self,policy,value):
        self.edges[:,4] = policy[self.available_moves_int.int().tolist()]
        self.state_value = value
        
def selection(current_node,cpuct=4):
    if current_node.edges.shape[0] == 0:
        action = None
    else:
    #     UCT = (current_node.visit_count)/(1+current_node.edges[:,1])
        UCT = np.sqrt(current_node.visit_count)/(1+current_node.edges[:,1])
    #     UCT = torch.sqrt((current_node.edges[:,1].sum()))/(1+current_node.edges[:,1])
        action_indx = (current_node.edges[:,3] + cpuct * current_node.edges[:,4] * UCT).argmax()
        action = current_node.edges[action_indx][...,0].item()
        current_node.isleaf = False
    return action

def check_winner(board):
    if (board.min() < 0) & (board.max() <= 0) & (board.nonzero().shape[0] > 1):
        return -1
    elif (board.min() >= 0) & (board.max() > 0) & (board.nonzero().shape[0] > 1):
        return 1
    else:
        return None
    
def expand(current_node,action,term):
    while True:
        if check_winner(current_node.state[-2]) != None:
            new_node = None
            break
        
        if current_node.Children[action] != None:
            current_node.visit_count += 1
            current_node = current_node.Children[action][0]
            action = selection(current_node)
            if action == None:
                new_node = None
                break
            if term == player1:
                term = player2
            else:
                term = player1
            continue
        
        
        else:
            # Edge not used before (leaf)
            state = current_node.state[-2].clone()
            i,j = int_to_pos_dct[action]

                
            state[i,j] += term
            state = chain_reaction(state)

            stacked_states = torch.cat((current_node.state[1:-1],state.unsqueeze(0),player_flag.unsqueeze(0)*term),0)
            if term == player1:
                term = player2
            else:
                term = player1
            new_node = Node(stacked_states.clone(),term)

            
            new_node.edge_root = action
            new_node.parent = current_node

            policy,value = conv_net.forward(new_node.state.unsqueeze(0))
            policy = torch.softmax(policy,-1)
            new_node.update(policy,value)

            current_node.Children[action] = (new_node,action)
            break

    return new_node


def backup(new_node):
    current_node = new_node
    while True:
        parent = current_node.parent
        root_edge = current_node.edge_root
        parent.edges[torch.where(parent.edges[:,0] == root_edge),...,1] += 1
        parent.edges[torch.where(parent.edges[:,0] == root_edge),...,2] += current_node.state_value
        parent.edges[torch.where(parent.edges[:,0] == root_edge),...,3] = parent.edges[torch.where(parent.edges[:,0] == root_edge),...,2]/parent.edges[torch.where(parent.edges[:,0] == root_edge),...,1]
        current_node = parent
        if current_node.parent == None:
            break
            
def mcts_search(current_node,term,itr = 1600):
    for m in range(itr):
        action = selection(current_node)
        new_node = expand(current_node,action,term)
        if new_node == None:
            continue
        backup(new_node)
        


conv_net = ConvNet()
optimizer = optim.SGD(conv_net.parameters(),1e-3,weight_decay=1e-4)
episodes = 10000

for e in range(episodes):
    print('******************************',e,'*******************************')
    sys.stdout.flush()
    t_start = time.time()
    conv_net = ConvNet()
    policy_losses = []
    values = []
    player1 = 1
    player2 = -1
    tau = 1.75
    player_flag = torch.ones(9,6)

    term = player1
    stacked_states = torch.zeros(9,6).expand(7,9,6).clone()
    stacked_states = torch.cat((stacked_states,player_flag.unsqueeze(0)*term),0)
    state = torch.zeros((9,6))
    stacked_states = torch.cat((stacked_states[1:-1],state.unsqueeze(0),player_flag.unsqueeze(0)*term),0)

    root_node = Node(stacked_states,term)
    policy,value = conv_net.forward(root_node.state.unsqueeze(0))
    policy = torch.softmax(policy,-1)
    root_node.update(policy,value)
    print(root_node.state[-2])
    sys.stdout.flush()
    values.append(root_node.state_value)
    n_moves = 0
    while True:
        t1 = time.time()
        print('**************',n_moves,'*******************')
        sys.stdout.flush()
        winner = check_winner(root_node.state[-2])
        if winner == None:
            mcts_search(root_node,term,500)
            actual_action_indx = ((root_node.edges[:,1]**(1/tau)) / (root_node.edges[:,1]**(1/tau)).sum()).argmax().item()
            actual_action = root_node.edges[:,0][actual_action_indx].item()
            mcts_policy = torch.softmax(root_node.edges[:,1]/root_node.edges[:,1].max(),-1)

            available_acts = dict(zip(root_node.edges[:,0].tolist(),mcts_policy))
            default_acts = {i:0.0 for i in range(54)}
            default_acts.update(available_acts)
            mcts_policy_updated = torch.FloatTensor((list(default_acts.values())))

            policy_loss = -(mcts_policy_updated.T * torch.log(policy)).sum()
            policy_losses.append(policy_loss)
            root_node = root_node.Children[actual_action][0]
            print(term, 'played',int_to_pos_dct[actual_action])
            sys.stdout.flush()
            print(root_node.state[-2])
            sys.stdout.flush()
            values.append(root_node.state_value)
            root_node.parent = None
        else:
            print('game_ended')
            sys.stdout.flush()
            print('Winner',winner)
            sys.stdout.flush()
            break
        n_moves += 1
        if term == player1:
            term = player2
        else:
            term = player1
        t2 = time.time()
        print('took',t2-t1)
        sys.stdout.flush()
    t_end = time.time()
    print('total time took',t_end - t_start)
    sys.stdout.flush()

    print('....training')
    sys.stdout.flush()
    values = torch.stack(values).squeeze()
    m = torch.ones(values.shape[0])
    m[max(winner,0)::2] -= 1
    value_loss_all = ((values - m)**2).mean()
    print('value_loss_all',value_loss_all.item())
    sys.stdout.flush()
    policy_loss_all = torch.stack(policy_losses).mean()
    print('policy_loss_all',policy_loss_all.item())
    sys.stdout.flush()
    loss_all = policy_loss_all + value_loss_all
    print('loss_all',loss_all.item())
    sys.stdout.flush()
    optimizer.zero_grad()
    loss_all.backward()
    optimizer.step()
    if e % 2 == 0:
        torch.save(conv_net.state_dict(), '/home/cdsw/RL/chain_reaction_checkpoints/conv_net_' + str(e))