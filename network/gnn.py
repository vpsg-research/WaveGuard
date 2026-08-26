import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config()
#将频域图像构造成图，建模空间邻接关系
def build_graph(images):
    graph_list = []
    batch_size = images.size(0)
    for i in range(batch_size):
        image = images[i].squeeze(0)# shape: [128, 128]
        num_nodes = 128 * 128
        x = image.view(-1, 1).to(cfg.device)# 每个像素为一个节点特征，flatten 后变为 [16384, 1]
        edge_index = []
        for row in range(128):
            for col in range(128):
                node_id = row * 128 + col
                if col < 127:
                    right_node_id = row * 128 + (col + 1)
                    edge_index.append([node_id, right_node_id])
                if row < 127:
                    bottom_node_id = (row + 1) * 128 + col
                    edge_index.append([node_id, bottom_node_id])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(cfg.device)
        data = Data(x=x, edge_index=edge_index)
        graph_list.append(data)
    return graph_list

#提取结构特征，用于对比原图与嵌入图的一致性
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = gnn.GCNConv(1, 16)
        self.conv2 = gnn.GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
