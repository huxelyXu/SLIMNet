from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(dist_emb) * C.view(-1, 1)
        v = self.lin(v)
        e = v[j] * W
        return e


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = Linear(num_filters, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, e, edge_index):
        _, i = edge_index
        out = scatter(e, i, dim=0)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        return v + out


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels+6, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch):
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v) #58, 4
        u = scatter(v, batch, dim=0) #1,

        return u

class softnet(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(softnet, self).__init__()
        self.monomer1 = Linear(hidden_channels, hidden_channels // 2)
        self.monomer2 = Linear(hidden_channels // 2, out_channels)
        self.chain1 = Linear((hidden_channels // 2)+5, hidden_channels // 2)
        self.chain2 = Linear(hidden_channels // 2, 2*out_channels)
        self.order1 = Linear((hidden_channels // 2)+1, hidden_channels // 2)
        self.order2 = Linear(hidden_channels // 2, out_channels)

        self.act = torch.nn.Softplus()
        self.out_channels = out_channels

    def forward(self, x, chain, order):
        x = self.act(self.monomer1(x))
        monomer = self.act(self.monomer2(x))
        # print(x.shape)
        # print(chain.shape)

        x_chain = torch.concat((x, chain), dim=1)
        # print(x_chain.shape)
        x_chain = self.act(self.chain1(x_chain))
        x_chain = self.act(self.chain2(x_chain))
        x_chain = torch.log(x_chain + 1 + 1e-6)

        x_order = torch.concat((x, order), dim=1)
        # print(x_order.shape)
        x_order = self.act(self.order1(x_order))
        x_order = self.act(self.order2(x_order))
        x_order = torch.log(x_order + 1e-6)

        # print(monomer.shape)
        # print(monomer)
        # print(x_chain.shape)
        # print(x_chain)
        # print(x_chain[:, :self.out_channels].shape)
        # print(x_chain[:, :self.out_channels])
        # print(x_chain[:, self.out_channels:].shape)
        # print(x_chain[:, self.out_channels:])
        # print(torch.pow(x_chain[:, :self.out_channels], x_chain[:, self.out_channels:]))
        # input()


        phi_gauss = monomer * torch.pow(x_chain[:, :self.out_channels], x_chain[:, self.out_channels:])
        # print(phi_gauss.shape)
        # print(x_order.shape)
        # input()

        return phi_gauss + x_order



class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class SchNet(torch.nn.Module):
    r"""
        The re-implementation for SchNet from the `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            num_layers (int, optional): The number of layers. (default: :obj:`6`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Output embedding size. (default: :obj:`1`)
            num_filters (int, optional): The number of filters to use. (default: :obj:`128`)
            num_gaussians (int, optional): The number of gaussians :math:`\mu`. (default: :obj:`50`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`).
    """
    def __init__(self, energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, out_channels=1, num_filters=128, num_gaussians=50):
        super(SchNet, self).__init__()

        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])
        
        self.update_u = update_u(hidden_channels, out_channels)
        self.softnet = softnet(hidden_channels, out_channels=4)
        

        self.reset_parameters()

    def reset_parameters(self):

        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        chain, order = batch_data.chain, batch_data.order

        if self.energy_and_force:
            pos.requires_grad_()

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)

        v = self.init_v(z)

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(v, dist, dist_emb, edge_index)
            v = update_v(v,e, edge_index)

        
        mol_feature = scatter(v, batch, dim=0)
        # print(v.shape)
        tmp = torch.zeros((v.shape[0], chain.shape[0])).cuda()
        chain_tmp = tmp + chain
        # print(chain.shape)
        tmp = torch.zeros((v.shape[0], order.shape[0])).cuda()
        order_tmp = tmp + order
        # print(order.shape)
        v = torch.cat((v, chain_tmp, order_tmp), dim=1)
        # print(v.shape)
        # print(self.hidden_channels + chain.shape[1] + order.shape[1])
        # input()
        #print(torch.cat))
        


        mono_y = self.update_u(v, batch)
        poly_y = self.softnet(mol_feature,
                              chain.reshape(len(batch.unique()), -1),
                              order.reshape(len(batch.unique()), -1)
                              )

        return mono_y, poly_y


