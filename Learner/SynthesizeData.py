# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
import igraph as ig
import numpy as np
from scipy.stats import truncnorm
from pgmpy.models.BayesianNetwork import BayesianNetwork as BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from Tools import Graph

numeachclass = {'small': 50, 'medium': 50, 'large': 50, 'vlarge': 50}
nodenumclass = {'small': (11, 20), 'medium': (21, 50), 'large': (51, 100), 'vlarge': (101, 1000)}
avgInDegree = (1.2, 1.7) # == avgOutDegree == #Edges/#Nodes, lower/ upper bound

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.
       partly stolen from https://github.com/xunzheng/notears
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF
        restrict_indegree (bool): set True to restrict nodes' indegree, specifically,
            for ER: from skeleton to DAG, randomly acyclic orient each edge.
                    if a node's degree (in+out) is large, we expect more of its degree is allocated for out, less for in.
                    so permute: the larger degree, the righter in adjmat, and
                                after the lower triangle, the lower upper bound of in-degree
            for SF: w.r.t. SF natrue that in-degree may be exponentially large, but out-degree is always low,
                    explicitly set the MAXTOLIND. transpose in/out when exceeding. refer to _transpose_in_out(B)
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _acyclic_orientation(B_und):
        # pre-randomized here. to prevent i->j always with i>j
        return np.tril(B_und, k=-1)

    def _remove_isolating_node(B):
        non_iso_index = np.logical_or(B.any(axis=0), B.any(axis=1))
        return B[non_iso_index][:, non_iso_index]

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
    elif graph_type == 'SF':
        G_und = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False, outpref=True, power=-3)
    else:
        raise ValueError('unknown graph type')

    B_und = _graph_to_adjmat(G_und)
    B_und = _random_permutation(B_und)
    B = _acyclic_orientation(B_und)
    B = _remove_isolating_node(B)
    B_perm = _random_permutation(B).astype(int)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def simulate_cards(B, card_param=None):
    if card_param == None:
        card_param = {'lower': 2, 'upper': 6, 'mu': 2, 'basesigma': 1.5} # truncated normal distribution
    def _max_peers():
        '''
        why we need this: to calculate cpd of a node with k parents,
            the conditions to be enumerated is the production of these k parents' cardinalities
            which will be very exponentially slow w.r.t. k.
            so we want that, if a node has many parents (large k), these parents' cardinalities should be small
        i also tried to restrict each node's indegree at the graph sampling step,
            but i think that selection-bias on graph structure is worse than that on cardinalities
        an alternative you can try:
            use SEM to escape from slow forwards simulation, and then discretize.

        denote peers_num: peers_num[i, j] = k (where k>0),
            means that there are k parents pointing to node i, and j is among these k parents.
        max_peers = peers_num.max(axis=0): the larger max_peers[j], the smaller card[j] should be.
        :return:
        '''
        in_degrees = B.sum(axis=0)
        peers_num = in_degrees[:, None] * B.T
        return peers_num.max(axis=0)

    lower, upper, mu, basesigma = card_param['lower'], card_param['upper'], card_param['mu'], card_param['basesigma']
    sigma = basesigma / np.exp2(_max_peers()) ########## simply _max_peers() !
    cards = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).\
        rvs(size=B.shape[0]).round().astype(int)
    return cards
    
def simulate_discrete(B, n, card_param=None, alpha_param=None):
    if alpha_param == None:
        alpha_param = {'lower': 0.1, 'upper': 1.0} #1.2? # larger alpha, smaller variation. is uniform good?
    def _random_alpha():
        return np.random.uniform(alpha_param['lower'], alpha_param['upper'])
    def _dirichlet(alpha, size):
        probs = np.random.dirichlet(np.ones(size) * alpha)
        probs[-1] = 1 - probs[:-1].sum() # to prevent numerical issues (not sum to 1)
        return probs
    
    cards = simulate_cards(B, card_param=card_param)
    diEdges = list(map(lambda x: (str(x[0]), str(x[1])), np.argwhere(B == 1))) # list of tuples
    bn = BayesianModel(diEdges) # so isolating nodes will echo error
    for node in range(len(cards)):
        parents = np.where(B[:, node] == 1)[0].tolist()
        parents_card = [cards[prt] for prt in parents]
        rand_ps = np.array([_dirichlet(_random_alpha(), cards[node]) for _ in range(int(np.prod(parents_card)))]).T.tolist()
        cpd = TabularCPD(str(node), cards[node], rand_ps, evidence=list(map(str, parents)), evidence_card=parents_card)
        bn.add_cpds(cpd)
    inference = BayesianModelSampling(bn)
    df = inference.forward_sample(size=n)
    topo_order = list(map(int, df.columns))
    topo_index = [-1] * len(topo_order)
    for ind, node in enumerate(topo_order): topo_index[node] = ind
    return df.to_numpy()[:, topo_index].astype(int)

def simulate_graphs():
    os.makedirs('./synthetics/graph/', exist_ok=True)
    os.makedirs('./synthetics/graph/imgs/', exist_ok=True)
    for cname in nodenumclass.keys():
        lower, upper = nodenumclass[cname]
        for gtype in ['ER', 'SF']:
            for id in range(numeachclass[cname]):
                synname = f'{cname}_{gtype}_{id}'
                print('now simulating graph structure for', synname)
                d = np.random.randint(lower, upper)
                s0 = np.round(np.random.uniform(avgInDegree[0], avgInDegree[1]) * d).astype(int)
                B = simulate_dag(d, s0, gtype)
                graphtxtpath = f'./synthetics/graph/{synname}.txt'
                np.savetxt(graphtxtpath, B, fmt='%i')
                dig = Graph.DiGraph(graphtxtpath)
                tforks, vstrucs, allIdentifiableEdges = dig.tforks, dig.vstrucs, dig.IdentifiableEdges

                DirectedEdges = list(map(tuple, np.argwhere(B == 1)))
                NodeIDs = list(range(len(B)))
                vedges = set()
                for vv in vstrucs:
                    vedges.add((vv[1], vv[0]))
                    vedges.add((vv[2], vv[0]))

                G = nx.DiGraph()
                G.add_edges_from(DirectedEdges)
                G.add_nodes_from(NodeIDs)
                pos = graphviz_layout(G, prog='dot')
                nx.draw_networkx_nodes(G, pos, node_color='lightblue')
                nx.draw_networkx_labels(G, pos, font_size=7)
                nx.draw_networkx_edges(G, pos, edgelist=set(DirectedEdges) - vedges)
                nx.draw_networkx_edges(G, pos, edgelist=vedges, edge_color='green')

                plt.title(f'{synname}, {len(NodeIDs)} nodes, {len(DirectedEdges)} edges, {len(vstrucs)} V, {len(tforks) - len(vstrucs)} nonV')
                plt.tight_layout()
                plt.savefig(f'./synthetics/graph/imgs/{synname}.png')
                plt.clf()


def simulate_data_discrt():
    os.makedirs('./synthetics/data/', exist_ok=True)
    for cname in nodenumclass.keys():
        for gtype in ['ER', 'SF', ]:
            for id in range(numeachclass[cname]):
                synname = f'{cname}_{gtype}_{id}'
                print('now forward sampling data for', synname)
                B = np.loadtxt(f'./synthetics/graph/{synname}.txt')
                np.save(f'./synthetics/data/{synname}.npy', simulate_discrete(B, n=10000))


if __name__ == '__main__':
    ########### for training synthetic data ###########
    simulate_graphs()
    simulate_data_discrt()
    # ########### for training synthetic data ###########