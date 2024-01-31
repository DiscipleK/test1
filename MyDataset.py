import pickle
import torch as th
import numpy as np
import dgl
import os
from dgl.data import DGLBuiltinDataset # 引入Dataset的子集
from dgl.data.utils import idx2mask, load_graphs, save_graphs
'''
* idx2mask:将索引列表转换为掩码张量
* load_graphs, save_graphs 保存和加载异构图
'''

__all__ = ['IMDBDataset', 'ACMDataset', 'DBLPDataset']

# 定义超图数据管道
class HeteroDataset(DGLBuiltinDataset):
    """
    Examples
    >>> dataset = IMDB4GTNDataset()
    >>> graph = dataset[0]
    """
    # 定义数据集（GitHub分享的已经预处理过的数据集保存在亚马逊云上）和文件路径
    def __init__(self, name, canonical_etypes, target_ntype, raw_dir=None, force_reload=False, verbose=False,
                    transform=None):
        assert name in ['dblp4GTN', 'acm4GTN', 'imdb4GTN']
        self._canonical_etypes = canonical_etypes
        self._target_ntype = target_ntype
        if raw_dir is None:
            raw_dir = './openhgnn/dataset'
        super(HeteroDataset, self).__init__(
            name,
            url='https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{}.zip'.format(name),
            raw_dir=raw_dir,
            force_reload=force_reload, verbose=verbose, transform=transform)

    # 读出原始数据并初始化为异质图
    def process(self):
        target_ntype = self.target_ntype
        canonical_etypes = self._canonical_etypes

        with open(self.raw_path + '/node_features.pkl', 'rb') as f:
            node_features = pickle.load(f)
        with open(self.raw_path + '/edges.pkl', 'rb') as f:
            edges = pickle.load(f)
        with open(self.raw_path + '/labels.pkl', 'rb') as f:
            labels = pickle.load(f)

        num_nodes = edges[0].shape[0]
        '''
        edges[0] 是边数据列表中的第一类边的稀疏矩阵表示。
        edges[0].shape[0] 则是该稀疏矩阵的行数，也就是节点的总数
        '''
        assert len(canonical_etypes) == len(edges)

        ntype_mask = dict()
        ntype_idmap = dict()
        ntypes = set()
        data_dict = {}

        # create dgl graph
        for etype in canonical_etypes:
            ntypes.add(etype[0])
            ntypes.add(etype[2])
        for ntype in ntypes: # 初始化关于节点类型的掩码和标识映射的数据结构
            ntype_mask[ntype] = np.zeros(num_nodes, dtype=bool)
            # 节点类型映射
            ntype_idmap[ntype] = np.full(num_nodes, -1, dtype=int)
        for i, etype in enumerate(canonical_etypes):
            src_nodes = edges[i].nonzero()[0] # 找到第 i 类边上存在的源节点的位置
            dst_nodes = edges[i].nonzero()[1]
            src_ntype = etype[0]
            dst_ntype = etype[2]
            ntype_mask[src_ntype][src_nodes] = True
            ntype_mask[dst_ntype][dst_nodes] = True
        for ntype in ntypes:
            # 最终的结果是为每个节点类型创建了一个从零开始的新的、连续的索引映射
            # 这在处理图数据时常常是有用的
            ntype_idx = ntype_mask[ntype].nonzero()[0]
            #  # 使用 'nonzero()' 获取布尔掩码为 True 的索引，并选择结果元组的第一个元素（[0]）
            ntype_idmap[ntype][ntype_idx] = np.arange(ntype_idx.size)
        for i, etype in enumerate(canonical_etypes):
            # 每个边类型，获取稀疏矩阵中非零元素的坐标，分别得到源节点和目标节点的索引数组
            src_nodes = edges[i].nonzero()[0]
            dst_nodes = edges[i].nonzero()[1]
            # 根据边类型的信息，获取源节点类型 (src_ntype) 和目标节点类型 (dst_ntype)
            src_ntype = etype[0]
            dst_ntype = etype[2]
            data_dict[etype] = \
                (th.from_numpy(ntype_idmap[src_ntype][src_nodes]),# 将映射后的节点索引构建为元组
                    th.from_numpy(ntype_idmap[dst_ntype][dst_nodes]))
        g = dgl.heterograph(data_dict)

        # 处理图数据的分割和标签
        all_label = np.full(g.num_nodes(target_ntype), -1, dtype=int)
        for i, split in enumerate(['train', 'val', 'code_test']):
            # 获取当前分割集合的节点索引和标签
            node = np.array(labels[i])[:, 0]# 将labels数组的第i行的第一个元素提取出来，并将其保存在node变量中
            label = np.array(labels[i])[:, 1]
            all_label[node] = label # 当前分割集合的标签赋值给对应节点
            g.nodes[target_ntype].data['{}_mask'.format(split)] = \
                th.from_numpy(idx2mask(node, g.num_nodes(target_ntype))).type(th.bool)
        # 将整体标签赋值给目标节点类型的节点
        g.nodes[target_ntype].data['label'] = th.from_numpy(all_label).type(th.long)

        # node feature
        node_features = th.from_numpy(node_features).type(th.FloatTensor)
        for ntype in ntypes:
            idx = ntype_mask[ntype].nonzero()[0]
            g.nodes[ntype].data['h'] = node_features[idx]

        self._num_classes = len(th.unique(g.nodes[self.target_ntype].data['label']))
        self._in_dim = g.ndata['h'][target_ntype].shape[1]
        self._g = g

    # 保存模型到磁盘
    def save(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        save_graphs(graph_path, self._g)

    # 磁盘中加载模型
    def load(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        gs, _ = load_graphs(graph_path)
        self._g = gs[0]

    @property
    def target_ntype(self):
        return self._target_ntype

    @property
    def category(self):
        return self._target_ntype
    # 节点类型
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def in_dim(self):
        return self._in_dim

    #数据集索引方法
    def __getitem__(self, idx):
        assert idx == 0
        return self._g

    # 数据集中数据长度方法
    def __len__(self):
        return 1

# 为不同的数据集定义不同的数据集类，并定义自己的元路径字典，后续可以通过基本路径拼出并统计
class DBLPDataset(HeteroDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        name = 'dblp4GTN'
        # 定义基本路径
        canonical_etypes = [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
                            ('paper', 'paper-conference', 'conference'), ('conference', 'conference-paper', 'paper')]
        target_ntype = 'author'
        super(DBLPDataset, self).__init__(name, canonical_etypes, target_ntype, raw_dir=raw_dir,
                                        force_reload=force_reload, verbose=verbose, transform=transform)
    @property
    def canonical_etypes(self):
        return self._canonical_etypes
    # 定义元路径字典
    @property
    def meta_paths_dict(self):
        return {
                'AP': [('author', 'author-paper', 'paper')],
                'APA': [('author', 'author-paper', 'paper'),
                        ('paper', 'paper-author', 'author')],
                'APC': [('author', 'author-paper', 'paper'),
                        ('paper', 'paper-conference', 'conference')],
                'APCP': [('author', 'author-paper', 'paper'),
                        ('paper', 'paper-conference', 'conference'),
                        ('conference', 'conference-paper', 'paper')],
                'APCPA': [('author', 'author-paper', 'paper'),
                        ('paper', 'paper-conference', 'conference'),
                        ('conference', 'conference-paper', 'paper'),
                        ('paper', 'paper-author', 'author')],
                }


class ACMDataset(HeteroDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        name = 'acm4GTN'
        canonical_etypes = [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
                            ('paper', 'paper-subject', 'subject'), ('subject', 'subject-paper', 'paper')]
        target_ntype = 'paper'
        super(ACMDataset, self).__init__(name, canonical_etypes, target_ntype, raw_dir=raw_dir,
                                        force_reload=force_reload, verbose=verbose, transform=transform)

    @property
    def canonical_etypes(self):
        return self._canonical_etypes

    @property
    def meta_paths_dict(self):
        return {'PAPSP': [('paper', 'paper-author', 'author'),
                            ('author', 'author-paper', 'paper'),
                            ('paper', 'paper-subject', 'subject'),
                            ('subject', 'subject-paper', 'paper')],
                'PAP': [('paper', 'paper-author', 'author'),
                        ('author', 'author-paper', 'paper')],
                'PSP': [('paper', 'paper-subject', 'subject'),
                        ('subject', 'subject-paper', 'paper')]
                }


class IMDBDataset(HeteroDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        name = 'imdb4GTN'
        canonical_etypes = [('movie', 'movie-director', 'director'), ('director', 'director-movie', 'movie'),
                            ('movie', 'movie-actor', 'actor'), ('actor', 'actor-movie', 'movie')]
        target_ntype = 'movie'
        super(IMDBDataset, self).__init__(name, canonical_etypes, target_ntype, raw_dir=raw_dir,
                                            force_reload=force_reload, verbose=verbose, transform=transform)

    @property
    def canonical_etypes(self):
        return self._canonical_etypes

    @property
    def meta_paths_dict(self):
        return {
                'MA':[('movie', 'movie-actor', 'actor')],
                'MAM': [('movie', 'movie-actor', 'actor'),
                        ('actor', 'actor-movie', 'movie')],
                'MD':[('movie', 'movie-director', 'director')],
                'MDM': [('movie', 'movie-director', 'director'),
                        ('director', 'director-movie', 'movie')]
                }
