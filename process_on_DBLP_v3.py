import numpy as np
from MyDataset import DBLPDataset
import time


# 实例化DBLP数据集
dataset = DBLPDataset()
g = dataset[0]

def get_metapath_dict(dataset):
    meta_paths_dict = dataset.meta_paths_dict
    return meta_paths_dict

def map_vertex_ID_optimized(graph):
    unique_id_mapping = {}
    reverse_mapping = {}  # 创建反向映射
    current_unique_id = 0
    
    for ntype in graph.ntypes:
        node_ids = graph.nodes(ntype).numpy()
        for original_node_id in node_ids:
            unique_id_mapping[current_unique_id] = [original_node_id, ntype]
            reverse_mapping[(original_node_id, ntype)] = current_unique_id  # 反向映射
            current_unique_id += 1
    
    return unique_id_mapping, reverse_mapping

def find_neighbors_optimized(graph, unique_id, edge_type, reverse_mapping):
    # 由于我们已经有了unique_id，可以直接从中获取原始节点号和类型
    original_node_id, node_type = unique_id_mapping[unique_id]
    
    # 使用DGL的API获取邻居节点ID
    neighbors = list(graph.successors(original_node_id, etype=edge_type).numpy())
    
    # 使用反向映射获取唯一ID
    neighbor_unique_ids = [reverse_mapping[(neighbor, edge_type[2])] for neighbor in neighbors]
    return neighbor_unique_ids



def build_metapath_instances(graph, metapaths_dict, unique_id_mapping, reverse_mapping):
    metapath_instances_dict = {}

    # 遍历每种元路径
    for metapath_key, etypes_sequence in metapaths_dict.items():
        instances_list = []

        # 第一种节点类型作为起点
        start_ntype = etypes_sequence[0][0]
        for start_node_id in range(graph.number_of_nodes(start_ntype)):
            # 从起点开始的当前路径实例
            current_path = [reverse_mapping[(start_node_id, start_ntype)]]

            # 递归构建元路径
            build_paths_recursively(graph, start_ntype, start_node_id, etypes_sequence, 0, current_path, instances_list, unique_id_mapping, reverse_mapping)

        metapath_instances_dict[metapath_key] = instances_list

    return metapath_instances_dict

def build_paths_recursively(graph, current_ntype, current_node_id, etypes_sequence, etype_index, current_path, instances_list, unique_id_mapping, reverse_mapping):
    # 基本情况：如果已经处理完所有的边类型
    if etype_index == len(etypes_sequence):
        instances_list.append(current_path.copy())
        return

    # 获取下一个边类型
    etype = etypes_sequence[etype_index]
    next_ntype = etype[2]  # 目标节点类型
    neighbors = list(graph.successors(current_node_id, etype=etype).numpy())

    for neighbor in neighbors:
        neighbor_unique_id = reverse_mapping[(neighbor, next_ntype)]
        # 添加邻居节点到当前路径并递归构建下一步的路径
        current_path.append(neighbor_unique_id)
        build_paths_recursively(graph, next_ntype, neighbor, etypes_sequence, etype_index + 1, current_path, instances_list, unique_id_mapping, reverse_mapping)
        # 回溯：移除最后一个节点，以便尝试下一个邻居
        current_path.pop()

start_time = time.time()
# 映射原图节点到唯一ID上，并创建反向映射的字典。正向映射为：{唯一ID：[原始ID，节点类型]}，反向映射为:{(原始ID，节点类型):唯一ID}
unique_id_mapping, reverse_mapping = map_vertex_ID_optimized(g)
metapaths_dict = get_metapath_dict(dataset)
metapaths_dict_test = {'AP': [('author', 'author-paper', 'paper')],'APA': [('author', 'author-paper', 'paper'), ('paper', 'paper-author', 'author')]}
metapath_instances_dict = build_metapath_instances(g, metapaths_dict_test, unique_id_mapping, reverse_mapping)
end_time = time.time()
print("运行时间:", end_time - start_time, "秒")
# 打印全部元路径实例的长度
for key in metapaths_dict_test:
    print(f"{key}元路径实例的数量:", len(metapath_instances_dict[key]))
