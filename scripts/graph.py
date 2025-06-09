#!/usr/bin/env python3
import json
import torch
import numpy as np
from collections import Counter

def analyze_graph_pytorch(json_file):
    print("Loading JSON...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("Building tensors...")
    # Create node mapping
    nodes = data['nodes']
    node_to_idx = {node['id']: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # Node features (levels)
    levels = torch.tensor([node['level'] for node in nodes])
    
    # Build sparse adjacency matrix
    edges = data['links']
    if edges:
        sources = [node_to_idx[edge['source']] for edge in edges]
        targets = [node_to_idx[edge['target']] for edge in edges]
        edge_indices = torch.tensor([sources, targets], dtype=torch.long)
        edge_values = torch.ones(len(edges))
        
        adjacency = torch.sparse_coo_tensor(
            edge_indices, edge_values, (num_nodes, num_nodes)
        ).coalesce()
    else:
        adjacency = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long), 
            torch.zeros(0), (num_nodes, num_nodes)
        )
    
    print("Computing statistics...")
    
    # Basic stats
    stats = {
        'nodes_total': num_nodes,
        'edges_total': len(edges),
        'density': len(edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
    }
    
    # Level distribution
    level_counts = torch.bincount(levels)
    if len(level_counts) > 1:
        stats['level1_nodes'] = level_counts[1].item()
    if len(level_counts) > 2:
        stats['level2_nodes'] = level_counts[2].item()
    
    # Degree analysis (vectorized)
    if adjacency._nnz() > 0:
        # Convert to dense for easier computation (only if not too large)
        if num_nodes < 50000:
            adj_dense = adjacency.to_dense()
            in_degrees = adj_dense.sum(dim=0)
            out_degrees = adj_dense.sum(dim=1)
        else:
            # For very large graphs, use sparse operations
            in_degrees = torch.sparse.sum(adjacency, dim=0).to_dense()
            out_degrees = torch.sparse.sum(adjacency, dim=1).to_dense()
        
        stats['avg_in_degree'] = in_degrees.float().mean().item()
        stats['avg_out_degree'] = out_degrees.float().mean().item()
        stats['max_in_degree'] = in_degrees.max().item()
        stats['max_out_degree'] = out_degrees.max().item()
        
        # Isolated nodes
        total_degrees = in_degrees + out_degrees
        stats['isolated_nodes'] = (total_degrees == 0).sum().item()
    else:
        stats.update({
            'avg_in_degree': 0, 'avg_out_degree': 0,
            'max_in_degree': 0, 'max_out_degree': 0,
            'isolated_nodes': num_nodes
        })
    
    # Edge types
    relations = [edge['relation'] for edge in edges]
    stats['edge_types'] = dict(Counter(relations))
    
    # Hierarchical structure (vectorized)
    level1_mask = (levels == 1)
    level2_mask = (levels == 2)
    
    if adjacency._nnz() > 0:
        edge_sources = adjacency.indices()[0]
        edge_targets = adjacency.indices()[1]
        
        l1_sources = level1_mask[edge_sources]
        l2_targets = level2_mask[edge_targets]
        l1_to_l2 = (l1_sources & l2_targets).sum().item()
        
        l2_sources = level2_mask[edge_sources]
        l1_targets = level1_mask[edge_targets]
        l2_to_l1 = (l2_sources & l1_targets).sum().item()
        
        stats['level1_to_level2_edges'] = l1_to_l2
        stats['level2_to_level1_edges'] = l2_to_l1
    else:
        stats['level1_to_level2_edges'] = 0
        stats['level2_to_level1_edges'] = 0
    
    # Simple clustering approximation (for speed)
    if adjacency._nnz() > 0 and num_nodes < 10000:
        # Only for smaller graphs due to memory
        try:
            adj_dense = adjacency.to_dense()
            adj_undirected = adj_dense + adj_dense.t()
            adj_undirected = (adj_undirected > 0).float()
            
            # Clustering coefficient approximation
            degrees = adj_undirected.sum(dim=1)
            triangles = torch.diag(torch.mm(torch.mm(adj_undirected, adj_undirected), adj_undirected))
            possible_triangles = degrees * (degrees - 1) / 2
            clustering = torch.where(possible_triangles > 0, triangles / possible_triangles, torch.zeros_like(triangles))
            stats['avg_clustering'] = clustering.mean().item()
        except:
            stats['avg_clustering'] = 0
    else:
        stats['avg_clustering'] = 0
    
    # Sparsity
    stats['sparsity'] = 1 - stats['density']
    
    # Node attributes
    stats['node_attributes'] = list(nodes[0].keys()) if nodes else []
    
    return stats

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_graph_torch.py unweighted_graph.json")
        sys.exit(1)
    
    results = analyze_graph_pytorch(sys.argv[1])
    
    print("\n=== GRAPH ANALYSIS RESULTS ===")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # Quick assessment
    print("\n=== QUICK ASSESSMENT ===")
    print(f"Average degree: {results['avg_out_degree']:.2f}")
    print(f"Sparsity: {results['sparsity']:.4f}")
    print(f"Clustering: {results['avg_clustering']:.4f}")
    if 'level1_nodes' in results and 'level2_nodes' in results:
        print(f"Hierarchy ratio: {results['level1_nodes']/results['level2_nodes']:.2f}")