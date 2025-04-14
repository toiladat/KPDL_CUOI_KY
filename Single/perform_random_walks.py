import random
from itertools import combinations

def perform_random_walks(graph, num_walks=10, walk_length=10):
    walks = []
    nodes = list(graph.keys())
    
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                curr = walk[-1]
                neighbors = list(graph.get(curr, {}).keys())
                
                if not neighbors:
                    break
                
                # Chọn neighbor có trọng số lớn nhất
                weights = [graph[curr][nbr] for nbr in neighbors]
                max_weight_neighbor = neighbors[weights.index(max(weights))]  # Chọn neighbor có trọng số cao nhất
                
                walk.append(max_weight_neighbor)
            
            walks.append(walk)
    
    return walks
