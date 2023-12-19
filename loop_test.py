
import torch
from simhash import *

def attn_new(query_layer, key_layer, bands, table_size, num_hashes, collision_matrices):
    # Prepare empty attention scores
    batch_size, num_heads, seq_len, _ = query_layer.shape
    attention_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=query_layer.device)

    # Loop through batches and heads
    for batch in range(batch_size):
        for head in range(num_heads):
            q = query_layer[batch, head, :, :]
            k = key_layer[batch, head, :, :]
            
            # stack vertically for LSH
            # stacked = torch.cat((q, k), dim=0)
            # lsh = LSH(bands=bands, table_size=table_size, num_hashes=num_hashes, feature_size=stacked.size(1))
            
            # collision_matrix = lsh.do_lsh(stacked)  # returns a seq_len x seq_len matrix
            collision_matrix = collision_matrices[batch, head]
            
            # Use boolean indexing to find collided indices
            collided_indices = torch.triu_indices(seq_len, seq_len, offset=0, device=query_layer.device)
            collided_mask = collision_matrix[collided_indices[0], collided_indices[1]] == 1
            
            # Filter the indices where collisions occurred
            q_indices = collided_indices[0][collided_mask]
            k_indices = collided_indices[1][collided_mask]
            
            # Compute the dot products for collided pairs
            q_vecs = q[q_indices]
            k_vecs = k[k_indices]
            attn_scores = (q_vecs * k_vecs).sum(dim=1)
            
            # Assign the computed attention scores
            attention_scores[batch, head, q_indices, k_indices] = attn_scores
            attention_scores[batch, head, k_indices, q_indices] = attn_scores  # symmetric assignment

    return attention_scores

def attn_old(query_layer, key_layer, bands, table_size, num_hashes, collision_matrices):
    # Prepare empty attention scores
    batch_size, num_heads, seq_len, _ = query_layer.shape
    attention_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=query_layer.device)

    # Loop through batches and heads
    for batch in range(batch_size):
        for head in range(num_heads):
            q = query_layer[batch, head, :, :]
            k = key_layer[batch, head, :, :]
            
            # stack vertically for LSH
            # stacked = torch.cat((q, k), dim=0)
            # lsh = LSH(bands=bands, table_size=table_size, num_hashes=num_hashes, feature_size=stacked.size(1))
            
            # collision_matrix = lsh.do_lsh(stacked) # returns a seq_len x seq_len matrix
            collision_matrix = collision_matrices[batch, head]    
            
            # dot product between Q & K vectors that collided
            for q_index in range(seq_len):
                for k_index in range(q_index, seq_len):
                    if collision_matrix[q_index, k_index] == 1:
                        q_vec = q[q_index, :]
                        k_vec = k[k_index, :]
                        
                        attn_score = torch.dot(q_vec, k_vec)
                        attention_scores[batch, head, q_index, k_index] = attn_score
                        attention_scores[batch, head, k_index, q_index] = attn_score
                        
    return attention_scores

def precompute_collision_matrices(query_layer, key_layer, bands, table_size, num_hashes):
    batch_size, num_heads, seq_len, _ = query_layer.shape
    collision_matrices = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=query_layer.device)

    # Loop through batches and heads
    for batch in range(batch_size):
        for head in range(num_heads):
            q = query_layer[batch, head, :, :]
            k = key_layer[batch, head, :, :]
            
            # Stack vertically for LSH
            stacked = torch.cat((q, k), dim=0)
            lsh = LSH(bands=bands, table_size=table_size, num_hashes=num_hashes, feature_size=stacked.size(1))
            
            # Compute and store the collision matrix
            collision_matrix = lsh.do_lsh(stacked)  # returns a seq_len x seq_len matrix
            collision_matrices[batch, head] = collision_matrix

    return collision_matrices

torch.manual_seed(0)

query = torch.rand((1, 2, 20, 64))
key = torch.rand((1, 2, 20, 64))

collision_matrices = precompute_collision_matrices(query, key, 8, 50, 4)

import time

# Time attn_old
start_old = time.perf_counter()
old_res = attn_old(query, key, 8, 50, 4, collision_matrices)
end_old = time.perf_counter()

# Time attn_new
start_new = time.perf_counter()
new_res = attn_new(query, key, 8, 50, 4, collision_matrices)
end_new = time.perf_counter()

# Calculate and print the times
old_time = end_old - start_old
new_time = end_new - start_new

print(f"Old function took {old_time:.6f} seconds")
print(f"New function took {new_time:.6f} seconds")

print("All close?", old_res.allclose(new_res))