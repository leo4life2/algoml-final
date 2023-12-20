import torch
from torch.nn.functional import softmax
import torch.profiler as profile
from tqdm import tqdm
import matplotlib

def simhash(vectors: torch.Tensor, bands: int, table_size: int, num_hashes: int):
    # stub
    return torch.randn(vectors.size(0), vectors.size(0))
    

class BertLSHSelfAttention:    
    def __init__(self, query_layer: torch.Tensor, key_layer: torch.Tensor):        
        self.query_layer = query_layer
        self.key_layer = key_layer
        self.Q_seqlen, self.K_seqlen = query_layer.size(2), key_layer.size(2)
        self.attention_scores = None

    def compute_attention(self, q_vector: torch.Tensor, k_vector: torch.Tensor, scale: float):        
        # Implement your attention computation logic here
        # For example, scaled dot-product attention
        print(q_vector.size())
        print(k_vector.size())
        matmul = torch.dot(q_vector, k_vector)
        return matmul

    
    def __call__(self, bands: int, table_size: int, num_hashes: int):
        batch_size, num_heads, qlen, dim = self.query_layer.shape
        klen = self.key_layer.size(2)
        scale = 1.0 / (dim ** 0.5)
        self.attention_scores = torch.zeros(batch_size, num_heads, qlen, klen)

        # 1. stack Q, K vertically
        for batch in range(batch_size):
            for head in range(num_heads): 
                # Reshape Q and K to [qlen, feature_size]q
                Q = self.query_layer[batch, head].view(qlen, -1)
                K = self.key_layer[batch, head].view(klen, -1)
                print(Q,K)
                
                # Stack Q and K vertically
                QK = torch.vstack([Q, K])
                print(QK)
        
                # 2. Get simhash collision matrix
                collision_matrix = simhash(QK, bands, table_size, num_hashes)
                
                # 3. compute scores
                for i in tqdm(range(qlen), desc="Q Loop"):
                    for j in tqdm(range(klen), desc="K Loop"):
                        if collision_matrix[i, j + qlen]:  # Check collision for Q[i] and K[j]
                            self.attention_scores[batch, head, i, j] = self.compute_attention(Q[i], K[j], scale)
                
                print(self.attention_scores)
                
        return self.attention_scores

# Example usage
num_batches = 1
num_heads = 1
seq_length = 10
feature_size = 512

query_layer = torch.randn(num_batches, num_heads, seq_length, feature_size)
key_layer = torch.randn(num_batches, num_heads, seq_length, feature_size)

# Define bands, table_size, num_hashes for SimHash
bands = 10
table_size = 100
num_hashes = 5
num_vectors = 10
vectors = torch.randn(10, feature_size)

attention_module = BertLSHSelfAttention(query_layer, key_layer)
attention_scores = attention_module(bands, table_size, num_hashes)

