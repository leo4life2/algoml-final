import torch

def sign_hash_function(vectors, x):
    """Compute the sign hash for a given vector x using vectorized operations."""
    return torch.sign(torch.matmul(vectors, x))

class LSH:
    def __init__(self, bands, table_size, num_hashes, feature_size):
        self.bands = bands
        self.table_size = table_size
        self.num_hashes = num_hashes
        self.feature_size = feature_size
        self.random_vectors = torch.randn(self.num_hashes, self.bands, self.feature_size)
        # Initialize random coefficients for each hash function
        self.hash_coefficients = torch.randint(0, table_size, (num_hashes, bands))
        self.bucket_table = [{} for _ in range(self.num_hashes)] # use a dictionary so we don't store buckets with nothing at all
        
    def _uniform_random_hash(self, vector, hash_idx):
        """Apply a uniform random hash function to the vector using vectorized operations."""
        # Vectorized computation of the hash value
        hash_values = (vector > 0) * self.hash_coefficients[hash_idx]
        hash_value = hash_values.sum() % self.table_size
        return hash_value.item()

    def _hash_to_table(self, vectors):
        """Compute the hashes for all vectors using all hash functions."""
        # Reshape random_vectors to combine num_hashes and bands for batch matrix multiplication
        random_vectors_reshaped = self.random_vectors.view(-1, self.feature_size)
        # Perform batch matrix multiplication
        sign_vectors = torch.sign(torch.matmul(random_vectors_reshaped, vectors.T))
        # Reshape sign_vectors back to [num_hashes, num_vectors, bands]
        sign_vectors = sign_vectors.view(self.num_hashes, self.bands, -1).transpose(1, 2)

        # Expand hash_coefficients to match the dimensions of sign_vectors
        expanded_hash_coefficients = self.hash_coefficients.view(self.num_hashes, 1, self.bands).expand(-1, vectors.shape[0], -1)

        # Perform the multiplication operation and sum across bands
        hash_values = ((sign_vectors > 0) * expanded_hash_coefficients).sum(dim=2) % self.table_size

        # Populate the bucket_table
        for hash_f_ix in range(self.num_hashes):
            unique_hashes, inverse_indices, counts = torch.unique(hash_values[hash_f_ix], return_inverse=True, return_counts=True)
            for i, hash_val in enumerate(unique_hashes):
                indices = torch.where(inverse_indices == i)[0]
                if len(indices) > 1:
                    self.bucket_table[hash_f_ix].setdefault(hash_val.item(), set()).update(indices.tolist())


    def do_lsh(self, vectors):
        """Perform LSH on all vectors and return the collision matrix using vectorized operations."""
        num_vectors = vectors.shape[0]
        collision_matrix = torch.zeros((num_vectors, num_vectors), device=vectors.device)

        self._hash_to_table(vectors)

        # Construct collision_matrix
        for hash_index, bucket_dict in enumerate(self.bucket_table):
            for bucket_key, vector_indices_set in bucket_dict.items():
                vector_indices = torch.tensor(list(vector_indices_set), device=vectors.device)
                collision_matrix.index_put_((vector_indices[:, None], vector_indices), torch.tensor(1, dtype=collision_matrix.dtype, device=vectors.device))

        return collision_matrix[:num_vectors // 2, num_vectors // 2:] # only interested in Q vs K indices.
    
import time
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

    def __str__(self):
        return f"{self.interval:.10f} seconds"

torch.manual_seed(0)
lsh = LSH(8, 64, 4, 64)

torch.manual_seed(0)
lsh2 = LSH2(8, 64, 4, 64)

stacked = torch.rand((20, 64))

with Timer() as t:
    old_res = lsh.do_lsh(stacked)

print("Old time", t)

with Timer() as t:
    new_res = lsh2.do_lsh(stacked)
    
print("New time", t)

print("All close?", old_res.allclose(new_res))