import torch

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
        
    def reset(self):
        """Reset the LSH instance"""
        self.random_vectors = torch.randn(self.num_hashes, self.bands, self.feature_size)
        # Initialize random coefficients for each hash function
        self.hash_coefficients = torch.randint(0, self.table_size, (self.num_hashes, self.bands))
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
        # one sign_vector is shape [1, bands]
        # sign_vectors should have shape [num_hashes, vectors.shape[0], bands]
        # [4, 20, 8]
        hash_values = (sign_vectors > 0).unsqueeze(-1) * self.hash_coefficients.unsqueeze(1)
        hash_values = hash_values.sum(dim=2) % self.table_size
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
                collision_matrix.index_put_((vector_indices[:, None], vector_indices), torch.tensor(1, device=vectors.device))

        return collision_matrix

# dummy q and k have seq len 10, feature size 128, so for each head (2 heads) feature size is 64
# stacked matrix for each head would be 20 x 64
lsh = LSH(8, 64, 4, 64)
stacked = torch.ones((20, 64))
lsh.do_lsh(stacked)
