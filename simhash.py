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

    def _hash_to_table(self, vector, vector_ix):
        """Compute the hashes for a given vector using all hash functions."""
        for hash_f_ix in range(self.num_hashes):
            sign_vector = sign_hash_function(self.random_vectors[hash_f_ix], vector)
            bucket_ix = self._uniform_random_hash(sign_vector, hash_f_ix)
            bucket = self.bucket_table[hash_f_ix].setdefault(bucket_ix, set())
            bucket.add(vector_ix)

    def do_lsh(self, vectors):
        """Perform LSH on all vectors and return the collision matrix using vectorized operations."""
        num_vectors = vectors.shape[0]
        collision_matrix = torch.zeros((num_vectors, num_vectors))

        # Hash vectors to the bucket table
        for i in range(num_vectors):
            self._hash_to_table(vectors[i], i)
        
        # Construct collision_matrix
        for hash_index, bucket_dict in enumerate(self.bucket_table):
            for bucket_key, vector_indices_set in bucket_dict.items():
                vector_indices_list = list(vector_indices_set)

                # Loop through every unique pair of indices
                for i in range(len(vector_indices_list)):
                    for j in range(i + 1, len(vector_indices_list)):
                        i1 = vector_indices_list[i]
                        i2 = vector_indices_list[j]
                        
                        collision_matrix[i1, i2] = 1
                        collision_matrix[i2, i1] = 1

        return collision_matrix

# Example code to demonstrate usage
# Initialize LSH
# bands 8-16
# m 100 - 500
# nhashes 4-8
lsh = LSH(bands=8, table_size=10, num_hashes=4, feature_size=128)

# Example vectors
vectors = torch.randn(10,128)

# Perform LSH
collision_matrix = lsh.do_lsh(vectors)
print(collision_matrix)

