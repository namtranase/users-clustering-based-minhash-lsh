"""
Program implement MinHash and MinhashLSH class.
@Author: namtran.ase@gmail.com.
"""
import random, copy, struct
import numpy as np

from datasketch.hashfunc import sha1_hash32

# Size of hash values i number of bytes
hashvalue_byte_size = len(bytes(np.int64(42).data))

# Max hash and range
_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)

class MinHash(object):
    def __init__(self,
                 num_perm=128,
                 seed=1,
                 hashfunc=sha1_hash32,
                 hashvalues=None,
                 permutations=None):
        if hashvalues is not None:
            num_perm = len(hashvalues)
        if num_perm > _hash_range:
            raise ValueError("Exceed value of range.")
        self.seed = seed
        if not callable(hashfunc):
            raise ValueError("Hash function isn't callable.")
        self.hashfunc = hashfunc
        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(num_perm)
        # Initialize permutation function parameters
        if permutations is not None:
            self.permutations = permutations
        else:
            generator = np.random.RandomState(self.seed)
            self.permutations = np.array([(generator.randint(1, _mersenne_prime, dtype=np.uint64),
                                           generator.randint(0, _mersenne_prime, dtype=np.uint64))
                                          for _ in range(num_perm)], dtype=np.uint64).T


    def _init_hashvalues(self, num_perm):
        return np.ones(num_perm, dtype=np.uint64)*_max_hash

    def _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)
    
    def update(self, b):
        print("hashvalues before update", self.hashvalues)
        hv = self.hashfunc(b)
        print("hv", hv)
        a, b = self.permutations
        print("permutation: ", (a,b))
        phv = np.bitwise_and((a * hv + b) % _mersenne_prime, np.uint64(_max_hash))
        print("phv", phv)
        self.hashvalues = np.minimum(phv, self.hashvalues)
        print("hashvalues after update", phv)
    
    def __len__(self):
        """Return the number of hasvalues.
        """
        return len(self.hashvalues)
