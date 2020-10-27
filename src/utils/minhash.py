"""
Program implement MinHash and MinhashLSH class.
@Author: namtran.ase@gmail.com.
"""
import random, copy, struct
import warnings
import numpy as np
import hashlib
import string
from datasketch.storage import (
    ordered_storage, unordered_storage, _random_name)

RANDOM_SEED = 2020
NUM_PERMS = 128
LSH_THRESHOLD = 0.95

_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)


def sha1_hash32(data):
    """A 32-bit hash function based on SHA1.
    """
    return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]

def _random_name(length):
    # For use with Redis, we return bytes
    return ''.join(random.choice(string.ascii_lowercase)
                   for _ in range(length)).encode('utf8')

class MinHash(object):
    def __init__(self,
                 num_perm=NUM_PERMS,
                 seed=RANDOM_SEED,
                 hashfunc=sha1_hash32,
                 hashvalues=None,
                 permutations=None,
                 cache=None):
        if hashvalues is not None:
            num_perm = len(hashvalues)
        self.seed = seed
        self.hashfunc = hashfunc
        self.cache = cache
        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(num_perm)
        # Initialize permutations parameters
        if permutations is not None:
            self.permutations = permutations
        else:
            generator = np.random.RandomState(self.seed)
            self.permutations = np.array(
                [(generator.randint(1, _mersenne_prime, dtype=np.uint64),
                  generator.randint(0, _mersenne_prime, dtype=np.uint64))
                 for _ in range(num_perm)], dtype=np.uint64).T
    def _init_hashvalues(self, num_perm):
        return np.ones(num_perm, dtype=np.uint64)*_max_hash

    def _parse_hashvalues(self, hashvalues):
        return np.array(hashvalues, dtype=np.uint64)

    def update(self, input_bytes):
        phv = self.cache.get(input_bytes)
        if phv is None:
            hv = self.hashfunc(input_bytes)
            a, b = self.permutations
            # using XOR bitwise instead of using new hash function
            phv = np.bitwise_and((a*hv + b) % _mersenne_prime, np.uint64(_max_hash))
            self.cache.put(input_bytes, phv)
        self.hashvalues = np.minimum(phv, self.hashvalues)

    def __len__(self):
        """The number of hash values.
        """
        return len(self.hashvalues)

class MinHashLSH(object):
    def __init__(self, threshold=0.9, num_perm=128, params=None):
        self._buffer_size = 50000
        storage_config = {'type': 'dict'}

        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")

        # Band and rows.
        self.h = num_perm
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError("Band and Row must less than num_perm")

        # Hashtables
        self.hashtables = [unordered_storage(storage_config) for i in range(self.b)]
        self.hashranges = [(i*self.r, (i+1)*self.r) for i in range(self.b)]
        self.keys = ordered_storage(storage_config)

    def insert(self, key, minhash, check_duplication=True):
        """
        Insert a key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.
        """
        self._insert(key, minhash, check_duplication=check_duplication, buffer=False)

    def insertion_session(self, buffer_size=50000):
        """
        Create a context manager for fast insertion into this index.
        """
        return MinHashLSHInsertionSession(self, buffer_size=buffer_size)

    def _insert(self, key, minhash, check_duplication=True, buffer=False):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d"
                              % (self.h, len(minhash)))

        if check_duplication and key in self.keys:
            raise ValueError("The given key already exists")
        Hs = [self._H(minhash.hashvalues[start:end])
              for start, end in self.hashranges]
        self.keys.insert(key, *Hs, buffer=buffer)
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key, buffer=buffer)

    def get_bucket_values(self):
        """Get all buckets.
        """
        buckets = list()
        for hash_table in self.hashtables:
            keys = list(hash_table.keys())
            values = hash_table.getmany(*keys)
            for value in values:
                if value not in buckets:
                    buckets.append(value)

        return buckets

    @staticmethod
    def _H(hs):
        return bytes(hs.byteswap().data)

class MinHashLSHInsertionSession:
    '''Context manager for batch insertion of documents into a MinHashLSH.
    '''

    def __init__(self, lsh, buffer_size):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def insert(self, key, minhash, check_duplication=True):
        """
        Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.
        """
        self.lsh._insert(key, minhash, check_duplication=check_duplication,
                         buffer=True)
