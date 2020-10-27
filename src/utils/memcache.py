"""
Program implement MinHash and MinhashLSH class.
@Author: namtran.ase@gmail.com.
"""
from collections import OrderedDict

class LRUCache:
    """Implement simple least recently used caching algorithm.
    """
    def __init__(self, capacity: int):
        """Initialize.
        """
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """Returns value associated with key or None if key not exists.
        """
        if key not in self.cache:
            return None

        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """Put new (key, value) to cache.
        """
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
