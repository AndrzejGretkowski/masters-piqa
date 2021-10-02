from collections import OrderedDict


class CacheDict(OrderedDict):
    def __init__(self, *args, maxlen=0, **kwargs):
        self._maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._maxlen > 0:
            if len(self) > self._maxlen:
                self.popitem(False)