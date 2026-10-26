from fraud.features import InMemoryFeatureStore, Transaction, features_for
from fraud.redis_store import RedisFeatureStore


class FakePipeline:
    def __init__(self, r):
        self.r, self.ops = r, []

    def __getattr__(self, name):
        def op(*a, **kw):
            self.ops.append((name, a, kw))
            return self
        return op

    def execute(self):
        for name, a, kw in self.ops:
            getattr(self.r, name)(*a, **kw)


class FakeRedis:
    """Just enough of the redis-py API for RedisFeatureStore."""

    def __init__(self):
        self.z, self.h = {}, {}

    def pipeline(self):
        return FakePipeline(self)

    def zadd(self, key, mapping):
        self.z.setdefault(key, {}).update(mapping)

    def zrangebyscore(self, key, lo, hi):
        lo = float("-inf") if lo == "-inf" else float(lo)
        return [m for m, s in sorted(self.z.get(key, {}).items(), key=lambda kv: kv[1]) if s >= lo]

    def zremrangebyscore(self, key, lo, hi):
        self.z[key] = {m: s for m, s in self.z.get(key, {}).items() if s > hi}

    def expire(self, key, ttl):
        pass

    def hset(self, key, mapping):
        self.h[key] = {k: str(v) for k, v in mapping.items()}

    def hgetall(self, key):
        return dict(self.h.get(key, {}))


def test_redis_store_gives_same_features_as_memory_store():
    txns = [Transaction(f"t{i}", "u1", 10.0 * (i + 1), "DE" if i < 4 else "US", f"d{i % 2}", "books",
                        1000.0 + 300 * i) for i in range(6)]
    mem, red = InMemoryFeatureStore(), RedisFeatureStore(FakeRedis())
    for t in txns[:-1]:
        mem.record(t)
        red.record(t)
    assert features_for(txns[-1], mem) == features_for(txns[-1], red)
    assert red.profile("u1").known_devices == {"d0", "d1"}
