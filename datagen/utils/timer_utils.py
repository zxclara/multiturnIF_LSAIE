import time
from functools import wraps

def measure_throughput(type="inference", num_instances=1000):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            res = func(*args, **kwargs)

            elapsed = time.perf_counter() - start
            throughput = num_instances / elapsed

            print(f"\n[{type} throughput] generated {num_instances} instances in {elapsed:.4f}s "
                  f": {throughput:.2f} inst/sec")

            return res
        return wrapper
    return decorator
