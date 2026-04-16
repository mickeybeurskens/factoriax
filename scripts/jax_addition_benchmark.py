"""Benchmark JAX addition: naive loop vs batched vmap vs scan."""

import time

import jax
import jax.numpy as jnp


@jax.jit
def add_step(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Add two arrays element-wise.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Element-wise sum of a and b.
    """
    return a + b


def benchmark_naive_loop(n_steps: int, size: int) -> float:
    """Run n_steps individual JIT'd additions and return steps/second.

    Args:
        n_steps: Number of addition steps.
        size: Length of each array.

    Returns:
        Steps per second.
    """
    a = jnp.ones(size)
    b = jnp.ones(size)

    # Warmup
    _ = add_step(a, b).block_until_ready()

    start = time.perf_counter()
    for _ in range(n_steps):
        add_step(a, b).block_until_ready()
    elapsed = time.perf_counter() - start
    return n_steps / elapsed


@jax.jit
def batched_add(a_batch: jnp.ndarray, b_batch: jnp.ndarray) -> jnp.ndarray:
    """Add two batches of arrays using vmap.

    Args:
        a_batch: Batch of first arrays, shape (batch, size).
        b_batch: Batch of second arrays, shape (batch, size).

    Returns:
        Batch of element-wise sums, shape (batch, size).
    """
    return jax.vmap(jnp.add)(a_batch, b_batch)


def benchmark_vmap_batch(n_steps: int, size: int) -> float:
    """Run n_steps additions as a single vmap batch and return steps/second.

    Args:
        n_steps: Number of addition steps (batch size).
        size: Length of each array.

    Returns:
        Steps per second.
    """
    a_batch = jnp.ones((n_steps, size))
    b_batch = jnp.ones((n_steps, size))

    # Warmup
    _ = batched_add(a_batch, b_batch).block_until_ready()

    start = time.perf_counter()
    batched_add(a_batch, b_batch).block_until_ready()
    elapsed = time.perf_counter() - start
    return n_steps / elapsed


@jax.jit
def scan_add(carry: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scan body: accumulate additions sequentially on-device.

    Args:
        carry: Running sum array.
        b: Array to add.

    Returns:
        Tuple of (updated carry, output for this step).
    """
    result = carry + b
    return result, result


def benchmark_scan(n_steps: int, size: int) -> float:
    """Run n_steps sequential additions via lax.scan and return steps/second.

    Args:
        n_steps: Number of addition steps.
        size: Length of each array.

    Returns:
        Steps per second.
    """

    @jax.jit
    def run_scan(init: jnp.ndarray, bs: jnp.ndarray) -> jnp.ndarray:
        """Run lax.scan over a batch of arrays.

        Args:
            init: Initial carry array.
            bs: Stacked arrays to add, shape (n_steps, size).

        Returns:
            Final accumulated sum.
        """
        final, _ = jax.lax.scan(scan_add, init, bs)
        return final

    init = jnp.zeros(size)
    bs = jnp.ones((n_steps, size))

    # Warmup
    _ = run_scan(init, bs).block_until_ready()

    start = time.perf_counter()
    run_scan(init, bs).block_until_ready()
    elapsed = time.perf_counter() - start
    return n_steps / elapsed


def main() -> None:
    """Run all three benchmarks and print results."""
    n_steps = 200
    size = 1024

    print(f"Benchmarking {n_steps} additions on arrays of size {size}")
    print(f"Device: {jax.devices()[0]}\n")

    naive_sps = benchmark_naive_loop(n_steps, size)
    print(f"Naive loop (individual JIT calls):  {naive_sps:>12,.0f} steps/s")

    vmap_sps = benchmark_vmap_batch(n_steps, size)
    print(f"Batched vmap (single kernel):        {vmap_sps:>12,.0f} steps/s")

    scan_sps = benchmark_scan(n_steps, size)
    print(f"lax.scan (fused sequential):         {scan_sps:>12,.0f} steps/s")

    print(f"\nvmap speedup over naive: {vmap_sps / naive_sps:>8.1f}x")
    print(f"scan speedup over naive: {scan_sps / naive_sps:>8.1f}x")


if __name__ == "__main__":
    main()
