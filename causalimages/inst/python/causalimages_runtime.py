"""Small, bounded-memory helpers shared by the R training entry points."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class DynamicScale(eqx.Module):
    """JMP-compatible scale whose pytree supports Equinox's None filtering."""

    loss_scale: jax.Array
    counter: jax.Array
    min_loss_scale: jax.Array
    period: int = eqx.field(static=True)
    factor: int = eqx.field(static=True)

    def scale(self, tree):
        return jax.tree.map(lambda x: x * self.loss_scale, tree)

    def unscale(self, tree):
        return jax.tree.map(lambda x: x / self.loss_scale, tree)

    def adjust(self, finite):
        grown = self.loss_scale * self.factor
        grown = jnp.where(jnp.isfinite(grown), grown, self.loss_scale)
        value = jnp.where(finite,
                          jnp.where(self.counter == self.period - 1, grown, self.loss_scale),
                          jnp.maximum(self.min_loss_scale, self.loss_scale / self.factor))
        counter = ((self.counter + 1) % self.period) * finite
        return DynamicScale(value, counter, self.min_loss_scale, self.period, self.factor)


class OwnedState:
    """Opaque across reticulate, which otherwise turns optimizer tuples into lists."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return self.value[index]

    @property
    def model(self):
        return self.value[0]

    @property
    def model_state(self):
        return self.value[2]

    @property
    def loss_scale(self):
        return self.value[3]


class TrainingStep:
    """Keep differentiation, finite checks, and optimizer updates in one JIT.

    Only the owned training state is donated. Fixed models, batches, and the
    precision policy remain reusable. A rejected update preserves both the
    optimizer and model state; its loss scale can still decrease.
    """

    def __init__(self, loss_fn, optimizer, loss_scaling=True, donate=True):
        import jmp

        self._optimizer = optimizer
        value_and_grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)

        def step(readonly, owned):
            fixed, images, x, treatment, y, keys, policy = readonly
            model, opt_state, model_state, loss_scale = owned
            (loss, next_state), grads = value_and_grad(
                model, fixed, images, x, treatment, y, keys, model_state,
                (policy, loss_scale), False,
            )
            if loss_scaling:
                loss = loss_scale.unscale(loss)
                grads = loss_scale.unscale(grads)
            grads = policy.cast_to_param(grads)
            next_state = policy.cast_to_param(next_state)
            leaves = [g for g in jax.tree.leaves(grads) if eqx.is_array(g) and g.size]
            norms = jnp.stack([jnp.mean(jnp.abs(g)) for g in leaves]) if leaves else jnp.zeros(1)
            norm = jnp.mean(norms)
            finite = jmp.all_finite(grads) & jnp.isfinite(loss)
            accepted = finite & (norm > 1e-10)
            params, static = eqx.partition(model, eqx.is_inexact_array)

            def update(_):
                updates, state_out = optimizer.update(grads, opt_state, params)
                return eqx.apply_updates(params, updates), state_out, next_state

            params, opt_state, model_state = jax.lax.cond(
                accepted, update, lambda _: (params, opt_state, model_state), None,
            )
            if loss_scaling:
                loss_scale = loss_scale.adjust(finite)
            model = eqx.combine(params, static)
            metrics = jnp.stack((loss, norm, jnp.mean(norms == 0), accepted.astype(jnp.float32)))
            return (model, opt_state, model_state, loss_scale), metrics

        self._step = eqx.filter_jit(step, donate="all-except-first" if donate else "none")

    @staticmethod
    def own(model, opt_state, model_state, loss_scale):
        # Initializer trees and R closures may share leaves, including optimizer
        # zero arrays. Donation needs independently owned storage for each leaf.
        loss_scale = DynamicScale(
            jnp.asarray(loss_scale.loss_scale), jnp.asarray(loss_scale.counter),
            jnp.asarray(loss_scale.min_loss_scale), int(loss_scale.period), int(loss_scale.factor),
        )
        return OwnedState(jax.tree.map(
            lambda x: jnp.array(x, copy=True) if eqx.is_array(x) else x,
            (model, opt_state, model_state, loss_scale),
        ))

    def init(self, model, model_state, loss_scale):
        params = eqx.filter(model, eqx.is_inexact_array)
        return self.own(model, self._optimizer.init(params), model_state, loss_scale)

    def update(self, owned, fixed, images, x, treatment, y, keys, policy):
        try:
            state, metrics = self._step((fixed, images, x, treatment, y, keys, policy), owned.value)
            # One small device-to-host transfer also surfaces asynchronous errors
            # before the caller attempts another step with consumed input buffers.
            return OwnedState(state), np.asarray(metrics)
        except Exception as exc:
            raise RuntimeError(f"Training update failed; its donated input state cannot be retried: {exc}") from exc


def flatten_video_inputs(images, x, keys):
    batch, frames = images.shape[:2]
    frame_keys = jax.vmap(lambda k: jax.random.split(k, frames))(keys)
    return (
        images.reshape((batch * frames,) + images.shape[2:]),
        jnp.repeat(x, frames, axis=0),
        frame_keys.reshape((batch * frames,) + keys.shape[1:]),
    )


def fold_keys(keys, index):
    """Fold an index into either a scalar key or a batch of typed/legacy keys."""
    if jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key):
        return jax.random.fold_in(keys, index) if keys.ndim == 0 else jax.vmap(lambda k: jax.random.fold_in(k, index))(keys)
    return jax.random.fold_in(keys, index) if keys.ndim == 1 else jax.vmap(lambda k: jax.random.fold_in(k, index))(keys)


def streaming_moments(draw, count):
    """Population mean/variance of a pytree, without retaining all draws."""
    count = int(count)
    if count < 1:
        raise ValueError("Monte Carlo count must be positive")
    first = draw(jnp.int32(1))
    first = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), first)
    zeros = jax.tree.map(jnp.zeros_like, first)

    def body(i, carry):
        means, m2 = carry
        sample = draw(i + 1)
        delta = jax.tree.map(lambda x, m: x.astype(jnp.float32) - m, sample, means)
        new_means = jax.tree.map(lambda m, d: m + d / (i + 1), means, delta)
        m2 = jax.tree.map(lambda acc, d, x, m: acc + d * (x - m), m2, delta, sample, new_means)
        return new_means, m2

    means, m2 = jax.lax.fori_loop(1, count, body, (first, zeros))
    return means, jax.tree.map(lambda x: jnp.maximum(x / count, 0), m2)


def expected_draws(draw, count, state):
    """Accumulate draws with rematerialization and sequential state semantics."""
    count = int(count)
    if count < 1:
        raise ValueError("Monte Carlo count must be positive")

    @eqx.filter_checkpoint
    def body(carry, i):
        total, state_in = carry
        value, state_out = draw(state_in, i)
        return (total + value.astype(jnp.float32) / count, state_out), None

    return jax.lax.scan(body, (jnp.float32(0), state), jnp.arange(1, count + 1))[0]


def channel_moments(tensor, chunk_bytes=8 * 1024**2):
    """Return R-compatible channel means/sample SDs using bounded scratch.

    NumPy data stays in Python. Only two channel vectors cross reticulate.
    Float64 blockwise accumulation retains the old R reduction precision.
    """
    values = np.asarray(tensor).reshape((-1, int(tensor.shape[-1])))
    channels = values.shape[1]
    block_rows = max(1, int(chunk_bytes) // (8 * channels))
    mean = np.zeros(channels, dtype=np.float64)
    m2 = np.zeros(channels, dtype=np.float64)
    n = 0
    for start in range(0, len(values), block_rows):
        block = np.array(values[start:start + block_rows], dtype=np.float64, copy=True)
        block_mean = block.mean(axis=0)
        block -= block_mean
        block_m2 = np.einsum("ij,ij->j", block, block)
        size = len(block)
        delta = block_mean - mean
        m2 += block_m2 + delta**2 * (n * size / (n + size))
        mean += delta * (size / (n + size))
        n += size
    sd = np.sqrt(m2 / (n - 1)) if n > 1 else np.full(channels, np.nan)
    return mean, sd


def element_bytes(value):
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, dict):
        return sum(element_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(element_bytes(v) for v in value)
    value = np.asarray(value)
    if value.dtype.kind in "OSU":
        return sum(element_bytes(v) for v in value.flat)
    return value.nbytes


def bounded_shuffle(dataset, requested, max_bytes, reshuffle_each_iteration):
    sample = next(iter(dataset.take(1).as_numpy_iterator()), None)
    size = element_bytes(sample) if sample is not None else 1
    capacity = max(1, min(int(requested), int(max_bytes) // max(1, size)))
    return dataset.shuffle(capacity, reshuffle_each_iteration=bool(reshuffle_each_iteration))


def bounded_prefetch(dataset, batches=1, threads=2):
    import tensorflow as tf

    options = tf.data.Options()
    options.autotune.enabled = False
    options.experimental_optimization.inject_prefetch = False
    options.threading.private_threadpool_size = int(threads)
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).prefetch(int(batches))


def restrict_tfrecord_keys(dataset, keys):
    """Select the training pool using key metadata, before decoding images."""
    import tensorflow as tf

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(list(keys), dtype=tf.string),
                                           tf.ones(len(keys), dtype=tf.int32)), 0,
    )

    def selected(record):
        content = tf.io.parse_single_example(record, {"key": tf.io.FixedLenFeature([], tf.string)})
        key = tf.reshape(tf.io.parse_tensor(content["key"], out_type=tf.string), ())
        return table.lookup(key) == 1

    return dataset.filter(selected)
