import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inst" / "python"))

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest

from causalimages_runtime import (
    TrainingStep, bounded_prefetch, bounded_shuffle, channel_moments, element_bytes,
    expected_draws, flatten_video_inputs, fold_keys, restrict_tfrecord_keys, streaming_moments,
)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_fused_update_matches_reference_and_does_not_consume_fixed_data(dtype):
    policy = jmp.Policy(param_dtype="float32", compute_dtype=dtype, output_dtype="float32")
    optimizer = optax.chain(optax.adaptive_grad_clip(.25), optax.adabelief(.001))
    model = {"w": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 30}
    template = jax.tree.map(np.asarray, model)
    fixed = jnp.ones(4)
    x = jnp.ones((2, 3))
    y = jnp.zeros((2, 4))
    scale = jmp.DynamicLossScale(jnp.float32(8))
    state = jnp.float32(0)

    def loss(model, fixed, images, x, treatment, y, keys, state, mp, inference):
        p, scale = mp
        pred = images @ p.cast_to_compute(model)["w"] + fixed.astype(p.compute_dtype)
        value = jnp.mean((pred.astype(jnp.float32) - y) ** 2)
        return (scale.scale(value) if dtype == "float16" else value), state + 1

    step = TrainingStep(loss, optimizer, loss_scaling=dtype == "float16")
    owned = step.own(model, optimizer.init(model), state, scale)
    reference = step.own(model, optimizer.init(model), state, scale)
    for _ in range(3):
        old, opt_state, state, scale = reference
        (_, next_state), grad = eqx.filter_value_and_grad(loss, has_aux=True)(
            old, fixed, x.astype(policy.compute_dtype), x, y, y, jax.random.split(jax.random.key(1), 2), state, (policy, scale), False
        )
        if dtype == "float16":
            grad = scale.unscale(grad)
        updates, opt_state = optimizer.update(grad, opt_state, old)
        reference = (eqx.apply_updates(old, updates), opt_state, next_state, scale.adjust(jnp.bool_(True)))
        owned, metrics = step.update(owned, fixed, x.astype(policy.compute_dtype), x, y, y, jax.random.split(jax.random.key(1), 2), policy)
        np.testing.assert_allclose(owned[0]["w"], reference[0]["w"], atol=2e-6)
        assert metrics[3] == 1
    np.testing.assert_array_equal(model["w"], template["w"])
    np.testing.assert_array_equal(fixed, np.ones(4))
    assert owned[2] == 3


def test_nonfinite_update_preserves_parameters_optimizer_state_and_state():
    policy = jmp.Policy(param_dtype="float32", compute_dtype="float32", output_dtype="float32")
    optimizer = optax.adabelief(.01)
    model = {"w": jnp.ones(3)}

    def loss(model, fixed, images, x, w, y, keys, state, mp, inference):
        return mp[1].scale(jnp.sum(model["w"] * images)), state + 1

    step = TrainingStep(loss, optimizer)
    owned = step.own(model, optimizer.init(model), jnp.float32(2), jmp.DynamicLossScale(jnp.float32(8)))
    before = jax.tree.map(lambda x: np.array(x, copy=True), owned[:3])
    owned, metrics = step.update(owned, jnp.float32(0), jnp.full(3, jnp.nan), None, None, None, None, policy)
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(owned[:3])):
        np.testing.assert_array_equal(old, new)
    assert metrics[3] == 0
    assert owned[3].loss_scale < 8


@pytest.mark.parametrize("legacy", [False, True])
def test_video_alignment_and_independent_frame_keys(legacy):
    key = jax.random.PRNGKey(0) if legacy else jax.random.key(0)
    images = jnp.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4, 1)
    x = jnp.array([[10, 20], [30, 40]])
    keys = jax.random.split(key, 2)
    flat, flat_x, flat_keys = flatten_video_inputs(images, x, keys)
    np.testing.assert_array_equal(flat.reshape(images.shape), images)
    np.testing.assert_array_equal(flat_x, np.repeat(x, 3, axis=0))
    assert len(np.unique(jax.random.key_data(flat_keys), axis=0)) == 6
    assert jax.random.key_data(fold_keys(keys, 3)).shape == (2, 2)


@pytest.mark.parametrize("count", [1, 7, 32])
def test_streaming_moments_match_materialized_samples(count):
    def draw(i):
        return {"x": jax.random.normal(jax.random.fold_in(jax.random.key(20), i), (4, 3)), "y": jnp.array([i * 1.0])}
    mean, var = jax.jit(lambda: streaming_moments(draw, count))()
    draws = [draw(i) for i in range(1, count + 1)]
    for name in mean:
        samples = np.stack([d[name] for d in draws])
        np.testing.assert_allclose(mean[name], samples.mean(0), atol=2e-6)
        np.testing.assert_allclose(var[name], samples.var(0), atol=2e-6)


def test_checkpointed_draws_preserve_gradients_and_state_order():
    def evaluate(weight, streamed):
        def draw(state, i):
            return jnp.sum(jnp.sin(weight * i + state)), state + .2
        if streamed:
            return expected_draws(draw, 5, jnp.float32(0))
        total, state = jnp.float32(0), jnp.float32(0)
        for i in range(1, 6):
            value, state = draw(state, i)
            total += value / 5
        return total, state
    weight = jnp.linspace(0, 1, 16)
    for fn in [lambda stream: evaluate(weight, stream),
               lambda stream: jax.grad(lambda w: evaluate(w, stream)[0])(weight)]:
        np.testing.assert_allclose(fn(True), fn(False), atol=2e-6)


@pytest.mark.parametrize("shape", [(3, 7, 9, 2), (2, 3, 7, 9, 4)])
def test_channel_moments_match_r_style_double_reductions(shape):
    x = np.random.default_rng(1).normal(size=shape).astype(np.float16)
    mean, sd = channel_moments(x, chunk_bytes=256)
    flat = x.astype(np.float64).reshape(-1, shape[-1])
    np.testing.assert_allclose(mean, flat.mean(0), atol=1e-12)
    np.testing.assert_allclose(sd, flat.std(0, ddof=1), atol=1e-12)


def test_shuffle_caps_serialized_and_decoded_records_by_bytes():
    class Dataset:
        def __init__(self, sample):
            self.sample = sample
        def take(self, n):
            return self
        def as_numpy_iterator(self):
            return iter([self.sample])
        def shuffle(self, capacity, **kwargs):
            return capacity
    assert bounded_shuffle(Dataset(b"x" * 24), 640, 64, False) == 2
    assert bounded_shuffle(Dataset((np.zeros((16, 4), np.float16), b"key")), 64, 256, True) == 1


def test_element_sizes_handle_string_scalars_and_empty_data():
    assert element_bytes(None) == 0
    assert element_bytes(np.array(["ab", "é"])) == 4
    assert element_bytes(np.array([b"ab", b"c"], dtype=object)) == 3


def test_training_key_filter_excludes_transport_records_before_decoding():
    tf = pytest.importorskip("tensorflow")
    records = []
    for key in ["train-a", "transport", "train-b"]:
        serialized_key = tf.io.serialize_tensor(tf.constant(key)).numpy()
        records.append(tf.train.Example(features=tf.train.Features(feature={
            "key": tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_key]))
        })).SerializeToString())
    dataset = tf.data.Dataset.from_tensor_slices(records)
    filtered = restrict_tfrecord_keys(dataset, ["train-a", "train-b"])
    assert list(bounded_prefetch(filtered).as_numpy_iterator()) == [records[0], records[2]]
