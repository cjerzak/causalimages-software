# Memory behavior

New models omit unused pooling/projection parameters and do not construct a
custom spatial transformer for frozen pretrained image embeddings. Temporal
transformers have their own stacked layers, rotary embeddings, and attention
pooling parameters. Concatenation/difference/variance aggregation does not
allocate a temporal transformer. Video frame inputs repeat each video's
covariates and split its random key to match the flattened frame batch.

Training uses one compiled loss/gradient/finite-check/AdaBelief update. Model
and optimizer buffers are owned by Python and donated to the next update;
fixed models and batches remain reusable. Non-finite updates leave model,
optimizer, and model state unchanged while adjusting the loss scale. Optimizer
slots are released before final inference. A failed donated update is terminal
and must not be retried with its consumed input state.

Attention and SwiGLU sublayers use activation checkpointing. This trades extra
backward computation for fewer saved activations. Toggle it when building a
model for a comparison with `options(causalimages.activation_checkpointing = FALSE)`.
It does not change the model's parameters or objective.

Frozen direct PyTorch encoders run under `no_grad()`. The pretrained cache holds
one configuration by default, evicting old entries before loading another.
`CleanupEnv = TRUE` clears both active bindings and cached entries. Advanced
users can increase `options(causalimages.pretrained_cache_size = 2L)` when the
memory cost of retaining multiple models is acceptable.

Input queues use these defaults:

```r
options(
  causalimages.input_buffer_bytes = 64 * 1024^2,
  causalimages.prefetch_batches = 1L,
  causalimages.input_threads = 2L
)
```

The byte budget caps each shuffle queue, using a representative record's size.
Fixed-shape images/videos are expected; a single record larger than the budget
still needs to fit in memory. This is a per-queue budget, not a process memory
limit, and a prefetched batch may be larger. The existing buffer scaler is
still an upper bound on record count. Reducing shuffle capacity changes the
randomized order (and order-based train/test assignments) relative to older
runs. Reinitialization of each fixed split remains deterministic.

Channel normalization uses chunked float64 reductions in Python and transfers
only channel statistics to R, retaining sample-SD and between-batch variance
semantics. Heterogeneity likelihood draws accumulate in a checkpointed scan;
deterministic image representations are computed once per batch. Stochastic
representations retain their per-draw keys and state order. Predictive means
and variances use streaming moments instead of retaining all Monte Carlo draws.
Cluster-weighted effects reduce over clusters separately for each observation,
and transport-only keys are excluded from the heterogeneity training pool.
Dataset filenames are absolute so background reads survive changes to R's
working directory; confounding representation initialization respects the
requested TFRecord precision.

Predictive artifacts now identify their model structure as version 2. Version 1
artifacts reconstruct the legacy parameter layout when loaded; new training
uses the corrected temporal architecture. Runtime memory controls are read
when a model or pipeline is constructed.

## Validation

Run the Python helper tests with a compatible JAX/Equinox/Optax/JMP backend:

```sh
JAX_PLATFORMS=cpu python -m pytest -q -p no:cacheprovider tests/python/test_runtime.py
```

R regressions in `tests/testthat/test-efficiency.R` cover cache eviction,
exception-safe no-grad scope, unused parameter removal, temporal gradients,
checkpointed output/gradient equivalence, pretrained image parameter allocation,
and legacy predictive artifact loading. Existing predictive and heterogeneity
transport tests exercise the shared training loop and inference, including one
and two clusters, multiple variational draws, and trainable representations.
A small confounding regression covers the same update path with float32 data.

Implementation validation used an isolated CPU environment with JAX 0.9.1,
Equinox 0.13.6, Optax 0.2.7, TensorFlow 2.21.0, and JMP 0.0.4. Heterogeneity
tests used `tfp-nightly==0.26.0.dev20260910`: released TFP 0.25.0 failed to import
with JAX 0.9.1. The backend installer and the user's Python environments were
not changed by this work.

In this R/Python 3.13 test environment, importing TFP after TensorFlow also
failed to find `distutils`. Importing `setuptools` before backend initialization
resolved that import error. The combined regression run skipped heterogeneity
for that reason; a separate run explicitly imported TFP first and passed both
heterogeneity cases without skips. Predictive, confounding, representation,
cache, seed, and backend regressions passed, as did all 15 Python helper tests
and a temporary package installation with an installed-runtime import check.

GPU peak memory and throughput must be measured on the intended hardware and
input shapes; CPU correctness tests do not establish those numbers.
