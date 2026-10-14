# pytest-numguard

pytest plugin for **numerical edge-case testing** of array code: it runs your
functions on NaN, ±inf, huge and subnormal magnitudes, cancellation-prone data,
empty arrays and float16/32/64 inputs, and compares them against references and
metamorphic properties.

```bash
pip install pytest-numguard
```

This package installs [numguard-ci](https://pypi.org/project/numguard-ci/),
which provides the plugin (the command and the import are `numguard`). Once it
is installed, pytest picks the plugin up automatically. The example needs SciPy
for the reference (`pip install scipy`):

```python
import numpy as np
import pytest
from scipy.special import logsumexp as reference


def naive_logsumexp(x):
    return np.log(np.sum(np.exp(x)))


def stable_logsumexp(x):
    x = np.asarray(x)
    if x.size == 0:
        return x.dtype.type(-np.inf)
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return (m + np.log(np.sum(np.exp(x - m)))).astype(x.dtype)


@pytest.mark.numguard(fn=naive_logsumexp, reference=reference)
def test_naive_logsumexp():
    pass


@pytest.mark.numguard(fn=stable_logsumexp, reference=reference, properties=["translation_equivariant"])
def test_stable_logsumexp():
    pass
```

Run `pytest`: the naive version fails on the `huge`, `large`, `cancellation` and
`wide_range` cases (it returns `inf` where the reference is finite), and the
stable version passes. Options such as `--numguard-backend torch` and the
`numguard.toml` configuration are documented at
https://github.com/bhaskargurram-ai/numguard.
