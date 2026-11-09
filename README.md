# supra-conceptmod

Supra2-IMG is a standalone model provider for [conceptmod](https://github.com/HyperGAN/conceptmod).
This repository owns the model's loading, adapters, encoding, sampling and
model-specific tests. Conceptmod core owns the DSL, training sessions, metrics
and complete edit-bundle format.

## Install and use

Install conceptmod 3 and this package in the same Python environment:

```bash
python -m pip install -e /path/to/conceptmod
python -m pip install -e .
conceptmod backends
cat > edit.json <<'JSON'
{"backend": {"device": "cuda:0", "lora_rank": 16}}
JSON
conceptmod train --backend supra --phrase "red=blue" --config edit.json --out outputs/edit
```

The entry point `conceptmod.backends:supra` exports lightweight metadata;
listing installed providers does not import inference libraries or load weights.
Python callers can select this provider explicitly:

```python
from conceptmod.providers import BackendRegistry
from supra_conceptmod.provider import PROVIDER

registry = BackendRegistry([PROVIDER], discover=False)
backend = registry.load("supra", device="cuda:0")
image = backend.generate("a cat on a windowsill", seed=42)
backend.save_trained("outputs/edit-bundle")
```

The default base-model identifier is `SupraLabs/Supra2-IMG`. Base weights are downloaded
only when constructing the real backend; no weights are shipped in this
package. Use the same model revision/local checkpoint when restoring an edit.
Diffusers is pinned to the supported `0.39.0` release; this package does not
install a floating Git checkout. The backend does not enable
`trust_remote_code`. Local tensor formats use safetensors; Supra's native
checkpoint reader uses `torch.load(..., weights_only=True)`.

## Model-dependent training defaults

The real Supra model uses 8 training sample steps / CFG 3;
the offline dummy uses 4 steps at the same CFG. These conditional values are returned by
`backend.training_defaults()` after construction. The lightweight provider
descriptor leaves its override list empty. When building a training
configuration directly, pass these values as `op_defaults`; caller-supplied
operator settings take precedence.

## Validation

```bash
python -m pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' python -m pytest -q
python -m build --wheel --no-isolation
```

CPU tests verify the provider contract and available local/stub behavior. They
do not establish pretrained-image quality or real-GPU training. The manual
`python scripts/smoke_supra.py` path can load large checkpoints and requires
CUDA; Supra additionally provides an offline tiny model_id="dummy" test path.
Run it deliberately on a machine with the required model access and memory.
No GPU test was run as part of this extraction.

## Origin

Extracted from `HyperGAN/conceptmod` at
`9a8a0c7f85c767a9222114d5d73706d85a2910f5`. Model computations and tensor mappings were retained; package
imports, provider metadata and test routing were moved to this product.
The original MIT license and attribution are retained in [LICENSE](LICENSE).
See [docs/origin.md](docs/origin.md) for the transferred-file inventory.
