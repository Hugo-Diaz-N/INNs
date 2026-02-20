# InvNNsRNLA

This repository contains our tests applied to invertible neural networks (INNs). We are exploring ideas using the implementation from the [Invertible ResNet GitHub repository](https://github.com/jhjacobsen/invertible-resnet) by Jacobsen et al.


## Reference Paper

We are working with the model introduced in:

> **Behrmann et al.**, "Invertible Residual Networks," *ICML 2019*.  
> [Proceedings version](https://proceedings.mlr.press/v97/behrmann19a.html)

This paper introduces a way to make ResNets invertible by constraining the Lipschitz constant of residual blocks.

## Installation Notes

The original implementation in the [invertible-resnet](https://github.com/jhjacobsen/invertible-resnet) repo can be tricky to install. Issues:

- Use `environment.yml` when creating the environment to avoid version mismatches.

### ✅ Create Environment from YAML File

Run:

```bash
conda env create -f environment.yml
```

[Source](https://gist.github.com/atifraza/b1a92ae7c549dd011590209f188ed2a0#creating-an-environment-file) 

## ⚠️ Things That Don't Work Anymore (and How to Fix Them)

```python
# Deprecated: `zero_gradients` from `torch.autograd.gradcheck`
# Fix: Manually define the function to zero gradients.
def zero_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()
```
```python
# Deprecated: `np.float(j)`
# Fix: Use built-in `float(j)` instead.
value = float(j)
```
```python
# ImportError: Relative import in `conv_iResNet.py`
# Fix: Replace
# from .model_utils import injective_pad, ActNorm2D, Split
# with
from models.model_utils import injective_pad, ActNorm2D, Split
```
```python
# Logging issue: `sigma_log.copy_(sigma.detach())` may break with functorch transforms
# Fix: Use try-except to skip inside transforms
try:
    with torch.no_grad():
        sigma_log.copy_(sigma.detach())
except RuntimeError:
    pass  # skip logging update inside transforms
```



## Related Work

Here are some other useful references related to invertible neural networks and their applications:

- **Ardizzone et al.**, *Analyzing Inverse Problems with Invertible Neural Networks*, 2018.  
  [arXiv:1808.04730](https://arxiv.org/pdf/1808.04730)

- **Köhler et al.**, *Equivariant Flows: Exact Likelihood for Generative Models*, 2021.  
  [arXiv:2006.02425](https://arxiv.org/abs/2006.02425)

- **Chen et al.**, *Neural ODEs*, 2018.  
  [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

## Notes

This is a work-in-progress and mainly for learning and experimentation.
