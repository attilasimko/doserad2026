# Vendored PyDoseRT

This directory is a verbatim copy of `src/pydosert` from

    https://github.com/UMU-DDI/PyDoseRT
    branch  doserad
    commit  d74b4038acdc5e68ec6b49bd7d58c2dfc567f332   ("Fluence volume speedup", 2026-08-24)

PyDoseRT is MIT licensed; the licence is retained in `LICENSE` alongside this
file and covers everything in this directory.

## Why it is vendored rather than pip-installed

The corrector is trained to predict the residual of ONE baseline dose. The
engine calibration it ships against — the TERMA constants, the multislab
residual calibration, both frozen priors — was fitted against this exact
PyDoseRT. Installing from a branch means a rebuild can silently change the
physics underneath a corrector trained on the old one, with no error anywhere:
the shapes all match, the dose is just wrong. A mismatched pair of this kind was
measured at plan gamma 93.3% -> 63.0%.

Pinning a commit in the Dockerfile fixed the container but not a local
checkout, and it still depended on the upstream repository staying reachable
and the commit staying present. Vendoring makes the physics part of this
repository's own history.

## Updating it

Do not edit these files in place — changes belong upstream. To move to a newer
PyDoseRT, replace the directory wholesale and re-run the full validation:

    git -C <PyDoseRT> archive <commit> src/pydosert | tar -x --strip-components=1 -C .

then update the commit recorded above, and re-check the engine against the
reference before trusting any downstream number.
