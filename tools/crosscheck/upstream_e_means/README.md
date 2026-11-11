# Vendored upstream e-means (development only)

These four files are faiss `contrib/e_means/{__init__.py, base.py, e_means.py,
e_means_spherical.py}` at the pinned commit
`2b648f0605f2ef3d14c4705350f8a0b214463878`, downloaded unmodified from
`https://raw.githubusercontent.com/facebookresearch/faiss/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means/<file>`.
They keep their MIT headers (copyright Meta Platforms, Inc. and affiliates);
the licence text is in the package's `inst/COPYRIGHTS`.

They are vendored because the PyPI `faiss-cpu` 1.16.0 wheel ships a different
revision of `e_means.py` (it lacks `_add_rows_at`). `faiss-cpu` still supplies
`faiss.knn`. Only `crosscheck.py` imports them.

SHA-256:

```
ad196786de8eb8ab7a229908b86ad8548a72459ade95e9e9c3275943dd50bb78  __init__.py
bcffd7829ebfc25e1354308770b49e6f99d8566ba5e474a41a22e9ec414b17cd  base.py
b1710504fdbef21efdd33d8489a56f8e7fc6056182afbed3f355985302c2e77c  e_means.py
d4e4177593f70e284bd2f6ad8812170e2a830ea62c697c514edc6298a00ea7a6  e_means_spherical.py
```
