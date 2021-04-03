# image-captioning

# References
- [Vaswani et al. - Attention is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [Tensorflow Tutorial - Image Captioning with Visual Attention](https://www.tensorflow.org/tutorials/text/image_captioning)
- [Google AI blog - Transformers for Image Recognition at Scale](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)

# Linting

Check linting with:

```
pylint --rcfile=.pylintrc ./<directory>
```

**Prerequisite**:

- [pylint](https://pypi.org/project/pylint/): install `pylint` with:

```
pip install pylint
```

# Auto-formatting

```
yapf -i ./image_captioning/**/*.py
```

Warning: The option `-i` will modify the files in-place.

**Prerequisite**:

- [yapf](https://github.com/google/yapf): install `yapf` with:

```
pip install yapf
```

`yapf` reads `pyproject.toml` configuration for autoformatting options.
