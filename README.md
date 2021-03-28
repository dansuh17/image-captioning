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
black ./
```

**Prerequisite**:

- [black](https://github.com/psf/black): install `black` with:

```
pip install black
```

`black` reads `pyproject.toml` configuration for autoformatting options.
