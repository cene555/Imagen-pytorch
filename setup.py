from setuptools import setup

setup(
    name="imagen_pytorch",
    packages=[
        "imagen_pytorch",
        "imagen_pytorch.clip",
        "imagen_pytorch.tokenizer",
    ],
    package_data={
        "imagen_pytorch.tokenizer": [
            "bpe_simple_vocab_16e6.txt.gz",
            "encoder.json.gz",
            "vocab.bpe.gz",
        ],
        "imagen_pytorch.clip": ["config.yaml"],
    },
    install_requires=[
        "Pillow",
        "attrs",
        "torch",
        "filelock",
        "requests",
        "tqdm",
        "ftfy",
        "regex",
        "numpy",
        "blobfile",
        "accelerate",
	"transformers",
    ],
    author="cene655",
)
)
