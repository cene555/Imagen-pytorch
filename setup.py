from setuptools import setup

setup(
    name="dalle2_decoder",
    packages=[
        "dalle2_decoder",
        "dalle2_decoder.clip",
        "dalle2_decoder.tokenizer",
    ],
    package_data={
        "dalle2_decoder.tokenizer": [
            "bpe_simple_vocab_16e6.txt.gz",
            "encoder.json.gz",
            "vocab.bpe.gz",
        ],
        "dalle2_decoder.clip": ["config.yaml"],
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
        "mpi4py",
	"transformers"
    ],
    author="",
)
