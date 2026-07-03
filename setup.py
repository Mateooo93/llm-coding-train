from setuptools import setup, find_packages

setup(
    name="attnres_lm",
    version="0.1.0",
    description="A revolutionary LLM with Attention Residuals (AttnRes) — replacing fixed residual connections with learned depth-wise attention",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "tiktoken>=0.5.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0",
        "numpy>=1.24.0",
    ],
)
