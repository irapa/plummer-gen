from setuptools import setup

setup(
    name="plummer-gen",
    version="1.0.0",
    description="Condições iniciais de esferas de Plummer para simulações de N-corpos",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Stephanie Moraes, Irapuan Rodrigues",
    license="MIT",
    packages=["plummer"],
    python_requires=">=3.6",
    install_requires=["numpy>=1.17", "scipy"],
    extras_require={"figures": ["matplotlib"], "test": ["pytest"]},
)
