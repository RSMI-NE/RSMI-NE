import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name='rsmine',
    version='0.0.1',
    author='Doruk Efe Gökmen',
    author_email='dgoekmen@ethz.ch',
    description='Optimal coarse graining transformations with RSMI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RSMI-NE/RSMI-NE/rsmine',
    project_urls={
        "Bug Tracker": "https://github.com/RSMI-NE/RSMI-NE/issues"
    },
    license='Apache License 2.0',
    packages=['toolbox'],
    install_requires=['requests',
                      'numpy',
                      'tensorflow',
                      'tensorflow-probability',
                      'pandas',
                      'matplotlib',
                      'wandb',
                      'tqdm',
                      'networkx'],
)
