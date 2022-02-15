from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='rsmine',
    version='0.1.1',
    packages=find_packages(include=['rsmine', 'rsmine.*']),
    author='Doruk Efe Gökmen',
    author_email='dgoekmen@ethz.ch',
    description='Optimal coarse graining transformations with RSMI neural estimation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RSMI-NE/RSMI-NE',
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
                      'networkx',
                      'scipy'],
)
