from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='rsmine',
    version='0.1.1rc1',
    packages=find_packages(include=['rsmine', 'rsmine.*']),
    author='Doruk Efe Gökmen',
    author_email='dgoekmen@ethz.ch',
    description='Optimal coarse graining transformations with RSMI neural estimation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RSMI-NE/RSMI-NE',
    project_urls={
        'Source': 'https://github.com/RSMI-NE/RSMI-NE',
        'Bug Tracker': 'https://github.com/RSMI-NE/RSMI-NE/issues'
    },
    license='Apache License 2.0',
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
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',

        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='renormalization',
)
