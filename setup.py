from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
version = '0.1.0'
setup(
    name='tner',
    packages=find_packages(exclude=["asset", "examples", "static", "templates", "tests"]),
    version=version,
    license='MIT',
    description='Transformer-based named entity recognition',
    url='https://github.com/asahi417/tner',
    download_url="https://github.com/asahi417/tner/archive/{}.tar.gz".format(version),
    keywords=['ner', 'nlp', 'language-model'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        'sudachipy',
        'sudachidict_core',
        # 'uvicorn==0.11.8',
        # 'jinja2==2.11.3',
        # 'aiofiles==0.5.0',
        # 'fastapi==0.65.2',
        # 'matplotlib==3.3.1',
        'uvicorn',
        'jinja2',
        'aiofiles',
        'fastapi',
        'matplotlib',
        'toml',
        'pandas',
        'torch==1.10.0',
        'transformers==4.12.5',
        'sentencepiece==0.1.96',
        'seqeval==1.2.2',
        'segtok==1.5.10',
        'allennlp==2.8.0'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'tner-train = tner_cl.train:main_train',
            'tner-train-search = tner_cl.train:main_train_search'
        ],
    }
)
