from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
version = '0.0.7'
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
        'Pillow>=7.1.0',
        'sudachipy',
        'sudachidict_core',
        'uvicorn==0.11.8',
        'jinja2==2.11.2',
        'aiofiles==0.5.0',
        'fastapi==0.61.0',
        'matplotlib==3.3.1',
        'toml',
        'tensorboard',
        'torch',
        'transformers',
        'sentencepiece',
        'seqeval',
        'segtok'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'tner-train = tner_cl.train:main',
            'tner-test = tner_cl.test:main',
            'tner-predict = tner_cl.predict:main'
        ],
    }
)
