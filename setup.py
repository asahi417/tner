from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
version = '0.1.8'
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
    extras_require={
        "app": [
            'uvicorn==0.11.8',
            'jinja2==2.11.3',
            'aiofiles==0.5.0',
            'fastapi==0.65.2',
            'matplotlib==3.3.1',
            'Pillow>=7.1.0',
        ],
        "japanese": [
            'sudachipy',
            'sudachidict_core',
        ]
    },
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        'torch',
        'allennlp>=2.0.0',
        'transformers',
        'sentencepiece',
        'seqeval',
        'datasets'
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'tner-train = tner.tner_cl.train:main_trainer',
            'tner-train-search = tner.tner_cl.train:main_trainer_with_search',
            'tner-evaluate = tner.tner_cl.evaluate:main',
            'tner-predict = tner.tner_cl.predict:main',
            'tner-push-to-hub = tner.tner_cl.push_to_hub:main'
        ],
    }
)
