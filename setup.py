from setuptools import setup, find_packages

VERSION = '0.0.0'
NAME = 'tner'
IS_RELEASED = False

with open('README.md') as f:
    readme = f.read()

setup(
    name=NAME,
    version=VERSION,
    description='A library for language model finetuning on named entity recognition and model evaluation over cross-domain datasets.',
    long_description=readme,
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
		'Pillow>=7.1.0',
		'mecab-python3==0.996.5',  # this version can only work
		'uvicorn==0.11.8',
		'jinja2==2.11.2',
		'aiofiles==0.5.0',
		'fastapi==0.61.0',
		'matplotlib==3.3.1',
		'toml',
		'tensorboard',
		'torch',
		'transformers',
		'seqeval',
		'segtok'
	]
)
