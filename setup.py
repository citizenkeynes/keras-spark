from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
   name='keras_spark',
   version='0.99',
   long_description=long_description,
   long_description_content_type='text/markdown',
   description='seamless support for spark datasets in keras .fit() and .predict()',
   author='christian sommeregger',
   author_email='csommeregger@gmail.com',
   packages=["keras_spark"],
   install_requires=["pandas==2.1.1","pyarrow==7.0.0",'tensorflow<=2.12.0','petastorm','s3fs']
)