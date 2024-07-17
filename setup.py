from setuptools import setup, find_packages

setup(
    name='Intellishield',
    version='0.9',
    description='A Privacy Preserving Explainable Educational Recommendation System',
    author='Austin Nicolas',
    author_email='austineamonn@gmail.com',
    url='https://github.com/austineamonn/SummerResearch2024',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)

#pip install -e .