from setuptools import setup, find_packages

setup(
    name='grs_tool',
    version='0.1.0',
    description='A toolkit for genetic risk scoring (GRS) feature selection and model optimization.',
    author='Jassica',
    author_email='2670548227@qq.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    python_requires='>=3.7',
)