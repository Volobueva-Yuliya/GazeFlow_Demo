from setuptools import find_packages, setup

setup(
    name='dcgazeflow_demo',
    version='0.1dev',
    packages=find_packages(),
    license='',
    python_requires=">=3.8.*",
    install_requires=[
    	'tensorflow-gpu==2.4.0',
        'tensorflow-probability==0.12.2',
        'numpy',
        'opencv-python',
        'pillow',
        'tqdm',
    ],
    include_package_data=True,
    package_data={'': ['*, yaml', '*.cu', '*.cpp', '*,h']},
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)
