from setuptools import setup

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='stickbugml',
    version='1.0.4',
    description='A framework to organize the process of designing supervised machine learning systems',
    keywords='stick bug, ml, machine learning, ai, artificial intelligence, framework, organization, organize',
    license="Apache 2.0",
    author='Aaron Janse',
    author_email='gitduino@gmail.com',
    url='https://github.com/aaronduino/stick-bug-ml',
    packages=['stickbugml'],  #same as name
    install_requires=['sklearn', 'pandas', 'numpy'],
    long_description=long_description,
    python_requires='>=3',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
