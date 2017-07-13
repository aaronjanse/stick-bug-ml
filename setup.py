from setuptools import setup
setup(
    name='stickbugml',
    version='1.0',
    description='A framework to organize the process of designing supervised machine learning systems',
    keywords='stick bug, ml, machine learning, ai, artificial intelligence, framework, organization, organize',
    license="Apache 2.0",
    author='Aaron Janse',
    author_email='gitduino@gmail.com',
    url='https://github.com/aaronduino/stick-bug-ml',
    packages=['stickbugml'],  #same as name
    install_requires=['sklearn', 'pandas', 'numpy'], #external packages as dependencies
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
