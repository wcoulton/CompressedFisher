from setuptools import setup

setup(
    name='CompressedFisher',
    version='0.1.0',    
    description='A Python package implementing the Compressed Fisher method of Coulton & Wandelt',
    url='https://github.com/wcoulton/CompressedFisher',
    author='Will Coulton',
    license='BSD 2-clause',
    packages=['CompressedFisher','CompressedFisher.distributions'],
    install_requires=[
                      'numpy',
                      'scipy'                  
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)