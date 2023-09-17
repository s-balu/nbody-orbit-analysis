from setuptools import setup

setup(
    name='orbitanalysis',
    version='0.1',
    description='Particle orbit analysis in N-body cosmological simulations',
    url='https://github.com/kriswalker/nbody-orbit-analysis',
    author='Kris Walker',
    author_email='kris.walker@icrar.org',
    package_dir={'orbitanalysis': './orbitanalysis/'},
    packages=['orbitanalysis'],
    install_requires=['numpy', 'h5py', 'pathos'],
    include_package_data=True,
    zip_safe=False,
)
