from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.12.0',
    'numpy',
    'matplotlib',
]


setup(
    name='meta_attack',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[
        p for p in find_packages() if p.startswith('meta_attack')
    ],
    description='Meta Adversarial Attack'
)