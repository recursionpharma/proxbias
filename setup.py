import roadie
from setuptools import find_packages, setup

setup(
    packages=find_packages(),
    version=roadie.infer_version("proxbias"),
    include_package_data=True,
)
