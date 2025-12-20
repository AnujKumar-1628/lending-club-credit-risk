from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> list[str]:
    """
    this function will return the list of requirement
    """
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [request.replace("\n", "") for request in requirements]

    return requirements


setup(
    name="lending-club-credit-risk",
    version="0.1.0",
    description="Production-grade credit risk and lending decision system using Lending Club data",
    author="anuj kumar",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements("requirements.txt"),
)
