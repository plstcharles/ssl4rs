import re

import setuptools

version_file_path = "ssl4rs/_version.py"
version_regex_pattern = r"^__version__ = ['\"]([^'\"]*)['\"]"
with open(version_file_path) as fd:
    match_output = re.search(version_regex_pattern, fd.read(), re.M)
version_str = match_output.group(1)

setuptools.setup(
    name="ssl4rs",
    version=version_str,
    description="TODO",
    author="plstcharles",
    author_email="pierreluc.stcharles@gmail.com",
    url="TODO",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=setuptools.find_packages(),
)
