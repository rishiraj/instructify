import setuptools
import subprocess
import os

instructify_version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

if "-" in instructify_version:
    # when not on tag, git describe outputs: "1.3.3-22-gdf81228"
    # pip has gotten strict with version numbers
    # so change it to: "1.3.3+22.git.gdf81228"
    # See: https://peps.python.org/pep-0440/#local-version-segments
    v,i,s = instructify_version.split("-")
    instructify_version = v + "+" + i + ".git." + s

assert "-" not in instructify_version
assert "." in instructify_version

assert os.path.isfile("instructify/version.py")
with open("instructify/VERSION", "w", encoding="utf-8") as fh:
    fh.write("%s\n" % instructify_version)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="instructify",
    version=instructify_version,
    author="Rishiraj Acharya",
    author_email="heyrishiraj@gmail.com",
    description="Instructify 📝 for easy Fine-Tuning preparation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rishiraj/instructify",
    packages=setuptools.find_packages(),
    package_data={"instructify": ["VERSION"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={"console_scripts": ["instructify = instructify.main:to_train_dataset"]},
    install_requires=requirements,
)
