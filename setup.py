from setuptools import find_packages, setup

setup(
    name="nzgd",  # Your package name
    version="0.1.0",
    packages=find_packages(),  # Automatically finds all packages
    include_package_data=True,  # Ensures additional files (like data) are included
    url="https://github.com/ucgmsim/nzgd",
    description="For extracting data from the New Zealand Geotechnical Database (NZGD)",
    entry_points={},
)
