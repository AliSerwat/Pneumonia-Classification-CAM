from setuptools import setup, find_packages
from pathlib import Path

def parse_requirements(filename: str) -> list[str]:
    '''Load requirements from a pip requirements file.'''
    with open(Path(__file__).parent / filename, 'r') as f: # Ensure filename is resolved relative to setup.py
        lineiter = (line.strip() for line in f)
        return [line for line in lineiter if line and not line.startswith('#')]

# Get the long description from the README file
try:
    long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = "A package for pneumonia detection from X-ray images using CAM."


setup(
    name="pneumonia_cam",
    version="0.1.0",
    author="AI Health Contributor", 
    author_email="contributor@example.com", 
    description="A package for pneumonia detection from X-ray images using CAM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/pneumonia_cam_project", # Replace with actual URL
    packages=find_packages(exclude=['notebooks', 'scripts*', 'tests*']), 
    # pneumonia_cam.bin will be included as it's under pneumonia_cam/
    
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.8', 
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", # Adding 3.11
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Applications", # Corrected typo
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        'console_scripts': [
            'preprocess_pneumonia_data=pneumonia_cam.bin.preprocess_data:main',
            'train_pneumonia_model=pneumonia_cam.train:main',
            'evaluate_pneumonia_model=pneumonia_cam.evaluate:main',
        ],
    },
    # Example of including package data if needed in the future
    # package_data={
    #     'pneumonia_cam': ['config_files/*.yaml'], 
    # },
    # include_package_data=True, 
)
