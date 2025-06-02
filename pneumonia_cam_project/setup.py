from setuptools import setup, find_packages
from pathlib import Path

def read_requirements():
    """Reads requirements from requirements.txt"""
    req_path = Path(__file__).parent / 'requirements.txt'
    if not req_path.exists():
        return []
    with open(req_path, 'r') as f:
        # Filter out comments and empty lines
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith('#')
        ]
    # The comment about removing version constraints was a bit misleading.
    # It's standard to pass requirements with versions to install_requires.
    # Pip will handle the resolution.
    return requirements

setup(
    name='pneumonia_cam',
    version='0.1.0',
    description='A project for pneumonia detection from chest X-rays using Class Activation Mapping.',
    author='AI Model (Originally from Jupyter Notebook)',
    author_email='example@example.com', # Placeholder
    long_description=open('README.md').read() if Path('README.md').exists() else '',
    long_description_content_type='text/markdown',
    url='https://github.com/example/pneumonia_cam_project', # Placeholder
    packages=find_packages(where='.'), # find_packages will find 'pneumonia_cam'
    package_dir={'': '.'}, # Tells setuptools that packages are under the project root
    install_requires=read_requirements(),
    python_requires='>=3.8', # Specify Python version compatibility
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License', # Assuming MIT License will be added
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'train_pneumonia_model=pneumonia_cam.train:main',
            'evaluate_pneumonia_model=pneumonia_cam.evaluate:main_evaluate',
            # 'run_pneumonia_app=pneumonia_cam.app:main', # Add when app.py exists
        ],
    },
    # Include_package_data can be used if you have non-code files inside your package
    # package_data={'pneumonia_cam': ['assets/*']}, # Example if assets were inside the package
)
