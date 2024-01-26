from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

    return requirements

setup(
    name='PartII Project -- iHMM on PoS Tagging',
    version='1.0',
    author='Binglun Li',
    author_email='bl499@cam.ac.uk',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'hmm-test=src.synthetic_test_hmm.run_hmm:main',
            'lang-test=src.synthetic_test_lang.run_lang:main',
        ],
    },
)

# if __name__ == '__main__':
#     print(get_requirements('requirements.txt'))