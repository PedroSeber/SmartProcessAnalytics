from setuptools import setup, find_packages

name = 'Smart Process Analytics'
author = 'Pedro Seber, forked from code by Vicky Sun'
author_email = 'pseber@mit.edu'
license = 'MIT'
version = '1.5.1'
description = 'Smart Process Analytics (SPA) is a software package for automatic machine learning. Given user-input data (and optional user preferences), SPA automatically cross-validates and tests ML and DL models. Model types are selected based on the properties of the data, minimizing the risk of data-specific variance.'
packages = find_packages(where = 'Code-SPA', include = ['SPA.py'])
package_dir = {'': 'Code-SPA'}
# Note: other versions of these packages most likely also work, but we have not determined what are the minimum version requirements
all_dependencies = [
        'notebook>=6.4.11',
        'scikit-learn>=1.3.2',
        'matplotlib>=3.5.1',
        'pandas>=1.4.2',
        'statsmodels>=0.13.2',
        'seaborn>=0.11.2',
        'openpyxl>=3.0.9',
        'xlrd>=2.0.1',
        'ace-cream>=0.4',
        'torch>=2.0.0',
        'skglm>=0.3.1']

try: # Regular install with ace-cream
    setup(name = name, author = author, author_email = author_email, license = license, version = version, description = description, zip_safe = False, packages = packages, package_dir = package_dir, install_requires = all_dependencies)
# ace-cream tends to fail to install on Windows and Macs. This will install everything but ace-cream
except: # This probably should be except ModuleNotFoundError, but I need more information on how ace-cream fails to be installed on Windows (specifically, whether it also fails as a ModuleNotFoundError)
    all_dependencies_without_ace_cream = [elem for elem in all_dependencies if 'ace-cream' not in elem]
    setup(name = name, author = author, author_email = author_email, license = license, version = version, description = description, zip_safe = False, packages = packages, package_dir = package_dir, install_requires = all_dependencies_without_ace_cream)
