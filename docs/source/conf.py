# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pySembrane'
copyright = '2024, NahyeonAn'
author = 'NahyeonAn'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autosummary',
                    'sphinx.ext.doctest',
                    'sphinx.ext.intersphinx',
                    'sphinx.ext.todo',
                    'sphinx.ext.coverage',
                    'sphinx.ext.ifconfig',
                    'sphinx.ext.napoleon',
                    "sphinx_rtd_theme",
                    'sphinx.ext.autodoc',
                    'sphinx.ext.ifconfig',
                    'sphinx.ext.viewcode',
                    'sphinx.ext.githubpages',
                    'sphinx.ext.mathjax',
                    ]

templates_path = ['_templates']
exclude_patterns = []

numfig = True
numfig_secnum_depth = 0

math_numfig = True
math_eqref_format = "Eq.{number}"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = 'images/pySembraneMain.png'
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


html_static_path = []

html_theme_options = { 'style_nav_header_background': '#FFFFFF',
                      'logo_only': True}