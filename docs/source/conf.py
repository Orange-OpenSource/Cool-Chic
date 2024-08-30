# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'coolchic/'))
sys.path.insert(0, basedir)
# sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Cool-chic'
copyright = '2024 Orange'
author = 'Théo Ladune'
release = '3.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx_immaterial",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    'sphinx.ext.duration',
    'sphinx.ext.todo',
    "sphinx_copybutton",
    "sphinx.ext.autodoc.typehints",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False
# typehints_use_rtype = False
# typehints_document_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# autodoc_mock_imports = ["torch"]

# templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


# extensions.append("sphinx_immaterial")
html_theme = "shibuya"  # "piccolo_theme" # "sphinx_book_theme"  # furo
html_title = "Cool-chic"

pygments_style = "sphinx"
pygments_dark_style = "material"

# build the templated autosummary files
autosummary_generate = True
add_module_names = False

maximum_signature_line_length = 100
# Don't show class signature with the class' name.
autodoc_class_signature = "separated"
# Type hints both in the detailed parameter description and in the
# function signature
autodoc_typehints = "both"
napoleon_use_rtype = True
# typehints_defaults = 'comma'
typehints_document_rtype = False
typehints_use_rtype = True
typehints_use_signature = True
typehints_use_signature_return = True
hide_none_rtype = True

# Display only once the detail of the __init__ function
autoclass_content = "class"
# Show functions by order of appearance (not by alphabetical order)
autodoc_member_order = 'bysource'

# Wrap function signature which are too long
wrap_signatures_with_css = True

# # Replace Union and Optional by more concise expression (e.g. using |)
# python_transform_type_annotations_pep604 = True
# python_transform_type_annotations_concise_literal = True
# object_description_options = [
#     ("py:.*", dict(include_fields_in_toc=False)),
#     ("py:.*", dict(include_object_type_in_xref_tooltip=False)),
# ]

html_static_path = ["_static/"]
html_favicon = "_static/favicon_16x16.png"
html_theme_options = {
    "dark_code": True,
    "color_mode": "light",
    "light_logo": "_static/coolchic-logo-light.png",
    "dark_logo": "_static/coolchic-logo-dark.png",
    "globaltoc_expand_depth": 3,
    "accent_color": "orange",
    "announcement": (
        "<center> 🚀 Cool-chic 3.3 An even faster decoder! "
        "<a href=https://github.com/Orange-OpenSource/Cool-Chic>"
        "Check out the git repository</a> 🚀 </center>"
    ),
}

html_context = {
    "source_type": "github",
    "source_user": "Orange-OpenSource",
    "source_repo": "Cool-Chic",
}
# html_static_path = ['_static']
