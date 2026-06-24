"""Sphinx configuration for FactoriaX documentation."""

project = "FactoriaX"
copyright = "2026, Mickey Beurskens"
author = "Mickey Beurskens"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
]

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
}

nb_execution_mode = "off"

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/mickeybeurskens/factoriax",
    "use_repository_button": True,
    "show_toc_level": 2,
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

exclude_patterns = ["_build", "media", "profiling", "Thumbs.db", ".DS_Store"]
