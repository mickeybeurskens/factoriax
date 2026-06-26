SPHINXBUILD = JAX_PLATFORMS=cpu uv run --group docs sphinx-build
NBSTRIPOUT  = uv run --group dev nbstripout
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean strip-notebooks

help:
	$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

strip-notebooks:
	find . -name "*.ipynb" -not -path "./_build/*" -exec $(NBSTRIPOUT) {} +

html: strip-notebooks
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)" api/generated

.DEFAULT_GOAL := html
