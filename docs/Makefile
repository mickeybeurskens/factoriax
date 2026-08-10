RUNNER ?= uv run --group docs 
SPHINXBUILD = JAX_PLATFORMS=cpu $(RUNNER) sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean 

help:
	$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html: 
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)" api/generated

.DEFAULT_GOAL := html
