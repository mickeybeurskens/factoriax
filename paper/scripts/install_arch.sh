#!/usr/bin/env bash
# Install paper-writing dependencies on Arch Linux.
set -euo pipefail

if ! command -v paru >/dev/null 2>&1; then
    echo "paru not found. Install it first: https://github.com/Morganamilo/paru" >&2
    exit 1
fi

paru -S --needed tectonic zathura zathura-pdf-mupdf
