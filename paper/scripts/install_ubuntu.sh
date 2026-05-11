#!/usr/bin/env bash
# Install paper-writing dependencies on Ubuntu.
set -euo pipefail

sudo apt-get update
# Ubuntu ships zathura-pdf-poppler (mupdf variant was dropped in noble).
sudo apt-get install -y --no-install-recommends \
    zathura \
    zathura-pdf-poppler \
    curl \
    ca-certificates

# tectonic: install the official static binary into /usr/local/bin.
# The Ubuntu apt package lags upstream, so we fetch the latest release directly.
if ! command -v tectonic >/dev/null 2>&1; then
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    pushd "$tmpdir" >/dev/null
    curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
    sudo install -m 755 tectonic /usr/local/bin/tectonic
    popd >/dev/null
fi

echo "Done. Verify with: tectonic --version && zathura --version"
