"""Tests for factoriax.analysis.utils."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from factoriax.analysis.utils import resolve_ax


class TestResolveAx:
    def test_none_creates_new_figure(self) -> None:
        fig, ax = resolve_ax(None, figsize=(6, 4))
        assert fig is not None
        assert ax is not None
        assert ax.figure is fig
        plt.close(fig)

    def test_none_respects_figsize(self) -> None:
        fig, ax = resolve_ax(None, figsize=(8, 3))
        w, h = fig.get_size_inches()
        assert (w, h) == (8.0, 3.0)
        plt.close(fig)

    def test_existing_ax_returns_its_figure(self) -> None:
        existing_fig, existing_ax = plt.subplots(figsize=(5, 5))
        fig, ax = resolve_ax(existing_ax, figsize=(99, 99))
        assert fig is existing_fig
        assert ax is existing_ax
        plt.close(existing_fig)

    def test_existing_ax_ignores_figsize(self) -> None:
        existing_fig, existing_ax = plt.subplots(figsize=(5, 5))
        resolve_ax(existing_ax, figsize=(99, 99))
        w, h = existing_fig.get_size_inches()
        assert (w, h) == (5.0, 5.0)
        plt.close(existing_fig)
