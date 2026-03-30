# Paper Writing Setup

Local LaTeX workflow using tectonic, nvim + VimTeX, and zathura.

## Dependencies

```
paru -S tectonic zathura zathura-pdf-mupdf
```

nvim plugins are handled by lazy.nvim and will install automatically on next launch.

## Usage

**Quick compile from the terminal:**

```
cd paper
tectonic main.tex
```

First run downloads required LaTeX packages (cached after that).

**Full editing workflow:**

```
cd paper
zathura main.pdf &
nvim main.tex
```

VimTeX compiles on save. Zathura auto-reloads the PDF.

## VimTeX Keybindings

| Key | Action |
|-----|--------|
| `\ll` | Start/stop continuous compilation |
| `\lv` | Forward search (jump from source to PDF location) |
| `\le` | Open error quickfix list |
| `\lt` | Toggle table of contents |
| ctrl+click in zathura | Reverse search (jump from PDF to source line) |

## File Structure

```
paper/
  main.tex          # root document
  references.bib    # bibliography
  neurips_2025.sty  # conference style (swap for other templates)
  sections/         # split content via \input{sections/intro}
  figures/          # plots and diagrams
```
