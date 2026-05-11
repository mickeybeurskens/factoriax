# Spec: FactoriaX EWRL 2026 Submission

## Objective

Ship a short paper to **EWRL 2026** (deadline 2026-05-29, AOE) introducing
FactoriaX as a JAX-native, single-agent factory-building benchmark for
long-horizon reinforcement learning. The paper announces the engine, the
two shippable benchmarks built on it (`rocket` and `skills_challenges`),
demonstrates throughput and a learnable-but-unsolved difficulty target,
and stakes the research territory the engine opens up. Multi-agent is
framed as a *design property* of the environment, not as evaluated work.

The paper is a deliberate **preview**, not the final artifact. The full
archival submission targets **AAAI** later, with multi-agent results,
deeper baselines, and a methods contribution layered on top. EWRL is
non-archival (no proceedings), so submitting there does not burn the
work. The strategic value is: feedback from the European RL community,
visibility before AAAI submission, and a date-stamped public statement
of the contribution.

### Target reader

A reviewer who has built or used Craftax / Brax / Jumanji / SocialJax,
knows the GPU-accelerated RL benchmark space, and wants to decide in
the first column whether FactoriaX is doing something the existing
benchmarks aren't. They should reach that decision by the end of the
introduction.

### What the paper claims

1. There is a real, unfilled gap in GPU-accelerated RL benchmarks at
   the intersection of *persistent shared infrastructure*, *coupled
   production dependencies*, *spatial logistics*, and *resource
   finitude*. (Argument supported by comparison table; the table is the
   anchor.)
2. FactoriaX fills this gap at competitive throughput: 200k steps/sec
   on a single laptop GPU.
3. The benchmarks built on FactoriaX are *learnable but unsolved* under
   the obvious baseline (PPO + skill-level curriculum): partial
   progress on the achievement DAG, no end-to-end rocket launch. (Needs to be validated).
4. The engine and benchmarks are immediately usable: gymnax interface,
   `uv sync` install, scripted reference policies for sanity, runnable
   on a laptop GPU.

### What the paper does **not** claim

- That FactoriaX is solved. Saying "PPO doesn't solve it" is a feature.
- That multi-agent is benchmarked. It is supported architecturally and
  shown in figures; numerical claims about MARL are deferred to AAAI.
- That FactoriaX subsumes Craftax. We are adjacent, not competing.

## The Strategy: EWRL Preview, AAAI Full

The two papers play different roles. Naming them up front so we can
make consistent decisions throughout writing.

| Aspect          | EWRL 2026 (this paper)               | AAAI follow-up (next paper)              |
|-----------------|--------------------------------------|------------------------------------------|
| Length          | ≤9 pages incl. figures (refs, checklist, appendix free) | Full paper (8–10 pages excl. refs at AAAI) |
| Tone            | Preview / announcement                | Archival contribution                    |
| Empirical scope | PPO baselines on `rocket` + scripted on `skills_challenges` | + MARL baselines, curriculum study, deeper ablations |
| Multi-agent     | Design property, future work          | Evaluated with baselines                 |
| Novelty bar     | Engine + benchmarks + crisp gap argument | + methodological contribution (e.g. probe-guided curriculum from `proposal.md`) |
| Audience        | European RL community, workshop      | Broad ML community, archival            |
| Reviewer signal | "Interesting direction, looking forward to the full paper" | "Solid empirical contribution"           |

This means: at EWRL, we are not trying to hide that this is a preview.
We are signalling the research program clearly so the AAAI paper feels
inevitable rather than surprising.

## What Makes a Great EWRL / Benchmark Paper

Research-derived findings the writing must honor. Pulled from the
Craftax paper (ICML 2024 Spotlight, closest analogue), RLC 2025
Outstanding Paper rationale, prior benchmark literature, and EWRL's
known reviewer pool.

### 1. The gap argument is the load-bearing claim

Craftax's contribution survives because its gap argument is sharp:
existing open-ended benchmarks are either too slow (Crafter, NetHack,
Minecraft) or too simple (Minigrid, Procgen). That's a clean 2x2 the
reader holds in their head.

FactoriaX's gap argument needs to be just as sharp. Draft version:
existing fast benchmarks test reactive, decomposable skills; the
interesting real-world problems (supply chain, multi-robot
construction, infrastructure) are not decomposable and involve
persistent shared state. No fast benchmark tests this.

**Rule:** the gap argument lives in the introduction *and* in a
comparison table in section 2. If a reader skims and only reads those
two things, they should already get it.

### 2. Throughput is the entry ticket, not the contribution

Every GPU-RL benchmark paper reports steps/sec. Our 200k steps/sec on
a laptop GPU is a respectable number but it is *table stakes*, not
the contribution. Report it precisely, with hardware, batch size,
map size, and what's being measured (action steps, not env
transitions if those differ). Move on. Don't dwell. If a CPU-baseline
speedup multiple is reported, it must be measured under matched
conditions, not extrapolated.

### 3. Learnable but unsolved is the sweet spot

Craftax's PPO reached ~9% of optimal on the full benchmark. That's
the magic zone: enough progress that the benchmark is alive, far
enough from ceiling that there's real research to do.

For FactoriaX the equivalent target is: PPO unlocks the early
achievements (mine, smelt, craft basic items) but does not approach
the full rocket launch. The achievement breakdown is the figure.

### 4. Achievements are a discourse aid

Discrete achievements give reviewers and future authors a shared
vocabulary. Craftax benefits from this enormously — papers can say
"we improve on the *find_diamond* achievement" without ambiguity.

### 5. Reproducibility is checked, not claimed

RLC 2025's "Resourcefulness" award category went to PufferLib for
making large-scale experimentation accessible. The signal is clear:
benchmark papers are judged on whether someone else can actually run
them. Concretely for us:
- Single GPU runs (RTX 3050 or A100, name both)
- Multi-seed (5 seeds minimum) with shaded confidence regions
- `uv sync` install, one-line baseline command
- Code released at submission, not "coming soon"
- All training curves in the paper reproducible from a single script

### 6. Comparison table is the centerpiece artifact

A single table comparing FactoriaX to Brax / Jumanji / PGX /
MiniGrid / Craftax / Multi-Agent Craftax / SocialJax across rows
that operationalize the gap: persistent state, shared infrastructure,
DAG dependencies, spatial logistics, resource finitude, achievement
structure, multi-agent supported, single-GPU throughput. Designed so
FactoriaX is the only row with most cells filled.

### 7. Honest limitations earn trust

State what FactoriaX is *not* for: continuous control, vision-based
agents, real-world transfer, language-conditioned tasks. Limitations
are not a weakness section, they are a positioning section.

### 8. Future work points at the AAAI paper

The future work section is not a wish list. It is a research program:
solving the full game via probe-guided curricula, safety analysis via
representation probes, multi-agent dynamics on shared infrastructure.
This signals to EWRL reviewers that AAAI is the next stop and to AAAI
reviewers (when we get there) that the program was already articulated.

### 9. Style: anti-hype, concrete, technically confident

Per `CLAUDE.md`: direct but explanatory, flowing prose, no
superlatives, no marketing speak. The throughput number is the
throughput number; do not call it "blazing fast." Concrete benefits
in real terms ("more training runs per dollar," not "faster").

### 10. EWRL-specific: workshop framing is allowed

EWRL is non-archival. Reviewers know that and read in-progress work
more permissively than at NeurIPS. We are explicitly allowed to say
"the full study is forthcoming." Use this room. Don't over-claim to
compensate for the workshop format.

### 11. The engine is a contribution, and so are the benchmarks
The design decision to decouple the engine from the benchmarks is crucial. 
Benchmark engine code is retooled all the time. Factoriax makes this dynamic a first class citizen. 
Similar to how Mujoco provides a physics engine, factoriax provides all the bits and pieces needed 
for first person factory building benchmarks.

## EWRL 2026 Format Constraints

Pulled verbatim from `ewrl_2026.pdf` (the official formatting
instructions distributed with the style bundle). Consolidated here so
editorial decisions have one place to check against.

### Length

- **9 pages maximum, including all figures and tables.**
- References, the reproducibility checklist, and the optional
  technical appendix do *not* count toward the 9 pages.
- Papers that exceed the limit will not be reviewed.

### Style file

- Use `ewrl_2026.sty` exactly as shipped. Tweaking it is grounds for
  rejection.
- Three style options: `final` (camera-ready only), `preprint`
  (arXiv-style, deanonymized, "Preprint. Work in progress." footer),
  and `nonatbib` (if natbib clashes with another package).
- **At submission time, omit both `final` and `preprint`.** The
  default anonymizes the author block and inserts line numbers for
  reviewers.
- Do not reference reviewer line numbers in the paper text — they are
  stripped at camera-ready time.

### Submission is double-blind

- Author names, affiliations, emails are anonymized at submission.
- **Self-references must use third person.** Write "In the prior
  work of Jones et al. [4]," not "In our prior work [4]."
- If citing an in-progress work by the same authors that is not
  publicly available, use "A. Anonymous" as the author and bundle
  the anonymized paper in supplementary material.

### Typography (auto-handled by the style file, but worth knowing)

- 10pt body, 11pt leading, Times New Roman.
- Text box: 5.5 inches wide by 9 inches tall, left margin 1.5 inches.
- Title: 17pt bold, between horizontal rules.
- Headings: lower case except first word and proper nouns, flush
  left, bold; 12pt for level 1, 10pt for levels 2 and 3.
- Paragraphs separated by half a line space (5.5pt), no indentation.

### Citations

- `natbib` loaded by default. Use `\citet{}` for inline narrative
  ("Jones et al. (1995) showed...") and `\citep{}` for parenthetical
  ("(Jones et al. 1995)").
- Citation format is author/year or numeric; pick one and be
  consistent.
- Reference section uses an unnumbered first-level heading. Font may
  be reduced to 9pt (`\small`) if space is tight.

### Tables

- No vertical rules (use `booktabs`: `\toprule`, `\midrule`,
  `\bottomrule`).
- Title appears *above* the table, lower case, with one line space
  before and after.
- Numbered consecutively.

### Figures

- `\includegraphics` from `graphicx`, with width as a multiple of
  `\linewidth` (e.g. `width=0.8\linewidth`) to avoid margin overflow.
- Caption appears *below* the figure, lower case, with one line
  space above and below.
- Color is fine, but make sure the paper remains legible when printed
  in black and white.

### Math

- Use LaTeX (`\[...\]`, `align`, `equation`) for display math, not
  bare `$$...$$` — the latter breaks reviewer line numbering.

### PDF requirements

- **US Letter** paper size, not A4.
- Only Type 1 or Embedded TrueType fonts. No Type 3.
- Verify with `pdffonts main.pdf` — every "type" column should be
  "Type 1," "TrueType," or "CID Type 0".
- The `\bbold` package uses bitmap fonts and is forbidden; use
  `amsfonts` / `amssymb` (`\mathbb{R}` etc.) instead.

### What goes in the appendix

The technical appendix has **no page limit**, so push to it:
- Full hyperparameter tables
- Per-seed training curves (main body shows means)
- Extended environment specification (action space, observation
  channels, reward decomposition)
- Sprite atlas / render examples beyond what fits in section 3
- Ablations that don't fit the main story

The trade-off: reviewers read appendices only if they care. Anything
load-bearing for the contribution belongs in the body.

## Paper Stack

- LaTeX, EWRL 2026 style file (`ewrl_2026.sty`, already in place,
  `\usepackage[preprint]{ewrl_2026}` during drafting → drop the
  `preprint` option at submission for double-blind anonymization).
- US Letter paper size (tectonic default; do not override to A4).
- Compiler: `tectonic` (single-binary, no aux file churn).
- Editor: nvim + VimTeX, viewer: zathura (continuous compile via
  `<space>ll`).
- Bibliography: `references.bib`. Citations via `natbib` (loaded by
  the style file): `\citet{}` for inline narrative, `\citep{}` for
  parenthetical.
- Tables: `booktabs` (already imported in `main.tex`), no vertical
  rules. Data comes from experiment dumps in `runs/`, no
  auto-generation.
- Figures: produced offline by Python scripts (`paper/figures/scripts/`),
  saved as PDF, included via `\includegraphics[width=...\linewidth]`.
- Math: `amsfonts` / `amssymb` only (no `\bbold` — it's bitmap and
  forbidden by the PDF font rules).
- All numerical claims traceable to a single source (a run id, a
  script, or a citation).

## Commands

```
# Compile during drafting (deanonymized, with "Preprint" footer)
# Requires \usepackage[preprint]{ewrl_2026} in main.tex
cd paper
tectonic main.tex

# Compile for submission (anonymized, with reviewer line numbers)
# Drop the [preprint] option from \usepackage before this run
tectonic main.tex

# Verify PDF compliance after the submission compile
pdfinfo main.pdf | grep -E 'Pages|Page size'   # ≤9 pages, US Letter
pdffonts main.pdf | awk 'NR>2 {print $3}' | sort -u  # only Type1/TrueType/CID Type 0

# Continuous compile inside nvim
<space>ll              # toggle on/off
<space>lv              # forward search to current cursor
<space>le              # show errors

# Validate citations against bib
grep -oE '\\cite[pt]?\{[^}]+\}' paper/main.tex | \
    sed -E 's/.*\{([^}]+)\}/\1/' | tr ',' '\n' | sort -u

# Word count (rough, excludes preamble + appendix)
detex paper/main.tex | wc -w

# Page count check after compile
pdfinfo paper/main.pdf | grep Pages

# Spell check (one section at a time)
aspell --mode=tex check paper/sections/<section>.tex
```

## Project Structure

```
paper/
    main.tex                  # root document, currently 149 lines, intro drafted
    references.bib            # bibliography, single source for all citations
    ewrl_2026.sty             # conference style file (do not modify)
    SPEC.md                   # this file
    README.md                 # how to compile and edit locally
    scripts/                  # install scripts for tectonic/zathura
        install_ubuntu.sh
        install_arch.sh
    sections/                 # each section in its own file, \input from main
        intro.tex
        related.tex
        env.tex
        benchmarks.tex
        baselines.tex
        results.tex
        limitations.tex
        future_work.tex
    figures/                  # final PDFs
        screenshots/          # env renders
        plots/                # training curves, throughput
        diagrams/             # DAG, architecture
        scripts/              # source Python for generated figures
            throughput.py
            achievements.py
            comparison_table.py
```

The repo-wide structure (`factoriax/`, `baselines/`, `benchmarks/`,
`runs/`, `SPEC_SKILLS_BENCHMARK.md`) supplies the artifacts the paper
points at. Paper writing does not modify code; if code changes are
needed (e.g., new experiment runs), they happen as a separate task
under their own spec.

## Writing Style

Inherited from `CLAUDE.md`, with paper-specific notes layered on.

### From CLAUDE.md (load-bearing)

- Direct but explanatory. Take time to paint the full picture, don't
  just state facts.
- Flowing prose. Let sentences build on each other naturally. Skip
  headers if the content flows without them.
- Anti-hype. No superlatives. The work speaks for itself.
- Technically confident. Assume the reader is a smart RL person.
- Concrete benefits. "More training runs" not "faster."
- Natural punctuation. Commas and periods over dashes and semicolons.
- A bit of personality. Dry is boring.

### Paper-specific

- **Past tense for results, present tense for the environment.**
  "We trained PPO for 1B steps." "FactoriaX exposes a gymnax interface."
- **Active voice unless passive is clearer.** "We measured 200k
  steps/sec" not "200k steps/sec was measured."
- **First person plural** for our own work in this paper ("we trained,"
  "we measured"). Standard for papers, and fine under double-blind so
  long as the authors are not named.
- **Third person for prior work by the same authors.** Under EWRL's
  double-blind rule, "In our prior work [X]" is forbidden. Write "In
  the prior work of [Anonymous] [X]" or simply "Prior work [X] showed
  that..." The submission must read as if written by people other than
  the authors of any cited self-work.
- **Numbers in figures, claims in prose.** Every numerical claim in
  prose is cross-referenced to a figure, table, or appendix run.
- **No bullet lists in the body** unless genuinely enumerating. Prose
  is denser and reads better.
- **One idea per paragraph.** First sentence states the claim, body
  defends it.
- **Citations as evidence, not decoration.** Every `\cite` should be
  load-bearing for the surrounding claim. If a citation is decorative,
  cut it.

### Example of the target voice

> GPU-accelerated reinforcement learning environments have collapsed
> experiment turnaround from days to minutes, but most of them test
> narrow capabilities. Brax covers continuous locomotion. Jumanji
> targets combinatorial optimization. PGX handles board games.
> MiniGrid and BabyAI serve grid navigation with language grounding.
> Actions in these environments have short-lived consequences;
> optimal policies can be discovered without reasoning about
> persistent state.

This is the proposal.md voice. Keep it.

## Validation Strategy

Papers don't have unit tests. They have proxies. Validate against
these throughout writing, not at the end.

### Per-section validation

| Section          | Validation                                                                 |
|------------------|----------------------------------------------------------------------------|
| Abstract         | A non-RL ML researcher reads it once and knows what we did, why, and what's new. |
| Introduction     | Reader can write the comparison table from memory after one read.          |
| Related work     | Every cited work is positioned, not just listed. "X does Y, we differ in Z." |
| Environment      | Reader can describe the action space and observation space without scrolling. |
| Benchmarks       | Reader knows what "solving" each benchmark means by the end.               |
| Baselines        | All hyperparameters in appendix; main text has the curves.                 |
| Results          | Every claim has a number, every number has a source.                       |
| Limitations      | Reader trusts us more after reading this, not less.                        |
| Future work      | Reader sees a research program, not a wish list.                           |

### Claim-evidence audit

Before submission, run this manually:
1. Highlight every numerical or comparative claim in the body.
2. For each, point at the figure, table, citation, or appendix run id.
3. Anything that can't be pointed at gets either evidence or cut.

### Citation completeness pass

Before submission:
1. Every paper named in proposal.md and main.tex must have a bib entry.
2. Every bib entry must be cited at least once.
3. Resolve every `[CITE: ...]` placeholder currently in main.tex.

### Reproducibility check

A second person (or me on a fresh checkout) follows the README and
produces one of the training curves end to end. If this doesn't work,
fix before submission, not after.

## Boundaries

### Always

- Compile and visually check the PDF after every batch of edits.
- Keep `main.tex` under the EWRL page limit at all times (don't write
  16 pages and trim later; write 9 from the start).
- Update `references.bib` in the same commit as the citing prose.
- Cross-link claims to their source (figure, table, or appendix).
- Use `\input{sections/...}` to keep `main.tex` thin once sections
  exist as separate files.

### Ask first

- Adding a new section that wasn't in the planned structure.
- Cutting an experiment from the experimental plan.
- Changing the title or the central claim.
- Adding co-authors or affiliations (we are still TBD on this).
- Repointing the paper away from EWRL 2026 (e.g., to a different venue).
- Bringing in any of the three full proposals from `proposal.md` as
  empirical contributions — that escalates this from preview to full.

### Never

- Inflate claims. If PPO unlocks 6 of 17 achievements, say six, not "many."
- Use marketing adjectives. No "powerful," "novel," "state-of-the-art."
  (Adjectives that describe specific properties, like "JAX-native" or
  "GPU-accelerated," are fine.)
- Fabricate citations or numbers. Every datum has a source.
- Submit without running a final claim-evidence audit and citation pass.
- Block on a missing experiment when a documented limitation will do.

## Success Criteria

The paper is ready to submit when *all* of these are true.

- [ ] PDF compiles cleanly with `tectonic main.tex`, no warnings about
      missing references or undefined labels.
- [ ] Page count is ≤9 pages *including* figures and tables, with
      references, reproducibility checklist, and technical appendix
      not counted (per EWRL 2026 instructions).
- [ ] PDF is US Letter, not A4.
- [ ] Author block is anonymized (no `final` or `preprint` style
      option at submission time).
- [ ] Self-references to prior work by the same authors are written
      in third person, per the double-blind rule.
- [ ] Reproducibility checklist completed and attached.
- [ ] All fonts in the PDF are Type 1 or Embedded TrueType (check
      with `pdffonts main.pdf`).
- [ ] Tables use `booktabs` (no vertical rules) and captions are
      lower case except for proper nouns.
- [ ] Abstract is under 250 words and previews the four claims listed
      in the Objective.
- [ ] Comparison table (FactoriaX vs at least 6 prior benchmarks) is
      present and pulls its weight visually.
- [ ] Throughput numbers are reported with hardware, batch size, map
      size, and the exact unit measured.
- [ ] At least one training curve (PPO on `rocket`) is included with
      ≥5 seeds and a visible shaded confidence region.
- [ ] An achievement-breakdown figure shows which achievements PPO
      unlocks vs. which remain unsolved.
- [ ] Every `[CITE: ...]` placeholder in `main.tex` is resolved.
- [ ] Limitations section names at least four categories of task
      FactoriaX is not designed for.
- [ ] Future work section points at AAAI-target directions without
      using the word "AAAI."
- [ ] Code is released and the `README.md` install path works from a
      clean machine.
- [ ] Claim-evidence audit completed: every numerical claim in prose
      has a source.
- [ ] Citation pass completed: zero unresolved citations, zero unused
      bib entries.
- [ ] One independent reader (co-author when finalized, or me on a
      fresh read) has signed off.

## Open Questions

- **Co-author list and affiliations.** Placeholder block can stay
  through submission (it gets anonymized anyway), but the real list
  must be settled before the camera-ready deadline.
- **Title.** Current: "Factoriax - An Open Ended Multi Agent Factorio
  Benchmark." Single-agent reframe means "Multi Agent" should come
  out. Working alternative: "FactoriaX: A JAX-Native Benchmark for
  Long-Horizon Factory-Building Reinforcement Learning." TBD with the
  author.
- **Whether to include the `skills_challenges` benchmark in the main
  paper or appendix.** It is the smaller, faster-to-explain benchmark
  and might earn its space in the body. Decide once the rocket
  baseline section is drafted and we know how much room is left.
- **Single-GPU claim hardware.** RTX 3050 laptop is currently the
  measurement basis. Decide whether to also report an A100 or
  similar reference number for community comparability.
- **Reproducibility checklist source.** EWRL 2026 instructions refer
  to a checklist that does not count toward page limit, but the
  template doesn't ship one in the style bundle. Confirm on the
  OpenReview submission page whether NeurIPS-style checklist applies
  by default or a custom one is required.

## Next Step

The user (paper author) reviews this spec, corrects assumptions, and
either approves it or asks for changes. After approval, I switch into
scientific-editor mode and we work section by section through the
existing `main.tex` draft.
