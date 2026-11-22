# AGENTS.md — `1-intro/`

Deliberately kept in this subfolder instead of the repo root so each member's branch stays
independent and merging branches doesn't collide on a root `AGENTS.md`. Don't move it up.
A session started at the repo root won't load it until something in `1-intro/` is read.

## What this repo is
Course project (CS313) presenting and reimplementing **DGMAE** (Zheng et al., KDD 2025, arXiv:2506.19343).
It is currently **content-only**: no code, build, tests, lint, or CI. The `dgmae/` directory tree on
slide 16 of `../README.md` is a *planned* layout — it does not exist. Don't add scaffolding or run
"build/test" commands unless asked; there is nothing to run.

## Sources of truth
- `paper.pdf` — the paper itself. Verify every formula, equation number, table value, and section
  reference (Eq. 1–12, Figure 2/3, Sections 4.4/5.3) against it before editing claims.
- `../README.md` — canonical 27-slide outline in Vietnamese: per slide "Trên slide / Lời thoại /
  Cần chuẩn bị", plus the shared symbol table and the appendix checklist.
- `requirement.md` — assignment spec. It duplicates README slides 1–10, then adds presenter
  requirements starting at `# Yêu cầu của người thuyết trình` (~line 189). It holds the **real member
  names** where `../README.md` still has `[ĐIỀN tên]`.

## Conventions that are easy to break
- Write slide content in Vietnamese; keep symbols identical to the symbol table (`x_i^D`, `z_i^D`,
  `L_sym`, `p_c`, `p_τ`, `γ₁`, `γ₂`, …) across all slides.
- `[ĐIỀN]` = a team result still to fill in; `[KIỂM TRA]` = a claim not yet verified against the paper.
  These are the team's TODO markers: don't delete them, and don't invent numbers to replace them.
- Keep paper results and team results in **separate columns/tables**; that split is deliberate.
- The deliverable is a **LaTeX Beamer** deck (per `requirement.md`); no `.tex` file exists yet.
- `../README.md` Part 1 is 10 slides, but `requirement.md`'s presenter spec asks for a 7-slide Part 1
  (GNN/SSL → homophily → Figure 2 evidence → contributions → architecture). These conflict — confirm
  with the team which governs instead of silently rewriting one to match the other.

## Repo state / git
- Working branch `1-intro`; this folder is **untracked** — only the root `README.md` is committed, and
  `main` tracks `origin/main`. So this file is local-only until you `git add` it.
- `../.commandcode/taste/taste.md` is an empty tool artifact; ignore it.
