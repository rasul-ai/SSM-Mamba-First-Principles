# Mamba blog: files for Medium

- `mamba_blog.docx`: the full post, with every figure placed where it belongs and its file name under the caption.
- `figures/png/`: upload these to Medium. 2400 px wide (2.4× the 1000 px design), white background; the cover is dark.
- `figures/svg/` and `figures/pdf/`: vector versions of the same figures, for editing in Figma, Illustrator or Inkscape.
- `figures/html/`: each figure as a standalone web page (the design source). Open in a browser; edit text or colors in any editor.
- `snippets/`: every code block in the post as its own file, ready to paste into a Medium code block.
- `mamba_lm.py`: the full script the snippets come from.

## Posting on Medium

1. Paste the text from the docx. Headings, bold and italics carry over.
2. For each figure, delete the placeholder and upload the PNG of the same name. Paste the italic caption under it.
3. Code blocks: press ``` (three backticks) on an empty line in Medium to start a code block, then paste the matching file from `snippets/`. Pasted docx code loses its formatting.
4. Use `fig00-cover.png` as the first image so it becomes the story's preview image.
5. Equations are plain text (for example `h_t = Ā·h_(t-1) + B̄·x_t`), so they survive pasting as is.

## Figures

| File | Caption |
|---|---|
| fig00-cover | Cover |
| fig01-three-ways | Three ways to carry the past forward |
| fig02-ssm | A state space model: two equations, four matrices |
| fig03-discretization | Discretization: the step size Δ sets the resolution |
| fig04-s4-views | S4: one model, two ways to compute it |
| fig05-selection | Selection: each token chooses its own Δ, B and C |
| fig06-parallel-scan | The parallel scan: eight words, three rounds |
| fig07-mamba-block | Inside a Mamba block |
| fig08-mamba2 | Mamba-2: one projection in, attention-like chunks inside |
| fig09-lookup-vs-state | At the word "got": look everything up, or read the notebook |
| fig10-memory | Memory carried while generating, as the text grows |
| fig11-training | Validation loss while training on Tiny Shakespeare |

Figures 10 and 11 use real numbers: figure 10 is computed from the model's sizes (4 KB of key/value cache per token for a transformer of the same width and depth, an 80 KB state for Mamba), and figure 11 is the validation loss logged by `mamba_lm.py train` with default settings.
