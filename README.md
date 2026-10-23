# Music Modeler – a piano tutor app: project website

The website for Music Modeler, a piano tutor app. It will be published with GitHub Pages at
https://arunanagabhushan-cmd.github.io/piano-tutor/

## Previewing before publication

While this repository is private, the site is not online yet. To view it:

1. Click **Code → Download ZIP**, unzip it, and open `index.html` in a browser.
   The main page, presentation, walkthrough and notes all work offline.
2. Read the technical report here on GitHub: open [`report.md`](report.md).
   (The site's "Report" link points to `report.html`, which GitHub creates only
   once the site is published, so it does not open in the offline copy.)

| File | Page |
|---|---|
| `index.html` | Main page |
| `slides.html` | Presentation |
| `walkthrough.html` | Toolset walkthrough |
| `notes.html` | Notes on the numbers |
| `report.md` | Technical report (published as `report.html`) |
| `assets/` | Figures and screenshots |

## Editing the report

Edit `report.md`, then commit and push. GitHub Pages rebuilds the site and
publishes the report as `report.html`, styled by `_layouts/default.html`,
usually within a minute or two. If the page does not update, the repository's
**Actions** tab shows the build and any error.

Images in the report use paths relative to this folder (for example
`assets/figures/fig1-scoring-pipeline.svg`).

The code lives in two separate repositories:
[piano-tutor-app](https://github.com/arunanagabhushan-cmd/piano-tutor-app) and
[piano-tutor-eval](https://github.com/arunanagabhushan-cmd/piano-tutor-eval).
