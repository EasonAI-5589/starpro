# STAR-Pro project page

Self-contained static project website. Serve this directory as the website root;
it has no runtime dependencies or external font/script requests.

Local preview from the repository root:

```sh
python3 -m http.server 4191 --bind 127.0.0.1 --directory website
```

The public URL is `https://easonai-5589.github.io/starpro/`.
Canonical and sharing metadata target that URL. The GitHub Pages workflow in
`.github/workflows/pages.yml` publishes `website/` when its files change on
`main`, and can also be run manually. Existing Python/research files are not
included in the published site.

## Sources and maintenance

- Paper: https://arxiv.org/abs/2609.05916 (v1, 2026-09-05).
- `starpro.bib`: official https://arxiv.org/bibtex/2609.05916 export, with braces
  added only to protect STAR-Pro capitalization.
- Authors, affiliation markers and release scope follow the public repository
  README and paper. No conference acceptance is claimed.
- Logo and Figures 1–4 are byte-for-byte copies of the repository originals.
  When replacing a paper figure, update both the repository original and its
  copy in `website/assets/` together.
- Headline measurements all refer to the same LLaVA-Video-7B setting. The
  speedup protocol is stated beside them and in the efficiency section.
- Research overview is a website summary; it is not labeled as the full abstract.
- No backend, analytics, tracker, generated research figure or data upload.

Verify desktop and mobile layout, image loading, section anchors, citation copy,
and citation download before publishing. Preserve the visible citation block so
it remains usable when JavaScript or clipboard permissions are unavailable.
