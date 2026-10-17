# annotations/

Human judgments about the datasets: image ids and verdicts only, never images.

`wikiart_style5_duplicates.csv`: verdicts on the candidate duplicate pairs found by `scripts/find_duplicates.py`
(the most similar pairs by ResNet-18 cosine similarity or perceptual hash), judged by eye on 2026-10-17.
Verdicts: `same_image` (same artwork, another scan or crop), `same_composition` (same artwork, other colours),
`same_hidden_painter` (different works by the same unnamed painter), `different`.

How `museum_guide.data.load_metadata` uses it:
- every verdict except `different` links two images; linked images form one group and always land in the same split;
- an image whose `same_image` copies carry different style labels is removed with all its copies (label noise).
