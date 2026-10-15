# data/

This folder is **git-ignored** (except this file). WikiArt images belong to their rights holders;
the dataset card says "license: unknown", so we commit the *recipe*, not the images.

Rebuild the exact subset used in the report:

```bash
pip install -r requirements.txt
bash scripts/build_dataset.sh          # -> data/wikiart_style5/
```

Result: an ImageFolder tree (`<class>/<id>.jpg`, 128x128 centre crops), `metadata.csv`
(one row per image: path, label, label_idx, style, genre, artist...) and `subset_info.json`
(exact command, pinned dataset revision, counts).

For a quick smoke test without downloading anything: `python scripts/make_fake_dataset.py`
(synthetic coloured patterns, **not** paintings - numbers obtained on it mean nothing).
