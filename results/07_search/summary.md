# Free-text search (Part 2)

Collection: 1496 artworks. CLIP ViT-B-32-quickgelu/openai on data/wikiart_style5_224; TF-IDF on the metadata (style, genre, artist).

## Metadata queries (relevance from metadata.csv)

TF-IDF reads the field that defines relevance: an upper bound, not a fair comparison. Random = expected precision of a random ranking (share of relevant candidates). TF-IDF ignores English stop words and the words every artwork matches (painting, picture, image, artwork): this list was added after 'a baroque painting' retrieved 'genre painting' records (precision 0.00 before, 1.00 after).

| query | clip_P@5 | clip_P@10 | tfidf_P@5 | tfidf_P@10 |
|---|---|---|---|---|
| an impressionist painting | 1.00 | 1.00 | 1.00 | 1.00 |
| a baroque painting | 1.00 | 1.00 | 1.00 | 1.00 |
| a cubist painting | 1.00 | 1.00 | 1.00 | 1.00 |
| a Japanese ukiyo-e woodblock print | 1.00 | 1.00 | 1.00 | 1.00 |
| a pop art image | 1.00 | 1.00 | 1.00 | 1.00 |
| a still life | 1.00 | 1.00 | 1.00 | 1.00 |
| a scene of everyday life | 0.20 | 0.30 | 0.00 | 0.00 |
| a sketch | 1.00 | 0.80 | 1.00 | 1.00 |
| a landscape | 0.80 | 0.40 | 1.00 | 0.80 |
| **mean** | **0.889** | **0.833** | **0.889** | **0.867** |

Random ranking, mean expected precision: 0.222

## Content queries (relevance judged by hand, search_eval/judgments.csv)

| query | clip_P@5 | clip_P@10 | tfidf_P@5 | tfidf_P@10 | random_P@5 | random_P@10 |
|---|---|---|---|---|---|---|
| a stormy sea | 0.80 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |
| portraits of children | 0.80 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |
| flowers in a vase | 0.80 | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 |
| a bowl of fruit | 0.60 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |
| a boat on the water | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| a mountain | 0.80 | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 |
| a winter scene with snow | 0.60 | 0.60 | 0.00 | 0.00 | 0.00 | 0.00 |
| a horse | 0.80 | 0.80 | 0.00 | 0.00 | 0.00 | 0.00 |
| a bird | 0.80 | 0.70 | 0.00 | 0.10 | 0.00 | 0.00 |
| a musical instrument | 1.00 | 0.80 | 0.00 | 0.00 | 0.00 | 0.00 |
| a close-up of a woman's face | 0.20 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 |
| a city street | 0.20 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
| angels | 0.20 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 |
| a skull | 0.80 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |
| a bridge | 0.60 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 |
| a crowd of people | 1.00 | 0.90 | 0.20 | 0.10 | 0.00 | 0.10 |
| a scene at night | 1.00 | 0.60 | 0.00 | 0.10 | 0.00 | 0.00 |
| a waterfall | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| warriors in battle | 0.20 | 0.20 | 0.00 | 0.10 | 0.00 | 0.00 |
| a bottle or a glass | 0.80 | 0.90 | 0.20 | 0.10 | 0.00 | 0.00 |
| **mean** | **0.650** | **0.550** | **0.020** | **0.025** | **0.000** | **0.005** |

Judge consistency: 67 already-judged pairs were shown again, blind, among the new pairs of the second round: 91% same verdict (relevant both times 36, not relevant both times 25, relevant only the first time 1, only the second time 5). The first-round verdicts are the ones used above.

## Phrasing: the brief's wording vs the short query

The brief's questions, verbatim, next to the short query on the same topic (left out of the mean above). Shared = images that appear in the top 10 of both wordings.

| query | clip_P@5 | clip_P@10 | tfidf_P@5 | tfidf_P@10 | random_P@5 | random_P@10 | CLIP top-10 shared | TF-IDF top-10 shared |
|---|---|---|---|---|---|---|---|---|
| c00: a stormy sea | 0.80 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |  |  |
| c20: Which painting shows a stormy sea? | 0.60 | 0.40 | 0.00 | 0.00 | 0.00 | 0.00 | 8 | 10 |
| c01: portraits of children | 0.80 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |  |  |
| c21: I would like to see portraits of children | 0.80 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 7 | 0 |

## CPU cost per query (1 thread)

- CLIP: median 115.7 ms (text encoder alone 89.6 ms); model 605 MB (text tower 63,428,097 parameters); image index 3.1 MB, built offline at 96 ms per image
- TF-IDF: median 1.09 ms; index 0.43 MB, 644 character n-grams
