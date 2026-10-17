# Sample output

Produced by `python tools/make_sample_output.py` with the current pipeline and `config.json`.

| Video | People (counted by eye) | Unique visitors (ours) | by face | body-only | entry+exit events |
|---|---|---|---|---|---|
| record_20250620_183903 | 11 | 12 | 10 | 2 | 32 |
| record_20250620_184302 | 8 | 8 | 6 | 2 | 20 |

Each video folder contains:

- `logs/events.log` - tracking, embedding, registration, recognition, entry and exit events
- `logs/entries/<date>/` and `logs/exits/<date>/` - cropped images (`F####` = identified by face, `P####` = tracked without a visible face)
- `events.csv`, `visitors.csv` - the database rows (`image_path` is relative to that video's `logs/` parent)
- `db_dump.sql` - PostgreSQL dump of the schema and data (embedding vectors omitted)

Counts are honest: the tracker can still split or merge people (see README, Assumptions).
