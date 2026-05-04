# chess_opening_recommendation_system
What openings should you play?

## Full pipeline
Run the entire pipeline (extract → preprocess → train):

```bash
bash run_pipeline.sh [path/to/lichess.pgn]
```

It expects a virtual environment in `.venv/` or `venv/`. If `games.csv` or `games_encoded.csv` already exist, new data is appended; delete those files to rebuild from scratch.
