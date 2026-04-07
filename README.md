# Top-bottom

This repository downloads OMIE day-ahead ES/PT prices, calculates:

- Top-Bottom spreads for 2h / 4h / 6h products
- Average daily market price
- Solar capture price

and stores the outputs locally in the `data/` folder as JSON files.

## Files

- `top_bottom.py`: main script
- `requirements.txt`: Python dependencies
- `.github/workflows/update_top_bottom.yml`: daily GitHub Actions workflow
- `data/`: output JSON files

## Local usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run for D-1:

```bash
python top_bottom.py --yesterday
```

Run for one date:

```bash
python top_bottom.py --date 20260406
```

Run a backfill:

```bash
python top_bottom.py --start 20260131 --end 20260407
```

## Notes

- Outputs are saved in `data/` relative to the repository root.
- The script no longer uses Google Drive or Colab paths.
- The workflow runs daily and updates D-1 automatically using Europe/Madrid time.
