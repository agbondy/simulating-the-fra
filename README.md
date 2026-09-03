# Simulating the FRA

Redistricting-ensemble simulations comparing ranked-choice-voting/multi-member districts (RCV/MMD, as proposed under the Fair Representation Act) against single-member-district plurality (SMD) across 8 states: MA, CA, NY, IL, MN, AZ, TX, UT.

## Structure

- `election_sim_partisan.py` — shared simulation logic for the partisan (Dem/Rep) model: precinct data merging, GerryChain ReCom ensemble generation, ballot simulation (Plackett-Luce), RCV/plurality counting, competitiveness scoring, and plotting.
- `election_sim_racial.py` — shared simulation logic for the racial model (white/asian/latino/black/native), structured in parallel to the partisan module but kept independent (some duplication by design, not a shared common file).
- `{STATE}.ipynb` — one notebook per state. Each imports both shared modules (aliased `esp`/`esr` to avoid name collisions between the two), sets that state's configuration (district/seat counts, in-group loyalty `pi`, status-quo seat shares, etc.), and runs both the partisan and racial simulations end to end.

## Data

Notebooks expect precinct-level election shapefiles, Census block shapefiles, and PL 94-171 demographic summary files, referenced via the `FRA_DATA_DIR` environment variable:

```bash
export FRA_DATA_DIR=/path/to/your/data
```

Defaults to `./data` if unset. Expected layout: `{FRA_DATA_DIR}/{STATE}/...` — see each notebook's config cell for the exact filenames expected per state. Data files themselves are not included in this repo.

## Competitiveness score

`C = min(pk, 1-pk)`, `pk = t_dist.cdf(M/sigma, df=2)`, `M = vote_share - 0.5`. `sigma` defaults to an ensemble-derived per-district standard deviation (redistricting/map-drawing variability); pass a fixed value instead via each simulation function's `sigma=` argument to use a different methodology (e.g. a temporal/electoral-swing-based sigma).
