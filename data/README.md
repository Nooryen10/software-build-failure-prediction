# Dataset Documentation

## Source
TravisTorrent dataset (final-2017-01-25.csv.gz), obtained from https://testroots.github.io/travistorrent-site
Citation: Beller, M., Gousios, G., & Zaidman, A. (2017). TravisTorrent: Synthesizing Travis CI and GitHub
for Full-Stack Research on Continuous Integration. MSR 2017.

## Filtering Steps
1. Loaded the raw CSV.gz in chunks (chunksize=50,000) to handle the 2.64M-row original dataset on limited hardware.
2. Filtered to `gh_lang == 'ruby'` (898 of 1,300 projects in TravisTorrent are Ruby).
3. Filtered to `tr_status` in ['passed', 'failed'], dropping 'errored' and 'canceled' builds (ambiguous outcomes).
4. Aggregated from job-level rows to build-level rows (grouped by `tr_build_id`), since the raw dataset
   contains one row per CI job, with multiple jobs per build.

## Resulting Dataset Sizes
- Raw filtered (job-level): 1,542,248 rows × 66 columns
- Aggregated (build-level): 252,151 rows × 66 columns
- Class distribution: 197,850 passed (78.5%) / 54,301 failed (21.5%)

## Feature Split
- `df_clean.csv`: 27 pre-execution features + target (features available before a build runs)
- `df_leaky.csv`: pre-execution + post-execution features + target (includes test results, build duration —
  used only for the leaky baseline comparison)

See `src/feature_split.py` for the exact column lists.
