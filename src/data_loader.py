"""
Chunked loading and build-level aggregation of the TravisTorrent dataset.
"""

import pandas as pd


def load_and_filter_raw(csv_path, language='ruby', chunksize=50_000):
    """
    Load the raw TravisTorrent CSV.gz in chunks, filtering to a single language
    and resolved build outcomes (passed/failed) to keep memory usage manageable.
    """
    chunks = []
    for chunk in pd.read_csv(csv_path, compression='gzip', chunksize=chunksize, low_memory=False):
        filtered = chunk[
            (chunk['gh_lang'] == language) &
            (chunk['tr_status'].isin(['passed', 'failed']))
        ]
        chunks.append(filtered)
    return pd.concat(chunks, ignore_index=True)


def aggregate_to_build_level(df):
    """
    Collapse job-level rows (one row per Travis CI job) into build-level rows
    (one row per tr_build_id), since a single build can trigger multiple jobs.
    Pre-execution metadata is identical across jobs of the same build (take 'first'),
    while test-log fields are aggregated (summed / max) across jobs.
    """
    log_agg_cols = [
        'tr_log_num_tests_failed', 'tr_log_num_tests_ok', 'tr_log_num_tests_run',
        'tr_log_bool_tests_failed', 'tr_log_testduration', 'tr_log_buildduration'
    ]
    agg_dict = {col: 'first' for col in df.columns if col not in log_agg_cols}
    agg_dict.update({
        'tr_log_num_tests_failed': 'sum',
        'tr_log_num_tests_ok': 'sum',
        'tr_log_num_tests_run': 'sum',
        'tr_log_bool_tests_failed': 'max',
        'tr_log_testduration': 'sum',
        'tr_log_buildduration': 'sum',
    })
    return df.groupby('tr_build_id', as_index=False).agg(agg_dict)
