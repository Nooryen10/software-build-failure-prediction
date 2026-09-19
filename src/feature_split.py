"""
Defines the pre-execution vs. post-execution feature split for the
Intelligent Software Build Failure Prediction project.

Pre-execution: available before a CI build runs.
Post-execution ("leaky"): only available after/during the build (test results, duration, etc.)
"""

PRE_EXECUTION_FEATURES = [
    'git_branch', 'git_prev_commit_resolution_status',
    'gh_is_pr', 'gh_pull_req_num', 'gh_lang', 'gh_team_size',
    'gh_num_issue_comments', 'gh_num_commit_comments', 'gh_num_pr_comments',
    'git_diff_src_churn', 'gh_diff_files_added', 'gh_diff_files_deleted',
    'gh_diff_files_modified', 'gh_diff_src_files',
    'gh_diff_doc_files', 'gh_diff_other_files', 'gh_num_commits_on_files_touched', 'gh_sloc',
    'gh_test_lines_per_kloc', 'gh_test_cases_per_kloc', 'gh_asserts_cases_per_kloc',
    'gh_by_core_team_member', 'gh_description_complexity', 'gh_repo_age',
    'gh_repo_num_commits', 'tr_build_number', 'gh_build_started_at'
]

POST_EXECUTION_FEATURES = [
    'tr_log_status', 'tr_log_setup_time', 'tr_log_analyzer', 'tr_log_frameworks',
    'tr_log_bool_tests_ran', 'tr_log_bool_tests_failed', 'tr_log_num_tests_ok',
    'tr_log_num_tests_failed', 'tr_log_num_tests_run', 'tr_log_num_tests_skipped',
    'tr_log_num_test_suites_run', 'tr_log_num_test_suites_ok', 'tr_log_num_test_suites_failed',
    'tr_log_tests_failed', 'tr_log_testduration', 'tr_log_buildduration', 'tr_duration', 'tr_log_lan'
]

DROP_COLUMNS = [
    'tr_build_id', 'gh_project_name', 'git_merged_with', 'tr_virtual_merged_into',
    'tr_job_id', 'tr_original_commit', 'tr_jobs'
]

ZERO_VARIANCE_COLUMNS = [
    'git_diff_test_churn', 'gh_diff_tests_added', 'gh_diff_tests_deleted'
]
