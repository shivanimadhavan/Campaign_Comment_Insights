"""Tests for strict date normalization and rerunnable comment version logic."""
import pandas as pd
import pytest

from src.pipeline import stamp, version, compare_versions, deduplicate


@pytest.mark.parametrize('raw,expected', [
    ('2025-09-15', '2025-09-15'),
    ('01.12.2025', '2025-12-01'),
])
def test_calendar_dates(raw, expected):
    assert stamp(raw, date_only=True) == expected


@pytest.mark.parametrize('raw,expected', [
    ('2026-05-25 20:33:00', '2026-05-25T20:33:00'),
    ('02.09.2026 16:56', '2026-09-02T16:56:00'),
    ('2026-01-04T18:49:00Z', '2026-01-04T18:49:00Z'),
    ('2026-05-25T20:33:00+02:00', '2026-05-25T18:33:00Z'),
    ('2026-05-25T20:33:00-05:00', '2026-05-26T01:33:00Z'),
    ('2026-05-25T20:33:00.123456Z', '2026-05-25T20:33:00.123456Z'),
    ('09/02/2026 04:56 PM', '2026-09-02T16:56:00'),
    ('1767225600', '2026-01-01T00:00:00Z'),
])
def test_timestamps(raw, expected):
    assert stamp(raw) == expected


def test_missing_values():
    assert stamp(None) is None
    assert stamp(pd.NA) is None
    assert stamp(' ') is None


@pytest.mark.parametrize('raw', [
    '31.02.2026', '2026-02-30', '2026-13-01',
    '31.02.2026 09:00', '2026-02-30T12:00:00Z', 'yesterday',
])
def test_invalid_dates_fail(raw):
    with pytest.raises(ValueError):
        stamp(raw, date_only=(' ' not in raw and 'T' not in raw))


def test_calendar_date_does_not_accept_timestamp():
    with pytest.raises(ValueError):
        stamp('2025-09-01 13:00', date_only=True)


def test_timestamp_does_not_accept_date_only():
    with pytest.raises(ValueError):
        stamp('2025-09-01')


def test_mixed_timezone_comparison_refuses_to_guess():
    with pytest.raises(ValueError, match='ambiguous timezone'):
        compare_versions('2026-10-05T01:04:00', '2026-10-05T00:04:00Z', context='comment')


def test_offsets_compare_as_instants():
    assert compare_versions('2026-10-05T03:04:00+02:00', '2026-10-05T01:04:00Z', context='comment') == 0


def test_most_recent_edit_selected():
    df = pd.DataFrame([
        {'comment_id':'CM000276', 'edited_at':None, 'comment_text':'dry mouth'},
        {'comment_id':'CM000276', 'edited_at':'2026-10-05 01:04:00', 'comment_text':'dry mouth and nausea'},
    ], dtype='string')
    cleaned, counts = deduplicate(df, 'comments.csv', 'comment_id')
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['comment_text'] == 'dry mouth and nausea'
    assert counts['repeated_versions_resolved'] == 1


def test_ambiguous_mixed_timezone_edits_rejected():
    df = pd.DataFrame([
        {'comment_id':'C1', 'edited_at':'2026-10-05T01:04:00Z', 'comment_text':'first'},
        {'comment_id':'C1', 'edited_at':'2026-10-05T03:04:00', 'comment_text':'second'},
    ], dtype='string')
    with pytest.raises(ValueError, match='ambiguous timezones'):
        deduplicate(df, 'comments.csv', 'comment_id')


def test_version_handles_pandas_na():
    assert version(pd.NA) is None
    assert version(None) is None
    assert version('  ') is None
