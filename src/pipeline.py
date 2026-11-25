"""Validated, rerunnable ingestion of WEFRA LIFE delivery_N CSV folders.

This stage deliberately does not call Gemini or openFDA. Original CSVs are read-only.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from src.database import SessionLocal, init_db, Product, Campaign, Post, Comment

BASE_DIR = Path(__file__).resolve().parent.parent
FILES = {
    'products.csv': (Product, 'product_id', ('product_id','product_name','category','form','active_ingredient','launch_date')),
    'campaigns.csv': (Campaign, 'campaign_id', ('campaign_id','product_id','campaign_name','start_date','end_date','budget_eur','target_age')),
    'posts.csv': (Post, 'post_id', ('post_id','campaign_id','platform','published_at','impressions','likes')),
    'comments.csv': (Comment, 'comment_id', ('comment_id','post_id','author_handle','created_at','edited_at','comment_text','likes')),
}


def clean(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


def number(value, *, integer=False, nonnegative=True):
    s = clean(value)
    if s is None:
        return None
    s = s.replace('€', '').replace('\u00a0', '').replace(' ', '')
    # Dot followed by precisely three digits is a European thousands separator.
    if re.fullmatch(r'[+-]?\d{1,3}(?:\.\d{3})+', s):
        s = s.replace('.', '')
    elif re.fullmatch(r'[+-]?\d{1,3}(?:\.\d{3})+,\d+', s):
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s and '.' not in s:
        s = s.replace(',', '.')
    try:
        v = float(s)
    except ValueError as exc:
        raise ValueError(f'Not a numeric value: {value!r}') from exc
    if not (float('-inf') < v < float('inf')):
        raise ValueError(f'Non-finite numeric value: {value!r}')
    if nonnegative and v < 0:
        raise ValueError(f'Negative numeric value: {value!r}')
    if integer:
        if not v.is_integer():
            raise ValueError(f'Expected whole number: {value!r}')
        return int(v)
    return v


def stamp(value, *, date_only=False):
    """Validate and normalize source dates without inventing a timezone.

    Calendar dates -> YYYY-MM-DD.
    Timestamps without timezone -> YYYY-MM-DDTHH:MM:SS (naive).
    Timestamps with Z or an explicit offset -> UTC, ending in Z.

    Also accepts supported German/US timestamp forms and 10-digit Unix seconds.
    Unknown or invalid formats raise ValueError; blanks become None.
    """
    s = clean(value)
    if s is None:
        return None

    if date_only:
        for fmt in ('%Y-%m-%d', '%d.%m.%Y'):
            try:
                # Avoid accepting a timestamp in a calendar-date column.
                return datetime.strptime(s, fmt).date().isoformat()
            except ValueError:
                pass
        raise ValueError(f'Invalid or unsupported calendar date: {value!r}')

    if re.fullmatch(r'\d{10}', s):
        try:
            dt = datetime.fromtimestamp(int(s), tz=timezone.utc)
        except (ValueError, OverflowError, OSError) as exc:
            raise ValueError(f'Invalid Unix timestamp: {value!r}') from exc
    else:
        dt = None
        # fromisoformat supports ISO dates/times with seconds, fractional seconds,
        # Z, and numeric UTC offsets such as +02:00 or -0530.
        if re.match(r'^\d{4}-\d{2}-\d{2}[T ]', s):
            try:
                dt = datetime.fromisoformat(s)
            except ValueError:
                pass
        if dt is None:
            for fmt in (
                '%d.%m.%Y %H:%M:%S',
                '%d.%m.%Y %H:%M',
                '%m/%d/%Y %I:%M:%S %p',
                '%m/%d/%Y %I:%M %p',
            ):
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    pass
        if dt is None:
            raise ValueError(f'Invalid or unsupported timestamp: {value!r}')

    if dt.tzinfo is not None and dt.utcoffset() is not None:
        dt = dt.astimezone(timezone.utc)
        suffix = 'Z'
    else:
        suffix = ''  # Do not assume unmarked timestamps are UTC or Berlin time.
    timespec = 'microseconds' if dt.microsecond else 'seconds'
    return dt.replace(tzinfo=None).isoformat(timespec=timespec) + suffix


def text_hash(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest() if s else None


def version(value):
    """Parse both new ISO timestamps and prior database timestamp formatting."""
    if clean(value) is None:
        return None
    normalized = stamp(value)
    return datetime.fromisoformat(normalized)


def compare_versions(a, b, *, context):
    """Compare known timestamps, refusing to guess between naive and aware.

    Returns -1, 0 or 1. Both arguments must be present.
    """
    left, right = version(a), version(b)
    if left is None or right is None:
        raise ValueError(f'{context}: cannot compare missing timestamps')
    if (left.utcoffset() is None) != (right.utcoffset() is None):
        raise ValueError(
            f'{context}: ambiguous timezone (one timestamp has an offset '
            'and the other does not); cannot determine which is later'
        )
    return (left > right) - (left < right)


def sortable_version(value):
    """Used only after validating versions of one ID are comparable."""
    dt = version(value)
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt and dt.tzinfo else (dt or datetime.min)


def deduplicate(df, file, key):
    """Deduplicate within a file; edited_at must disambiguate changed comment text."""
    original = len(df)
    df = df.drop_duplicates().copy()
    exact = original - len(df)
    # Primary keys are mandatory; no silent dropping.
    if df[key].map(clean).isna().any():
        raise ValueError(f'{file}: missing/blank primary key {key}')
    df[key] = df[key].map(clean)
    df['_row_order'] = range(len(df))
    if file == 'comments.csv':
        df['_edited_normalized'] = df['edited_at'].map(stamp)
        for ident, group in df.groupby(key, sort=False):
            for t, same in group.groupby('_edited_normalized', dropna=False):
                # Metadata-only duplicates with same timestamp cannot be ordered reliably either.
                if len(same.drop(columns=['_row_order','_edited_normalized']).drop_duplicates()) > 1:
                    raise ValueError(f'{file}: ambiguous versions for {ident}, edited_at={t}')
        for ident, group in df.groupby(key, sort=False):
            populated = group['_edited_normalized'].dropna().tolist()
            if populated and any(
                (version(t).utcoffset() is None) != (version(populated[0]).utcoffset() is None)
                for t in populated[1:]
            ):
                raise ValueError(f'{file}: ambiguous timezones for edited versions of {ident}')
        df['_version_sort'] = df['_edited_normalized'].map(sortable_version)
        df = df.sort_values(['_version_sort', '_row_order'], kind='stable').drop_duplicates(key, keep='last')
        df = df.drop(columns=['_row_order','_edited_normalized','_version_sort'])
    else:
        repeated = df[df.duplicated(key, keep=False)]
        if not repeated.empty:
            raise ValueError(f'{file}: conflicting records for IDs {repeated[key].unique()[:5].tolist()}')
        df = df.drop(columns='_row_order')
    return df, {'source_rows':original,'exact_duplicates_removed':exact,'repeated_versions_resolved':original-exact-len(df),'unique_rows':len(df)}


def upsert(session, model, ident, values, counters):
    old = session.get(model, ident)
    if old is None:
        session.add(model(**values))
        counters['inserted'] += 1
        return
    if model is Comment:
        old_v, new_v = version(old.edited_at), version(values['edited_at'])
        if old_v and new_v:
            chronological_order = compare_versions(
                values['edited_at'], old.edited_at, context=f'comment {ident}'
            )
        else:
            chronological_order = None
        if old_v and (new_v is None or chronological_order < 0):
            counters['skipped_older'] += 1
            return
        if old_v == new_v and old.text_hash != values['text_hash']:
            # Without a newer edited_at, order cannot be established safely.
            raise ValueError(f'Ambiguous changed comment {ident}: same edited_at, different text')
    changed = False
    for field, value in values.items():
        if getattr(old, field) != value:
            setattr(old, field, value)
            changed = True
    counters['updated' if changed else 'unchanged'] += 1


def process_delivery(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Delivery not found: {folder}')
    init_db()
    report = {'delivery':folder.name,'files':{},'actions':{'inserted':0,'updated':0,'unchanged':0,'skipped_older':0},'relationships':{}}
    with SessionLocal.begin() as session:
        for filename, (model, key, required) in FILES.items():
            path = folder / filename
            if not path.is_file():
                report['files'][filename] = {'status':'not_supplied'}
                continue
            df = pd.read_csv(path, dtype='string', encoding='utf-8-sig')
            missing = sorted(set(required) - set(df.columns))
            if missing:
                raise ValueError(f'{filename}: missing required columns: {missing}')
            info = {'status':'processed','blank_values':{col:int(df[col].map(clean).isna().sum()) for col in required}}
            df, dedup = deduplicate(df, filename, key)
            info.update(dedup)
            report['files'][filename] = info
            for row in df.to_dict('records'):
                ident = clean(row[key])
                try:
                    if model is Product:
                        values = {k:clean(row[k]) for k in ('product_id','product_name','category','form','active_ingredient')}
                        if not values['product_name']:
                            raise ValueError('missing product_name')
                        values['launch_date'] = stamp(row['launch_date'], date_only=True)
                    elif model is Campaign:
                        pid = clean(row['product_id'])
                        # Explicit, auditable correction: P1 is a zero-padding variant of P01.
                        # Apply only when the canonical product exists and raw ID does not.
                        resolved_pid = pid
                        if pid and session.get(Product, pid) is None and re.fullmatch(r'P[1-9]\d*', pid):
                            candidate = 'P' + pid[1:].zfill(2)
                            if session.get(Product, candidate):
                                resolved_pid = candidate
                                report.setdefault('normalizations', []).append({
                                    'file':filename,'record':ident,'field':'product_id',
                                    'original':pid,'resolved':candidate,'reason':'verified zero-padding'
                                })
                        values = dict(campaign_id=ident,source_product_id=pid,
                                      product_id=resolved_pid if resolved_pid and session.get(Product,resolved_pid) else None,
                                      campaign_name=clean(row['campaign_name']),start_date=stamp(row['start_date'], date_only=True),
                                      end_date=stamp(row['end_date'], date_only=True),budget_eur=number(row['budget_eur']),
                                      target_age=clean(row['target_age']))
                        if values['start_date'] and values['end_date'] and values['end_date'] < values['start_date']:
                            raise ValueError('end_date earlier than start_date')
                    elif model is Post:
                        cid = clean(row['campaign_id'])
                        platform = clean(row['platform'])
                        aliases = {'insta':'Instagram','instagram':'Instagram','facebook':'Facebook','tiktok':'TikTok','youtube':'YouTube'}
                        values = dict(post_id=ident,source_campaign_id=cid,
                                      campaign_id=cid if cid and session.get(Campaign,cid) else None,
                                      platform=aliases.get(platform.lower(),platform) if platform else None,
                                      published_at=stamp(row['published_at']),impressions=number(row['impressions'],integer=True),
                                      likes=number(row['likes'],integer=True))
                    else:
                        pid = clean(row['post_id'])
                        txt = clean(row['comment_text'])
                        values = dict(comment_id=ident,source_post_id=pid,
                                      post_id=pid if pid and session.get(Post,pid) else None,
                                      author_handle=clean(row['author_handle']),
                                      created_at=stamp(row['created_at']),edited_at=stamp(row['edited_at']),
                                      comment_text=txt,likes=number(row['likes'],integer=True),text_hash=text_hash(txt))
                        if values['created_at'] and values['edited_at']:
                            edited_dt = version(values['edited_at'])
                            created_dt = version(values['created_at'])
                            if (edited_dt.utcoffset() is None) != (created_dt.utcoffset() is None):
                                # The export does not supply a timezone for one date.
                                # Keep the comment, but do not pretend they can be ordered.
                                report.setdefault('date_warnings', []).append({
                                    'file': filename, 'record': ident,
                                    'issue': 'created_at and edited_at have mixed timezone information; chronological validation skipped',
                                    'created_at': values['created_at'],
                                    'edited_at': values['edited_at'],
                                })
                            elif edited_dt < created_dt:
                                raise ValueError('edited_at earlier than created_at')
                    upsert(session,model,ident,values,report['actions'])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{filename} record {ident}: {exc}') from exc
            session.flush()
        # Repair unresolved parent references, including when later deliveries provide a missing parent.
        for row in session.scalars(select(Campaign).where(Campaign.product_id.is_(None))):
            if row.source_product_id:
                raw = row.source_product_id
                canonical = ('P' + raw[1:].zfill(2)) if re.fullmatch(r'P[1-9]\d*', raw) else raw
                if session.get(Product,raw):
                    row.product_id=raw
                elif session.get(Product,canonical):
                    row.product_id=canonical
        for row in session.scalars(select(Post).where(Post.campaign_id.is_(None))):
            if row.source_campaign_id and session.get(Campaign,row.source_campaign_id):
                row.campaign_id = row.source_campaign_id
        for row in session.scalars(select(Comment).where(Comment.post_id.is_(None))):
            if row.source_post_id and session.get(Post,row.source_post_id):
                row.post_id = row.source_post_id
        session.flush()
        report['relationships'] = {
            'campaigns_missing_product':sum(1 for x in session.scalars(select(Campaign)) if x.product_id is None),
            'posts_missing_campaign':sum(1 for x in session.scalars(select(Post)) if x.campaign_id is None),
            'comments_missing_post':sum(1 for x in session.scalars(select(Comment)) if x.post_id is None),
            'comments_blank_text':sum(1 for x in session.scalars(select(Comment)) if not x.comment_text),
            'sqlite_foreign_key_violations':len(session.connection().exec_driver_sql('PRAGMA foreign_key_check').fetchall()),
        }
    # Write only after the transaction succeeds. No change to source data.
    report_folder = BASE_DIR / 'output' / 'ingestion_reports'
    report_folder.mkdir(parents=True,exist_ok=True)
    report_path = report_folder / f'{folder.name}.json'
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print(f'Validation report: {report_path}')
    return report
