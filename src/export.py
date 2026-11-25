"""Export only verified, current classifications in the exact required schema."""
import csv
from pathlib import Path
from sqlalchemy import select
from src.database import SessionLocal, Comment, CommentAnalysis

BASE_DIR = Path(__file__).resolve().parent.parent
COLUMNS = ('comment_id','topic','sentiment','side_effect_flag','reason')


def export_classifications(path=None):
    target = Path(path) if path else BASE_DIR / 'output' / 'classified_comments.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    with SessionLocal() as session:
        rows = session.execute(select(Comment,CommentAnalysis).outerjoin(
            CommentAnalysis,Comment.comment_id == CommentAnalysis.comment_id
        ).order_by(Comment.comment_id)).all()
        verified=[]
        pending=0
        blank=0
        for comment,analysis in rows:
            if not comment.comment_text:
                blank += 1
                continue
            if (analysis is None or analysis.analysed_hash != comment.text_hash or
                    not analysis.topic or not analysis.sentiment or not analysis.reason):
                pending += 1
                continue
            verified.append({
                'comment_id':comment.comment_id,
                'topic':analysis.topic,
                'sentiment':analysis.sentiment,
                'side_effect_flag':str(bool(analysis.side_effect_flag)).lower(),
                'reason':analysis.reason,
            })
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(verified)
    result={'exported':len(verified),'pending_classification':pending,'blank_text':blank,'path':str(target)}
    print('Classification export:',result)
    if pending or blank:
        print('WARNING: CSV is not a complete classification of all stored comments; unresolved rows are reported above.')
    return result
