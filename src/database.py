
from pathlib import Path

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Boolean,
    ForeignKey,
    Text,
    UniqueConstraint,
    event,
    inspect,
)
from sqlalchemy.orm import declarative_base, sessionmaker


# --------------------------------------------------
# 1. DATABASE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "campaign_insights.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(DATABASE_URL)

# Enable SQLite foreign-key enforcement.
@event.listens_for(engine, "connect")
def enable_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# --------------------------------------------------
# 2. PRODUCTS TABLE
# --------------------------------------------------

class Product(Base):
    __tablename__ = "products"

    product_id = Column(String, primary_key=True)
    product_name = Column(String, nullable=False)
    category = Column(String)
    form = Column(String)
    active_ingredient = Column(String)
    launch_date = Column(String)


# --------------------------------------------------
# 3. CAMPAIGNS TABLE
# --------------------------------------------------

class Campaign(Base):
    __tablename__ = "campaigns"

    campaign_id = Column(String, primary_key=True)

    # Original reference from CSV
    source_product_id = Column(String)

    # Validated relationship to products
    product_id = Column(
        String,
        ForeignKey("products.product_id"),
        nullable=True,
    )

    campaign_name = Column(String)
    start_date = Column(String)
    end_date = Column(String)
    budget_eur = Column(Float)
    target_age = Column(String)


# --------------------------------------------------
# 4. POSTS TABLE
# --------------------------------------------------

class Post(Base):
    __tablename__ = "posts"

    post_id = Column(String, primary_key=True)

    source_campaign_id = Column(String)

    campaign_id = Column(
        String,
        ForeignKey("campaigns.campaign_id"),
        nullable=True,
    )

    platform = Column(String)
    published_at = Column(String)
    impressions = Column(Integer, nullable=True)
    likes = Column(Integer)


# --------------------------------------------------
# 5. COMMENTS TABLE
# --------------------------------------------------

class Comment(Base):
    __tablename__ = "comments"

    comment_id = Column(String, primary_key=True)

    # Preserve the original post ID even if missing.
    source_post_id = Column(String)

    # Only a valid existing post ID is stored here.
    post_id = Column(
        String,
        ForeignKey("posts.post_id"),
        nullable=True,
    )

    author_handle = Column(String)
    created_at = Column(String)
    edited_at = Column(String, nullable=True)
    comment_text = Column(Text, nullable=True)
    likes = Column(Integer)

    # Used to detect edited comments.
    text_hash = Column(String(64))


# --------------------------------------------------
# 6. COMMENT ANALYSIS TABLE
# --------------------------------------------------

class CommentAnalysis(Base):
    __tablename__ = "comment_analysis"

    comment_id = Column(
        String,
        ForeignKey("comments.comment_id"),
        primary_key=True,
    )

    topic = Column(String)
    sentiment = Column(String)
    side_effect_flag = Column(Boolean)
    reason = Column(Text)

    # Hash of the text successfully analysed.
    analysed_hash = Column(String(64))

    # Keep track of the prompt version.
    prompt_version = Column(String)


# --------------------------------------------------
# 7. FDA REACTIONS TABLE
# --------------------------------------------------

class FDAReaction(Base):
    __tablename__ = "fda_reactions"

    reaction_id = Column(Integer, primary_key=True)

    product_id = Column(
        String,
        ForeignKey("products.product_id"),
        nullable=False,
    )

    reaction = Column(String, nullable=False)
    report_count = Column(Integer)

    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "reaction",
            name="uq_product_reaction",
        ),
    )


# --------------------------------------------------
# 8. INITIALIZE DATABASE
# --------------------------------------------------

def init_db():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()

    print(f"Database created at: {DB_PATH}")

    inspector = inspect(engine)

    print("\nDatabase tables:")
    for table_name in sorted(inspector.get_table_names()):
        print(f"  - {table_name}")
