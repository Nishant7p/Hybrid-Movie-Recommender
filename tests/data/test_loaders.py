import pandas as pd
from cf_pipeline.data.loaders import load_ml1m_ratings, load_ml1m_movies

def test_load_ml1m_ratings(tmp_path):
    raw = tmp_path / "ml-1m"
    raw.mkdir()
    (raw / "ratings.dat").write_text("1::100::5::978300760\n1::200::3::978300761\n2::100::4::978300762\n")
    df = load_ml1m_ratings(raw)
    assert list(df.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert len(df) == 3
    assert df.iloc[0]["user_id"] == 1
    assert df.iloc[0]["rating"] == 5

def test_load_ml1m_movies(tmp_path):
    raw = tmp_path / "ml-1m"
    raw.mkdir()
    (raw / "movies.dat").write_bytes(
        "1::Toy Story (1995)::Animation|Children's|Comedy\n2::Jumanji (1995)::Adventure|Children's|Fantasy\n".encode("latin-1")
    )
    df = load_ml1m_movies(raw)
    assert list(df.columns) == ["item_id", "title", "genres"]
    assert df.iloc[0]["title"] == "Toy Story (1995)"
