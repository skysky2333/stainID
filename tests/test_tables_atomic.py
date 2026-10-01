from concurrent.futures import ThreadPoolExecutor

from stainid.tables import write_text_atomic


def test_write_text_atomic_never_exposes_partial_text(tmp_path):
    path = tmp_path / "provenance.json"
    text = "x" * 200_000
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: write_text_atomic(path, text), range(8)))
    assert path.read_text() == text and not list(tmp_path.glob(".*.tmp"))
