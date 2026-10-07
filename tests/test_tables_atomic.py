from concurrent.futures import ThreadPoolExecutor

import numpy as np

from stainid.tables import save_npz_atomic, write_text_atomic


def test_write_text_atomic_never_exposes_partial_text(tmp_path):
    path = tmp_path / "provenance.json"
    text = "x" * 200_000
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: write_text_atomic(path, text), range(8)))
    assert path.read_text() == text and not list(tmp_path.glob(".*.tmp"))


def test_stale_neun_claims_are_taken_over(tmp_path):
    import os

    from stainid.stains.neun.pipeline import claim_field

    claim = tmp_path / "T.claim"
    assert claim_field(claim) and claim.read_text() == str(os.getpid())
    assert not claim_field(claim)
    claim.write_text("")
    assert claim_field(claim)
    claim.write_text("999999999")
    assert claim_field(claim)


def test_save_npz_atomic_leaves_only_the_finished_file(tmp_path):
    path = tmp_path / "tile.npz"
    save_npz_atomic(path, masks=np.arange(6).reshape(2, 3))
    assert [p.name for p in tmp_path.iterdir()] == ["tile.npz"]
    assert np.load(path)["masks"].tolist() == [[0, 1, 2], [3, 4, 5]]
