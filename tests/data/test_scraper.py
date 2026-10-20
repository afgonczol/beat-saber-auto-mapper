import json

import pytest

from bsmapper.data.filters import FilterConfig
from bsmapper.data.manifest import MANIFEST_NAME, STATE_NAME, ScrapeState, read_manifest
from bsmapper.data.scraper import ScrapeConfig, load_curated_keys, scrape


def cfg(tmp_path, **kw) -> ScrapeConfig:
    kw.setdefault("min_free_bytes", 0)
    return ScrapeConfig(out_dir=tmp_path, **kw)


def docs(doc, n, start=0):
    return [doc(f"k{i:02d}") for i in range(start, start + n)]


def keys(tmp_path):
    return [e.record.key for e in read_manifest(tmp_path)]


def test_downloads_filtered_maps(tmp_path, fake, doc):
    ds = [
        doc("good1"),
        doc("ai1", automapper=True),
        doc("low1", score=0.5),
        doc("good2", ranked=True),
        doc("ne1", diff={"ne": True}),
        doc("good3", blRanked=True, curatedAt="2024-01-01Z"),
    ]
    fb = fake(ds, page_size=2)
    res = scrape(fb.client(), cfg(tmp_path, limit=10))

    assert res.stop_reason == "exhausted"
    assert res.downloaded == res.total == 3
    assert keys(tmp_path) == ["good1", "good2", "good3"]
    assert sorted((tmp_path / "zips").iterdir()) == [
        tmp_path / "zips" / f"{k}.zip" for k in ("good1", "good2", "good3")
    ]
    assert res.rejected == {"ai": 1, "score": 1, "mods": 1}
    assert res.gold == 2
    by_key = {e.record.key: e for e in read_manifest(tmp_path)}
    assert by_key["good3"].gold_reasons == ["beatleader_ranked", "beatsaver_curated"]
    assert by_key["good1"].zip_path == "zips/good1.zip"
    assert by_key["good1"].bytes == (tmp_path / "zips" / "good1.zip").stat().st_size
    assert fb.user_agents == {fb.client()._http.headers["user-agent"]}
    assert "automapper" not in fb.last_params


def test_limit_then_resume(tmp_path, fake, doc):
    fb = fake(docs(doc, 10), page_size=3)
    res = scrape(fb.client(tmp_path / "cache"), cfg(tmp_path, limit=4))
    assert res.stop_reason == "limit" and res.total == 4
    assert ScrapeState.load(tmp_path).next_page == 1  # page 1 not finished yet

    res = scrape(fb.client(tmp_path / "cache"), cfg(tmp_path, limit=8))
    assert res.downloaded == 4 and res.total == 8
    assert keys(tmp_path) == [f"k{i:02d}" for i in range(8)]
    assert len(fb.download_calls) == 8  # nothing downloaded twice
    assert fb.search_calls == [0, 1, 2]  # page 1 came from the cache on resume


def test_resume_after_interruption(tmp_path, fake, doc):
    fb = fake(docs(doc, 7), page_size=3)
    count = [0]

    def explode(h):
        count[0] += 1
        if count[0] == 5:
            raise KeyboardInterrupt

    fb.on_download = explode
    with pytest.raises(KeyboardInterrupt):
        scrape(fb.client(tmp_path / "cache"), cfg(tmp_path, limit=100))
    assert len(keys(tmp_path)) == 4
    assert ScrapeState.load(tmp_path).next_page == 1

    fb.on_download = None
    res = scrape(fb.client(tmp_path / "cache"), cfg(tmp_path, limit=100))
    assert res.downloaded == 3 and res.total == 7
    assert keys(tmp_path) == [f"k{i:02d}" for i in range(7)]
    assert not list((tmp_path / "zips").glob("*.part"))


def test_torn_manifest_line_and_leftover_zip(tmp_path, fake, doc):
    fb = fake(docs(doc, 3), page_size=3)
    scrape(fb.client(), cfg(tmp_path, limit=2))
    with (tmp_path / MANIFEST_NAME).open("a", encoding="utf-8") as f:
        f.write('{"key": "k02", "hash"')  # crash mid-write
    # The zip for k02 landed but its manifest line did not.
    (tmp_path / "zips" / "k02.zip").write_bytes((tmp_path / "zips" / "k00.zip").read_bytes())
    (tmp_path / "zips" / "junk.zip.part").write_bytes(b"partial")

    res = scrape(fb.client(), cfg(tmp_path, limit=3))
    assert res.total == 3 and keys(tmp_path) == ["k00", "k01", "k02"]
    assert fb.download_calls.count(fb.docs[2]["versions"][0]["hash"]) == 0  # reused
    assert not (tmp_path / "zips" / "junk.zip.part").exists()
    lines = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["key"] == "k02"


def test_bad_and_missing_zips_are_skipped(tmp_path, fake, doc):
    ds = docs(doc, 4)
    fb = fake(ds, page_size=4)
    fb.bad_zips.add(ds[1]["versions"][0]["hash"])
    fb.missing.add(ds[2]["versions"][0]["hash"])
    res = scrape(fb.client(max_retries=0), cfg(tmp_path, limit=10))
    assert keys(tmp_path) == ["k00", "k03"]
    assert set(res.failed) == {"k01", "k02"}
    assert not (tmp_path / "zips" / "k01.zip").exists()


def test_max_bytes_cap(tmp_path, fake, doc):
    fb = fake(docs(doc, 10), page_size=5)
    first = scrape(fb.client(), cfg(tmp_path, limit=1))
    per_zip = first.total_bytes
    res = scrape(fb.client(), cfg(tmp_path, limit=10, max_bytes=3 * per_zip))
    assert res.stop_reason == "max_bytes" and res.total == 3


def test_min_free_disk(tmp_path, fake, doc):
    fb = fake(docs(doc, 3))
    res = scrape(fb.client(), ScrapeConfig(out_dir=tmp_path, min_free_bytes=1 << 62))
    assert res.stop_reason == "min_free_disk" and res.total == 0


def test_changed_filters_restart_cursor(tmp_path, fake, doc):
    fb = fake(docs(doc, 6), page_size=2)
    scrape(fb.client(), cfg(tmp_path, limit=5))
    assert ScrapeState.load(tmp_path).next_page == 2

    res = scrape(fb.client(), cfg(tmp_path, limit=6, filters=FilterConfig(min_votes=50)))
    assert fb.search_calls[-3:] == [0, 1, 2]
    assert res.downloaded == 1 and len(fb.download_calls) == 6


def test_max_pages(tmp_path, fake, doc):
    fb = fake(docs(doc, 10), page_size=2)
    res = scrape(fb.client(), cfg(tmp_path, max_pages=2))
    assert res.stop_reason == "max_pages" and res.total == 4
    assert json.loads((tmp_path / STATE_NAME).read_text())["next_page"] == 2


def test_curated_list(tmp_path, fake, doc):
    gold = tmp_path / "gold.txt"
    gold.write_text("# hand-picked\nK01\n  k02  # nice\n\n", encoding="utf-8")
    curated = load_curated_keys(gold)
    assert curated == {"k01", "k02"}
    fb = fake(docs(doc, 3))
    scrape(fb.client(), cfg(tmp_path / "out", curated_keys=curated))
    gold_keys = [e.record.key for e in read_manifest(tmp_path / "out") if e.gold]
    assert gold_keys == ["k01", "k02"]
