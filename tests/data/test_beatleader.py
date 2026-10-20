from bsmapper.data.beatleader import attach_ratings, parse_ratings
from bsmapper.data.filters import BeatLeaderRating, MapRecord
from bsmapper.data.manifest import MANIFEST_NAME, ManifestEntry, read_manifest
from bsmapper.data.scraper import ScrapeConfig, backfill_ratings, ratings_coverage, scrape


def hash_of(d) -> str:
    return d["versions"][0]["hash"]


def test_parse_ratings(bl):
    data = {
        "difficulties": [
            bl(),
            bl("Lawless", "ExpertPlus", status=0, techRating=None, passRating=None),
            bl("OneSaber", "ExpertPlus", hash="OTHERHASH"),  # another version: skipped
            bl("Standard", "Hard", hash="ABC", status=99),
        ]
    }
    rs = parse_ratings(data, "abc")
    assert [(r.characteristic, r.difficulty) for r in rs] == [
        ("Standard", "Expert"),
        ("Lawless", "ExpertPlus"),
        ("Standard", "Hard"),
    ]
    assert rs[0] == BeatLeaderRating("Standard", "Expert", "ranked", 5.1, 3.6, 2.7, 7.1, 0.98)
    assert rs[1].status == "unranked" and rs[1].tech_rating is None and rs[1].stars == 5.1
    assert rs[2].status == "99"


def test_attach_ratings_matches_per_difficulty(doc):
    d = doc("abc")
    d["versions"][0]["diffs"].append(dict(d["versions"][0]["diffs"][0], difficulty="Hard"))
    rec = MapRecord.from_api(d)
    rating = BeatLeaderRating("standard", "EXPERT", "ranked", tech_rating=3.0)
    out = attach_ratings(rec, [rating])
    assert out.diffs[0].beatleader == rating
    assert out.diffs[1].beatleader is None  # no rating for Hard: null, not an error
    assert attach_ratings(rec, None).diffs[0].beatleader is None
    assert MapRecord.from_dict(out.to_dict()) == out


def test_client_caches_hits_and_misses(tmp_path, fake, doc, bl):
    fb = fake([])
    fb.bl_maps["aaa"] = [bl()]
    c = fb.bl_client(tmp_path)
    assert c.ratings("AAA")[0].tech_rating == 3.6
    assert c.ratings("aaa")[0].tech_rating == 3.6
    assert c.ratings("bbb") is None  # 404
    assert c.ratings("bbb") is None
    assert fb.bl_calls == ["aaa", "bbb"]
    assert any("bsmapper" in ua for ua in fb.user_agents)


def test_scrape_attaches_ratings(tmp_path, fake, doc, bl):
    ds = [doc("k0"), doc("k1"), doc("k2")]
    fb = fake(ds)
    fb.bl_maps[hash_of(ds[0])] = [bl(), bl("Standard", "ExpertPlus")]
    fb.bl_maps[hash_of(ds[1])] = [bl("Standard", "Expert", techRating=None)]
    # k2 is unknown to BeatLeader.
    cfg = ScrapeConfig(out_dir=tmp_path, min_free_bytes=0)
    scrape(fb.client(), cfg, ratings=fb.bl_client(tmp_path / "cache"))

    by = {e.record.key: e for e in read_manifest(tmp_path)}
    assert by["k0"].record.diffs[0].beatleader.tech_rating == 3.6
    assert by["k1"].record.diffs[0].beatleader.tech_rating is None
    assert by["k1"].record.diffs[0].beatleader.status == "ranked"
    assert by["k2"].record.diffs[0].beatleader is None
    assert all(e.beatleader_at for e in by.values())
    cov = ratings_coverage(list(by.values()))
    assert (cov.maps, cov.maps_with_ratings, cov.diffs_with_tech, cov.not_looked_up) == (
        3,
        1,
        1,
        0,
    )


def test_beatleader_outage_does_not_fail_scrape_and_is_retried(tmp_path, fake, doc, bl):
    ds = [doc("k0"), doc("k1")]
    fb = fake(ds)
    fb.bl_down = True
    cfg = ScrapeConfig(out_dir=tmp_path, min_free_bytes=0)
    res = scrape(fb.client(), cfg, ratings=fb.bl_client())
    assert res.total == 2
    assert all(e.beatleader_at is None for e in read_manifest(tmp_path))

    fb.bl_down = False
    fb.bl_maps[hash_of(ds[1])] = [bl()]
    res = scrape(fb.client(), cfg, ratings=fb.bl_client())
    assert res.downloaded == 0
    entries = read_manifest(tmp_path)
    assert all(e.beatleader_at for e in entries)
    assert entries[1].record.diffs[0].beatleader.tech_rating == 3.6
    lines = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # compacted after the backfill


def test_backfill_old_manifest_and_refresh(tmp_path, fake, doc, bl):
    ds = [doc("k0"), doc("k1")]
    fb = fake(ds)
    scrape(fb.client(), ScrapeConfig(out_dir=tmp_path, min_free_bytes=0))  # no ratings
    # A manifest line written before ratings existed has no beatleader fields at all.
    line = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()[0]
    assert ManifestEntry.from_json(line).beatleader_at is None

    fb.bl_maps[hash_of(ds[0])] = [bl()]
    assert backfill_ratings(fb.bl_client(tmp_path / "cache"), tmp_path) == 2
    assert backfill_ratings(fb.bl_client(tmp_path / "cache"), tmp_path) == 0
    assert len(fb.bl_calls) == 2

    fb.bl_maps[hash_of(ds[0])] = [bl(techRating=9.9)]
    refresh = fb.bl_client(tmp_path / "cache", cache_ttl=0.0)
    assert backfill_ratings(refresh, tmp_path, refresh=True) == 2
    assert read_manifest(tmp_path)[0].record.diffs[0].beatleader.tech_rating == 9.9
