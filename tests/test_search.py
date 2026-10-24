"""Thread search tests — ?search= on /api/threads and memory.search_threads."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-search-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "search.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

from fastapi.testclient import TestClient  # noqa: E402

from app import memory  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def _seed():
    t1 = memory.create_thread("gutter genie quote")
    memory.add_message(t1, "user", "price a downspout reroute in steinbach")
    t2 = memory.create_thread("quonset financing")
    memory.add_message(t2, "user", "what lenders fund steel buildings")
    t3 = memory.create_thread("weekend plans")
    memory.add_message(t3, "user", "gutter cleaning saturday morning")
    return t1, t2, t3


def test_search_matches_title():
    _seed()
    ids = [t["id"] for t in memory.search_threads("quonset")]
    assert len(ids) == 1


def test_search_matches_message_content():
    # "gutter" appears in t1's message AND t3's message; t1's title too.
    ids = {t["id"] for t in memory.search_threads("gutter")}
    assert len(ids) == 2


def test_search_no_match():
    assert memory.search_threads("zzz-no-such-thing") == []


def test_search_empty_query_falls_back_to_list():
    _seed()
    assert len(memory.search_threads("")) >= 3
    assert len(memory.search_threads("   ")) >= 3


def test_search_endpoint():
    _seed()
    r = client.get("/api/threads?search=steel")
    assert r.status_code == 200
    threads = r.json()["threads"]
    assert threads  # every seed adds a "steel buildings" thread
    assert {t["title"] for t in threads} == {"quonset financing"}


def test_search_endpoint_empty_means_all():
    _seed()
    r = client.get("/api/threads")
    assert r.status_code == 200
    assert len(r.json()["threads"]) >= 3
