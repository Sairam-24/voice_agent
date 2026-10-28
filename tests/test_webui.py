"""Which UI is served where (the React build, the classic page, or an explicit folder)."""

from pathlib import Path

from app.webui import resolve_web_dirs


def project(tmp_path: Path, *, react: bool, classic: bool) -> Path:
    if react:
        (tmp_path / "frontend" / "dist").mkdir(parents=True)
        (tmp_path / "frontend" / "dist" / "index.html").write_text("<title>react</title>")
    if classic:
        (tmp_path / "web").mkdir()
        (tmp_path / "web" / "index.html").write_text("<title>classic</title>")
    return tmp_path


def test_react_build_is_served_at_root_and_the_classic_page_stays_available(tmp_path):
    root = project(tmp_path, react=True, classic=True)
    assert resolve_web_dirs(None, root) == (root / "frontend" / "dist", root / "web")


def test_without_a_react_build_the_classic_page_is_served_at_root(tmp_path):
    root = project(tmp_path, react=False, classic=True)
    assert resolve_web_dirs(None, root) == (root / "web", None)


def test_a_react_folder_without_index_html_is_not_a_build(tmp_path):
    root = project(tmp_path, react=False, classic=True)
    (root / "frontend" / "dist").mkdir(parents=True)  # e.g. a half-finished build
    assert resolve_web_dirs(None, root) == (root / "web", None)


def test_nothing_to_serve_is_not_an_error(tmp_path):
    assert resolve_web_dirs(None, project(tmp_path, react=False, classic=False)) == (None, None)


def test_an_explicit_folder_is_served_alone(tmp_path):
    root = project(tmp_path, react=True, classic=True)
    custom = tmp_path / "custom"
    custom.mkdir()
    assert resolve_web_dirs(custom, root) == (custom, None)
    assert resolve_web_dirs(tmp_path / "missing", root) == (None, None)


def test_the_app_serves_an_explicit_folder_and_keeps_api_routes_first(tmp_path, make_client):
    (tmp_path / "index.html").write_text("<title>custom ui</title>")
    client = make_client(web_dir=str(tmp_path))
    assert "custom ui" in client.get("/").text
    assert client.get("/health").json()["status"] in {"ok", "starting", "degraded"}
    assert client.get("/classic/").status_code == 404
