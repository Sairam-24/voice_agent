"""Which browser UI the server serves.

Two clients ship with the project and speak the same protocol:
  * frontend/  the React HUD. It needs `npm run build` once; the result is frontend/dist.
  * web/       the classic page. Plain HTML/JS, no build step, always works.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_web_dirs(web_dir: Path | None, root: Path = ROOT) -> tuple[Path | None, Path | None]:
    """Return (directory served at /, directory served at /classic/). Either may be None.

    An explicit WEB_DIR is served exactly as given and nothing else is added. Without one, the React
    build is served at / when it exists, with the classic page kept at /classic/ for comparison and
    as a fallback; if there is no build, the classic page is served at /.
    """
    if web_dir is not None:
        return (web_dir if web_dir.is_dir() else None), None
    react = root / "frontend" / "dist"
    classic = root / "web"
    classic_dir = classic if classic.is_dir() else None
    if (react / "index.html").is_file():
        return react, classic_dir
    return classic_dir, None
