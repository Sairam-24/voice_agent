# GitHub publication checklist

Work through this before the first push. No public repository, URL, badge or screenshot exists unless you create it. Use a private repository first if you want to look it over before anyone else can.

## 1. Decisions only you can make

- [x] **Licence for your code.** Decided: GPL-3.0-or-later, with the unmodified text in `LICENSE`. `piper-tts` (imported by `app/services/tts.py`) is GPL-3.0-or-later, so this is the simplest licence consistent with it. Not legal advice. The README's Licensing section explains it and lists the third-party licences.
- [ ] **Default voice.** The default is `en_US-ljspeech-medium` (LJ Speech, public domain per its model card). The popular `en_US-lessac-medium` voice is trained on a dataset whose licence is *research only*; do not switch to it for anything public without reading <https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html>.
- [ ] **Public demo or not.** If yes, read `docs/deployment.md` first. A public instance with no authentication can burn your CPU and API quota.

## 2. Make sure no secrets or large files go in

- [ ] `.env` is listed in `.gitignore` (it is) and `git status` does not show it.
- [ ] Search the history-to-be for keys before the first commit:
  ```
  git init
  git add -A
  git status                 # check what is staged: no .env, no models/, no .venv/
  git grep -n -i -E "api[_-]?key|secret|token" -- ':!*.md' ':!.env.example' ':!tests' ':!app/config.py'
  ```
- [ ] `models/` (downloaded weights, about 200 MB), `.venv/`, `benchmarks/inputs/` (generated audio) are ignored (they are).
- [ ] If you ever pasted a real `GEMINI_API_KEY` anywhere other than `.env`, rotate it.

## 3. Verify it works from a clean checkout

On another folder or machine:

- [ ] Follow the README setup commands verbatim; every command should work without edits.
- [ ] `python -m pytest -m "not integration"` passes.
- [ ] `python scripts/download_models.py`, then `python -m pytest -m integration` passes (the Gemini test skips without a key).
- [ ] `node --test "web/test/*.test.mjs"` passes.
- [ ] Start the server, open the page, record, hear the reply.
- [ ] If you changed any default (model, voice, prompt), re-run the benchmark and update the README table. Numbers in the README must come from files in `benchmarks/results/`.

## 4. Repository hygiene

- [ ] Review the README's "Known limitations" and "Not yet verified" lists and keep them true.
- [ ] Decide whether to commit `benchmarks/results/*.json|csv` (they are raw evidence for the README numbers; they contain your CPU model and OS version, and per-request transcripts of synthetic prompts only, no personal audio).
- [ ] Add a short description and topics (`fastapi`, `faster-whisper`, `piper-tts`, `websocket`, `voice-assistant`) once the repo exists.
- [ ] Only add a screenshot or demo GIF you have captured yourself from the running app.
- [ ] Do not add badges, deployment links or claims you cannot back up. In particular the CI badge only makes sense once a workflow exists and has passed.

## 5. Optional: continuous integration

A minimal workflow would run the unit tests and the JS tests on Linux (`pip install -r requirements-dev.txt`, `python -m pytest -m "not integration"`, `node --test "web/test/*.test.mjs"`). Those tests do not need model files. It has **not** been written or run, and the unit tests have only been run on Windows, so expect to fix small things on the first Linux run.

## 6. First commit

```
git init
git add -A
git commit -m "Initial commit: low-latency voice agent"
git branch -M main
git remote add origin <the URL GitHub shows you after creating the repository>
git push -u origin main
```
