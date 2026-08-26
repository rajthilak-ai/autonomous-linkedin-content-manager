# Autonomous LinkedIn Content Manager

Five CrewAI agents research, write, critique, optimize, and schedule a LinkedIn post. The Streamlit dashboard is the main interface.

## CI/CD pipeline

The repository uses a two-part pipeline:

1. **Continuous integration (GitHub Actions):** every pull request and push to
   `master` installs the Python 3.12 dependencies, checks dependency
   compatibility and Python syntax, runs automated tests, and confirms that the
   Streamlit server starts successfully.
2. **Continuous deployment (Streamlit Community Cloud):** after a change is
   merged or pushed to `master`, Streamlit detects the GitHub update and
   automatically rebuilds the deployed app.

The CI workflow does not use `OPENAI_API_KEY` or `SERPER_API_KEY`, so automated
checks cannot consume API quota or generate content. Keep production keys only
in Streamlit Community Cloud **Secrets**, never in the repository or workflow.

You can also run the same core checks locally:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pip check
python -m compileall -q app_config.py linkedin_content_manager.py streamlit_app.py tests
python -m pytest -q
```

## Deploy from GitHub (Streamlit Community Cloud)

This app cannot run on GitHub Pages. Deploy it from this repo with Streamlit Community Cloud:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=rajthilak-ai/autonomous-linkedin-content-manager&branch=master&mainModule=streamlit_app.py)

1. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **Create app** → **Deploy from GitHub**.
3. Select:
   - Repository: `rajthilak-ai/autonomous-linkedin-content-manager`
   - Branch: `master`
   - Main file: `streamlit_app.py`
4. Open **Advanced settings** and set **Python version to 3.12** (required — 3.13/3.14 breaks CrewAI).
5. In **Advanced settings** → **Secrets**, paste:

```toml
OPENAI_API_KEY = "your_groq_api_key_here"
SERPER_API_KEY = "your_serper_api_key_here"
```

6. Click **Deploy**. Future pushes to `master` automatically rebuild the app.

If the app was already deployed on Python 3.14, reboot will not fix it. Delete the app and redeploy with Python 3.12 selected. Streamlit Cloud ignores `.python-version` / `runtime.txt`.

Keep the repo **public** for the free Streamlit Cloud plan, or use a Streamlit Cloud plan that supports private repos.

## Local run

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run streamlit_app.py
```

The CLI still works:

```bash
python linkedin_content_manager.py --topic "multi-agent AI systems"
```
