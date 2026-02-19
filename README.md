# YouTube Comment Analysis - Agent-Style Insights

This project helps creators analyze YouTube comments from either a full channel or a single concept playlist.  
You collect comments in a selected date range, then ask natural-language questions and get structured insights with execution traces.

## What This Project Does

- Collects comments from:
  - a channel upload stream (`Channel` scope)
  - a specific playlist (`Playlist / Concept` scope)
- Runs sentiment analysis on comments (`POSITIVE`, `NEUTRAL`, `NEGATIVE`)
- Supports question-based analysis with an agent-style execution flow:
  - validate dataset
  - sentiment tool
  - trend tool
  - QA tool (OpenAI)
- Returns both:
  - final natural-language answer
  - execution artifacts (sentiment distribution, top negative videos, monthly negative ratio)

## Demo Video

[![Watch the demo](https://img.youtube.com/vi/1EvXgXP_BKV9lDnUOsECHEY14UGHHunI_/0.jpg)](https://drive.google.com/file/d/1EvXgXP_BKV9lDnUOsECHEY14UGHHunI_/view?usp=drive_link)

> **Click the image above to watch the demo video on Google Drive.**

## Why These Technologies

- **Streamlit**: fastest way to ship an interactive analytics UI and iterate quickly.
- **Google API Python Client (YouTube Data API v3)**: official and reliable source for video/comment metadata.
- **Transformers + PyTorch (`cardiffnlp/twitter-roberta-base-sentiment`)**: solid baseline for sentiment classification.
- **OpenAI Python Client**: high-quality reasoning/summarization for question answering.
- **Pandas**: simple and powerful tabular processing for filtering, grouping, and aggregation.
- **Matplotlib / Seaborn**: lightweight visual summaries inside Streamlit.

## Current Workflow

1. Select scope (`Channel` or `Playlist`).
2. Select date range (`Last 2 Months`, `Last 3 Months`, or `Custom`).
3. Collect comments (invalid/private/deleted videos are skipped and reported).
4. Select dataset in analysis tab.
5. Ask a question and inspect:
   - answer
   - step-by-step execution status
   - artifacts produced by the pipeline

## Project Structure

```text
.
├── data/
│   └── comments_*/                  # Run outputs: csv + metadata.json
├── src/
│   ├── streamlit_app.py             # UI and orchestration entrypoint
│   ├── agent_executor.py            # Deterministic step runner (agent-style flow)
│   ├── analysis_service.py          # Sentiment + OpenAI QA services
│   ├── comment_collection_service.py# Scope-based collection and merge logic
│   ├── get_id.py                    # YouTube channel/playlist/video ID helpers
│   ├── comment.py                   # Comment fetching per video
│   ├── config.py                    # Config + .env loading
│   ├── models.py                    # Shared dataclasses
│   ├── exceptions.py                # App-specific exception types
│   └── logging_utils.py             # Logging setup helpers
├── .env.example                     # Environment variable template
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- YouTube Data API key
- OpenAI API key

### Install

```bash
python3 -m venv youtubenv
source youtubenv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create `.env` in project root:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0
```

## Run

```bash
source youtubenv/bin/activate
streamlit run src/streamlit_app.py --server.fileWatcherType none
```

## Notes

- If a playlist contains deleted/private videos, those videos are skipped and reported in metadata.
- Execution artifacts are intended for transparency and debugging (not just final answer text).

## License

This project is developed for educational and research purposes.


