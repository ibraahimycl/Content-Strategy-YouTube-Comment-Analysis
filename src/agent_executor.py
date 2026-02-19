from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

from analysis_service import analyze_sentiment, get_ai_answer


@dataclass
class AgentStepResult:
    step_id: str
    title: str
    status: str
    message: str


@dataclass
class AgentRunResult:
    steps: List[AgentStepResult] = field(default_factory=list)
    answer: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    comments_df: Optional[pd.DataFrame] = None


def _add_step(steps: List[AgentStepResult], step_id: str, title: str, status: str, message: str) -> None:
    steps.append(
        AgentStepResult(
            step_id=step_id,
            title=title,
            status=status,
            message=message,
        )
    )


def _build_trend_artifacts(comments_df: pd.DataFrame) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    artifacts["sentiment_distribution"] = comments_df["Sentiment"].value_counts().to_dict()

    by_video = (
        comments_df.groupby("Video Title")["Sentiment"]
        .apply(lambda values: (values == "NEGATIVE").mean())
        .sort_values(ascending=False)
    )
    artifacts["top_negative_videos"] = [
        {"video_title": title, "negative_ratio": round(ratio, 3)}
        for title, ratio in by_video.head(5).items()
    ]

    if "Published At" in comments_df.columns:
        dated_df = comments_df.copy()
        dated_df["Published At"] = pd.to_datetime(dated_df["Published At"], errors="coerce")
        dated_df = dated_df.dropna(subset=["Published At"])
        if not dated_df.empty:
            dated_df["month"] = dated_df["Published At"].dt.to_period("M").astype(str)
            monthly_negative_ratio = (
                dated_df.assign(is_negative=dated_df["Sentiment"] == "NEGATIVE")
                .groupby("month")["is_negative"]
                .mean()
                .sort_index()
            )
            artifacts["monthly_negative_ratio"] = {
                month: round(float(ratio), 3) for month, ratio in monthly_negative_ratio.items()
            }

    return artifacts


def run_question_agent(
    question: str,
    comments_df: pd.DataFrame,
    llm_client: OpenAI,
    llm_model_name: str,
    model,
    tokenizer,
    device,
    max_comments: int = 50,
) -> AgentRunResult:
    steps: List[AgentStepResult] = []
    working_df = comments_df.copy()

    try:
        _add_step(steps, "validate_input", "Validate Dataset", "in_progress", "Checking dataset availability.")
        if working_df.empty:
            raise ValueError("Dataset is empty. Please collect comments first.")
        _add_step(steps, "validate_input", "Validate Dataset", "completed", "Dataset is ready.")

        _add_step(steps, "sentiment", "Run Sentiment Tool", "in_progress", "Ensuring sentiment labels exist.")
        if "Sentiment" not in working_df.columns:
            working_df = analyze_sentiment(working_df, model, tokenizer, device)
            sentiment_msg = "Sentiment analysis completed."
        else:
            sentiment_msg = "Sentiment labels already present, skipped recomputation."
        _add_step(steps, "sentiment", "Run Sentiment Tool", "completed", sentiment_msg)

        _add_step(steps, "trend", "Run Trend Tool", "in_progress", "Computing sentiment trend artifacts.")
        artifacts = _build_trend_artifacts(working_df)
        _add_step(steps, "trend", "Run Trend Tool", "completed", "Trend artifacts generated.")

        _add_step(steps, "answer", "Run QA Tool", "in_progress", "Generating final answer with LLM.")
        answer = get_ai_answer(
            client=llm_client,
            model_name=llm_model_name,
            question=question,
            comments_df=working_df,
            max_comments=max_comments,
        )
        _add_step(steps, "answer", "Run QA Tool", "completed", "Answer generated.")

        return AgentRunResult(
            steps=steps,
            answer=answer,
            artifacts=artifacts,
            comments_df=working_df,
            error=None,
        )
    except Exception as exc:
        _add_step(
            steps,
            "failed",
            "Execution Failed",
            "failed",
            f"Agent execution stopped: {exc}",
        )
        return AgentRunResult(
            steps=steps,
            answer="",
            artifacts={},
            comments_df=working_df,
            error=str(exc),
        )

