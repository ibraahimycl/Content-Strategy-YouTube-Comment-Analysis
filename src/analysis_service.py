import pandas as pd
import torch
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import AppConfig


def build_llm(config: AppConfig) -> OpenAI:
    return OpenAI(api_key=config.openai_api_key)


def _invoke_openai(client: OpenAI, model_name: str, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device


def analyze_sentiment(comments_df: pd.DataFrame, model, tokenizer, device) -> pd.DataFrame:
    sentiment_labels = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    def get_sentiment(comment):
        if not isinstance(comment, str):
            return "NEUTRAL"

        inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(scores, dim=1).item()

        return sentiment_labels[predicted_class]

    comments_df["Sentiment"] = comments_df["Comment"].apply(get_sentiment)
    return comments_df


def get_comment_insights(
    client: OpenAI,
    model_name: str,
    comments_df: pd.DataFrame,
    max_comments: int = 100,
) -> str:
    top_comments = comments_df.nlargest(max_comments, "Like Count")

    video_count = comments_df["Video Title"].nunique()
    if video_count > 1:
        video_names = comments_df["Video Title"].unique().tolist()
        video_info = (
            f"Analyzing {video_count} videos: {', '.join(video_names[:3])}"
            f"{'...' if len(video_names) > 3 else ''}"
        )
    else:
        video_info = f"Analyzing video: {comments_df['Video Title'].iloc[0]}"

    comments_text = "\n".join(
        [
            f"Video: {row['Video Title']}\nComment: {row['Comment'][:200]}...\n"
            f"Likes: {row['Like Count']}\nSentiment: {row['Sentiment']}\n"
            for _, row in top_comments.iterrows()
        ]
    )

    total_comments = len(comments_df)
    sentiment_dist = comments_df["Sentiment"].value_counts().to_dict()
    avg_likes = comments_df["Like Count"].mean()

    summary = f"""{video_info}
Total Comments: {total_comments}
Sentiment Distribution: {sentiment_dist}
Average Likes: {avg_likes:.1f}

Top {max_comments} Comments (by likes):
{comments_text}"""

    prompt = f"""Analyze these YouTube video comments and provide insights about:
1. Overall sentiment and tone of the comments
2. Main topics or themes discussed
3. Common suggestions or feedback
4. Notable complaints or concerns
5. Most liked comments and their significance
6. General audience reaction and engagement

Comment Summary:
{summary}

Please provide a detailed analysis in a structured format. If analyzing multiple videos, highlight any differences or patterns across videos."""

    return _invoke_openai(
        client=client,
        model_name=model_name,
        system_prompt="You are a helpful assistant that analyzes YouTube comments and provides insights.",
        user_prompt=prompt,
    )


def get_ai_answer(
    client: OpenAI,
    model_name: str,
    question: str,
    comments_df: pd.DataFrame,
    max_comments: int = 50,
) -> str:
    video_count = comments_df["Video Title"].nunique()
    if video_count > 1:
        video_names = comments_df["Video Title"].unique().tolist()
        video_info = (
            f"Analyzing {video_count} videos: {', '.join(video_names[:3])}"
            f"{'...' if len(video_names) > 3 else ''}"
        )
    else:
        video_info = f"Analyzing video: {comments_df['Video Title'].iloc[0]}"

    relevant_comments = comments_df.nlargest(max_comments, "Like Count")
    comments_summary = "\n".join(
        [
            f"Video: {row['Video Title']}\nComment: {row['Comment'][:150]}...\n"
            f"Likes: {row['Like Count']}\nSentiment: {row['Sentiment']}\n"
            for _, row in relevant_comments.iterrows()
        ]
    )

    total_comments = len(comments_df)
    sentiment_dist = comments_df["Sentiment"].value_counts().to_dict()

    summary = f"""{video_info}
Total Comments Analyzed: {total_comments}
Sentiment Distribution: {sentiment_dist}

Relevant Comments:
{comments_summary}"""

    prompt = f"""Based on these comments, answer the following question: {question}

Comment Summary:
{summary}

Please provide a detailed and helpful answer focusing on the most relevant comments. If analyzing multiple videos, highlight any differences or patterns across videos."""

    return _invoke_openai(
        client=client,
        model_name=model_name,
        system_prompt="You are a helpful assistant that analyzes YouTube comments and provides insights.",
        user_prompt=prompt,
    )

