import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_service import (
    analyze_sentiment as service_analyze_sentiment,
    build_llm,
    get_ai_answer as service_get_ai_answer,
    get_comment_insights as service_get_comment_insights,
    load_sentiment_model as service_load_sentiment_model,
)
from agent_executor import run_question_agent
from comment_collection_service import collect_comments_for_request, list_collections
from config import load_config
from exceptions import AppError
from logging_utils import get_logger, setup_logging
from models import AnalysisRequest

setup_logging()
logger = get_logger(__name__)
config = load_config()
llm_client = build_llm(config)


def _resolve_date_range(preset_label: str):
    today = datetime.now().date()
    if preset_label == "Last 2 Months":
        start = today - timedelta(days=60)
        return start, today
    if preset_label == "Last 3 Months":
        start = today - timedelta(days=90)
        return start, today
    return None, None


def collect_comments(api_key: str):
    """Collect comments from YouTube videos and save them to CSV files"""
    st.subheader("Collect Comments")

    scope_label = st.radio(
        "Analysis scope",
        ["Channel", "Playlist (Concept)"],
        horizontal=True,
        help="Channel analyzes overall performance, Playlist analyzes a specific content concept.",
    )
    scope = "channel" if scope_label == "Channel" else "playlist"

    url_label = "Channel URL" if scope == "channel" else "Playlist URL"
    url = st.text_input(
        f"Enter {url_label}:",
        placeholder="https://www.youtube.com/...",
    )

    date_preset = st.radio(
        "Date range",
        ["Last 2 Months", "Last 3 Months", "Custom"],
        horizontal=True,
    )

    preset_start, preset_end = _resolve_date_range(date_preset)
    col1, col2 = st.columns(2)
    if date_preset == "Custom":
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")
    else:
        start_date = preset_start
        end_date = preset_end
        with col1:
            st.date_input("Start Date", value=start_date, disabled=True)
        with col2:
            st.date_input("End Date", value=end_date, disabled=True)
    
    if st.button("Collect Comments"):
        if not api_key:
            st.error("YouTube API key is required. Set YOUTUBE_API_KEY in your environment.")
            return

        if not url:
            st.error("Please enter a URL")
            return

        if start_date > end_date:
            st.error("Start date cannot be after end date.")
            return
        
        # Convert dates to datetime objects
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        with st.spinner("Collecting comments..."):
            try:
                request = AnalysisRequest(
                    scope=scope,
                    url=url,
                    start_date=start_datetime,
                    end_date=end_datetime,
                )
                metadata = collect_comments_for_request(request=request, api_key=api_key)
                output_dir = metadata.output_dir

                st.success(f"Successfully collected and saved comments to {output_dir}")
                st.info(
                    f"Scope: {metadata.scope} | Requested: {metadata.requested_video_count} | "
                    f"Processed: {metadata.video_count} | Skipped: {metadata.skipped_video_count} | "
                    f"Range: {start_date} to {end_date}"
                )
                st.balloons()
                st.experimental_rerun()
            except AppError as exc:
                st.error(str(exc))
            except Exception as exc:
                logger.exception("Unexpected error during comment collection")
                st.error(f"An unexpected error occurred: {exc}")

# Initialize sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    return service_load_sentiment_model()

def analyze_sentiment(comments_df, model, tokenizer, device):
    return service_analyze_sentiment(comments_df, model, tokenizer, device)

def get_comment_insights(comments_df, max_comments=100):
    try:
        return service_get_comment_insights(
            client=llm_client,
            model_name=config.openai_model,
            comments_df=comments_df,
            max_comments=max_comments,
        )
    except Exception as exc:
        logger.exception("Failed to generate insights")
        return f"Error getting insights: {exc}"

def get_ai_answer(question, comments_df, max_comments=50):
    try:
        return service_get_ai_answer(
            client=llm_client,
            model_name=config.openai_model,
            question=question,
            comments_df=comments_df,
            max_comments=max_comments,
        )
    except Exception as exc:
        logger.exception("Failed to generate answer")
        return f"Error getting answer: {exc}"

def plot_sentiment_distribution(comments_df):
    """Create sentiment distribution visualization"""
    plt.figure(figsize=(10, 6))
    sentiment_counts = comments_df['Sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Distribution of Comment Sentiments')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    return plt

def plot_sentiment_by_video(comments_df):
    """Create sentiment distribution by video visualization using pie charts"""
    # Get unique videos
    videos = comments_df['Video Title'].unique()
    
    # Calculate number of rows and columns for subplot grid
    n_videos = len(videos)
    n_cols = 2
    n_rows = (n_videos + 1) // 2  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    # Create a pie chart for each video
    for idx, video in enumerate(videos):
        video_data = comments_df[comments_df['Video Title'] == video]
        sentiment_counts = video_data['Sentiment'].value_counts()
        
        # Create pie chart
        ax = axes[idx]
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=['#ff9999', '#66b3ff', '#99ff99'],  # Red for negative, blue for neutral, green for positive
            startangle=90
        )
        
        # Set title
        ax.set_title(f"{video[:30]}..." if len(video) > 30 else video)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
    
    # Hide any unused subplots
    for idx in range(len(videos), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def _render_agent_steps(steps):
    status_icon = {
        "completed": "✅",
        "in_progress": "⏳",
        "failed": "❌",
    }
    for step in steps:
        icon = status_icon.get(step.status, "•")
        st.write(f"{icon} **{step.title}** - {step.message}")

def main():
    st.title("YouTube Comments Analysis Agent")
    st.sidebar.header("Configuration")
    st.sidebar.caption("API keys are loaded from environment variables.")

    if not config.youtube_api_key:
        st.sidebar.warning("Missing YouTube API key.")
    if not config.openai_api_key:
        st.sidebar.warning("Missing OpenAI API key.")
    
    # Main tabs
    tab_collect, tab_analyze = st.tabs(["Collect Comments", "Analyze Comments"])
    
    with tab_collect:
        collect_comments(api_key=config.youtube_api_key)
    
    with tab_analyze:
        st.sidebar.header("Data Selection")
        analysis_scope = st.sidebar.radio(
            "Dataset scope",
            ["All", "Channel", "Playlist (Concept)"],
            help="Filter collected datasets by scope.",
        )

        selected_scope = None
        if analysis_scope == "Channel":
            selected_scope = "channel"
        elif analysis_scope == "Playlist (Concept)":
            selected_scope = "playlist"

        collections = list_collections(scope=selected_scope)
        if not collections:
            st.warning("No comment files found. Please collect comments first using the 'Collect Comments' tab.")
            return

        options = [item.merged_file for item in collections]
        labels = {
            item.merged_file: (
                f"{os.path.basename(item.output_dir)} | {item.scope} | "
                f"processed={item.video_count} | skipped={item.skipped_video_count}"
            )
            for item in collections
        }
        selected_file = st.sidebar.selectbox(
            "Select a comment file",
            options,
            format_func=lambda x: labels[x],
        )
        
        # Load and process data
        if selected_file:
            selected_collection = next(item for item in collections if item.merged_file == selected_file)
            st.caption(
                f"Using `{selected_collection.scope}` dataset from `{selected_collection.output_dir}`"
            )
            df = pd.read_csv(selected_file)
            
            # Add date filtering
            st.sidebar.header("Date Range Filter")
            df['Published At'] = pd.to_datetime(df['Published At'])
            min_date = df['Published At'].min().date()
            max_date = df['Published At'].max().date()
            
            start_date = st.sidebar.date_input("Start Date", min_date)
            end_date = st.sidebar.date_input("End Date", max_date)
            
            # Filter data by date
            mask = (df['Published At'].dt.date >= start_date) & (df['Published At'].dt.date <= end_date)
            filtered_df = df[mask].copy()
            
            # Load sentiment model
            model, tokenizer, device = load_sentiment_model()
            
            # Perform sentiment analysis if not already done
            if 'Sentiment' not in filtered_df.columns:
                with st.spinner('Analyzing sentiment...'):
                    filtered_df = analyze_sentiment(filtered_df, model, tokenizer, device)
            
            # Analysis tabs
            tab1, tab2, tab3 = st.tabs(["Overview", "Video Analysis", "AI Insights"])
            
            with tab1:
                st.subheader("Overall Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Comments", len(filtered_df))
                with col2:
                    st.metric("Unique Videos", filtered_df['Video Title'].nunique())
                with col3:
                    st.metric("Average Likes", round(filtered_df['Like Count'].mean(), 1))
                
                st.subheader("Overall Sentiment Distribution")
                st.pyplot(plot_sentiment_distribution(filtered_df))
                
                st.subheader("Sentiment Distribution by Video")
                st.pyplot(plot_sentiment_by_video(filtered_df))
                st.caption("Each pie chart shows the proportion of positive, negative, and neutral comments for each video")
            
            with tab2:
                st.subheader("Analysis by Video")
                selected_video = st.selectbox(
                    "Select a video",
                    filtered_df['Video Title'].unique()
                )
                
                video_df = filtered_df[filtered_df['Video Title'] == selected_video]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Comments", len(video_df))
                    st.metric("Average Likes", round(video_df['Like Count'].mean(), 1))
                
                with col2:
                    sentiment_counts = video_df['Sentiment'].value_counts()
                    st.metric("Most Common Sentiment", sentiment_counts.index[0])
                    st.metric("Positive Comments", len(video_df[video_df['Sentiment'] == 'POSITIVE']))
                
                st.subheader("Top Comments")
                top_comments = video_df.nlargest(5, 'Like Count')
                for _, comment in top_comments.iterrows():
                    st.write(f"**{comment['Like Count']} likes:** {comment['Comment']}")
                    st.write(f"*Sentiment: {comment['Sentiment']}*")
                    st.write("---")
            
            with tab3:
                st.subheader("AI-Powered Insights")
                
                # Add analysis scope selection
                analysis_scope = st.radio(
                    "Choose analysis scope:",
                    ["All Videos", "Single Video"],
                    horizontal=True,
                    help="Select whether to analyze all videos or focus on a single video"
                )
                
                # If single video is selected, show video selector
                if analysis_scope == "Single Video":
                    selected_video = st.selectbox(
                        "Select a video to analyze",
                        filtered_df['Video Title'].unique(),
                        help="Choose a specific video for detailed analysis"
                    )
                    # Filter data for selected video
                    analysis_df = filtered_df[filtered_df['Video Title'] == selected_video].copy()
                    st.info(f"Analyzing comments from: {selected_video}")
                else:
                    analysis_df = filtered_df.copy()
                    st.info(f"Analyzing comments from all {len(filtered_df['Video Title'].unique())} videos")
                
                # Add comment limit selector with dynamic max value
                total_comments = len(analysis_df)
                max_comments_slider = min(200, total_comments)  # Don't allow slider max to exceed available comments
                
                max_comments = st.slider(
                    "Maximum number of comments to analyze",
                    min_value=10,
                    max_value=max_comments_slider,
                    value=min(100, max_comments_slider),
                    step=10,
                    help=f"Limiting the number of comments helps prevent API rate limits. Total available comments: {total_comments}"
                )
                
                if st.button("Generate Insights"):
                    with st.spinner("Analyzing comments..."):
                        insights = get_comment_insights(analysis_df, max_comments)
                        st.write(insights)
                
                st.subheader("Ask About Comments")
                
                # Add example questions based on scope
                if analysis_scope == "Single Video":
                    example_questions = [
                        "What are the main topics discussed in this video's comments?",
                        "What feedback are viewers giving about this specific video?",
                        "What are the most common questions about this video?",
                        "How do viewers feel about this particular video?",
                        "What aspects of this video are most appreciated by viewers?"
                    ]
                else:
                    example_questions = [
                        "What are the common themes across all videos?",
                        "How does viewer engagement compare across different videos?",
                        "What patterns can you see in the comments across all videos?",
                        "What are the most common suggestions across all videos?",
                        "How has viewer sentiment changed over time across all videos?"
                    ]
                
                # Show example questions
                with st.expander("Example Questions"):
                    for question in example_questions:
                        if st.button(question, key=question):
                            st.session_state['user_question'] = question
                
                # Question input with example question support
                user_question = st.text_input(
                    "Ask a question about the comments:",
                    value=st.session_state.get('user_question', ''),
                    help="Ask any question about the comments. Use the example questions above for inspiration."
                )
                
                # Add comment limit for Q&A with dynamic max value
                qa_max_comments = st.slider(
                    "Maximum comments to consider for answer",
                    min_value=10,
                    max_value=min(100, total_comments),
                    value=min(50, total_comments),
                    step=10,
                    help=f"Limiting the number of comments helps prevent API rate limits. Total available comments: {total_comments}"
                )
                
                if user_question and st.button("Get Answer"):
                    with st.spinner("Thinking..."):
                        agent_result = run_question_agent(
                            question=user_question,
                            comments_df=analysis_df,
                            llm_client=llm_client,
                            llm_model_name=config.openai_model,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            max_comments=qa_max_comments,
                        )

                    st.subheader("Agent Execution")
                    _render_agent_steps(agent_result.steps)

                    if agent_result.error:
                        st.error(agent_result.error)
                    else:
                        st.subheader("Answer")
                        st.write(agent_result.answer)
                        with st.expander("Execution Artifacts"):
                            st.json(agent_result.artifacts)
                        
                        # Add a note about the analysis scope
                        if analysis_scope == "Single Video":
                            st.info(f"Analysis based on comments from: {selected_video}")
                        else:
                            st.info(f"Analysis based on comments from all {len(filtered_df['Video Title'].unique())} videos")

if __name__ == "__main__":
    main() 