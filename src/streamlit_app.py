import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from openai import OpenAI
import json
from collections import defaultdict
import glob
from get_id import extract_channel_id, get_uploads_playlist_id, get_videos_by_date_range, extract_playlist_id, get_video_title
from comment import video_comments
import re

# Initialize OpenAI client
client = OpenAI(api_key="")

API_KEY = ""

def sanitize_filename(title):
    # Remove special characters and replace spaces with underscores
    sanitized = re.sub(r'[^\w\s-]', '', title.lower())
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized

def collect_comments():
    """Collect comments from YouTube videos and save them to CSV files"""
    st.subheader("Collect Comments")
    
    # Input type selection
    input_type = st.radio(
        "Choose input type:",
        ["Channel URL", "Playlist URL"],
        horizontal=True
    )
    
    # URL input
    url = st.text_input(
        f"Enter the {input_type.lower()}:",
        placeholder="https://www.youtube.com/..."
    )
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    
    if st.button("Collect Comments"):
        if not url:
            st.error("Please enter a URL")
            return
        
        # Convert dates to datetime objects
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        with st.spinner("Collecting comments..."):
            try:
                # Get video IDs based on input type
                if input_type == "Channel URL":
                    channel_id = extract_channel_id(url, API_KEY)
                    if not channel_id:
                        st.error("Could not get channel ID.")
                        return
                    
                    playlist_id = get_uploads_playlist_id(channel_id)
                    video_ids = get_videos_by_date_range(playlist_id, start_datetime, end_datetime, is_channel_uploads=True)
                else:
                    playlist_id = extract_playlist_id(url, API_KEY)
                    if not playlist_id:
                        st.error("Could not get playlist ID.")
                        return
                    
                    video_ids = get_videos_by_date_range(playlist_id, start_datetime, end_datetime)
                
                if not video_ids:
                    st.warning("No videos found in the specified date range.")
                    return
                
                # Create a directory for the CSV files
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"data/comments_{timestamp}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # List to store all CSV filenames
                csv_files = []
                
                # Process each video
                for i, video_id in enumerate(video_ids, 1):
                    status_text.text(f"Processing video {i}/{len(video_ids)}")
                    progress_bar.progress(i / len(video_ids))
                    
                    # Get video title
                    video_title = get_video_title(video_id, API_KEY)
                    if video_title:
                        safe_title = sanitize_filename(video_title)
                        csv_filename = os.path.join(output_dir, f"{safe_title}.csv")
                    else:
                        csv_filename = os.path.join(output_dir, f"video_{video_id}.csv")
                    
                    try:
                        video_comments(video_id, csv_filename)
                        csv_files.append(csv_filename)
                    except Exception as e:
                        st.error(f"Error getting comments for video {video_id}: {e}")
                
                if not csv_files:
                    st.error("Could not get comments for any videos.")
                    return
                
                # Merge all CSV files
                status_text.text("Merging comment files...")
                all_comments = []
                
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        video_title = os.path.basename(csv_file).replace('.csv', '')
                        df['Video Title'] = video_title
                        all_comments.append(df)
                    except Exception as e:
                        st.error(f"Error reading file {csv_file}: {e}")
                
                if all_comments:
                    # Combine all dataframes
                    merged_df = pd.concat(all_comments, ignore_index=True)
                    
                    # Save merged file
                    merged_filename = os.path.join(output_dir, "all_comments.csv")
                    merged_df.to_csv(merged_filename, index=False)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"Successfully collected and saved comments to {output_dir}")
                    st.balloons()
                    
                    # Automatically refresh the page to show the new data
                    st.experimental_rerun()
                else:
                    st.error("No comments to merge.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Initialize sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

def analyze_sentiment(comments_df, model, tokenizer, device):
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
    
    comments_df['Sentiment'] = comments_df['Comment'].apply(get_sentiment)
    return comments_df

def get_comment_insights(comments_df, max_comments=100):
    """Get insights about comments using GPT-4"""
    # Sort comments by like count and get top comments
    top_comments = comments_df.nlargest(max_comments, 'Like Count')
    
    # Get video count and names if analyzing multiple videos
    video_count = comments_df['Video Title'].nunique()
    if video_count > 1:
        video_names = comments_df['Video Title'].unique().tolist()
        video_info = f"Analyzing {video_count} videos: {', '.join(video_names[:3])}{'...' if len(video_names) > 3 else ''}"
    else:
        video_info = f"Analyzing video: {comments_df['Video Title'].iloc[0]}"
    
    # Prepare the comments data with a more concise format
    comments_text = "\n".join([
        f"Video: {row['Video Title']}\nComment: {row['Comment'][:200]}...\nLikes: {row['Like Count']}\nSentiment: {row['Sentiment']}\n"
        for _, row in top_comments.iterrows()
    ])
    
    # Add summary statistics
    total_comments = len(comments_df)
    sentiment_dist = comments_df['Sentiment'].value_counts().to_dict()
    avg_likes = comments_df['Like Count'].mean()
    
    summary = f"""{video_info}
Total Comments: {total_comments}
Sentiment Distribution: {sentiment_dist}
Average Likes: {avg_likes:.1f}

Top {max_comments} Comments (by likes):
{comments_text}"""

    # Create the prompt
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

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes YouTube comments and provides insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting insights: {str(e)}"

def get_ai_answer(question, comments_df, max_comments=50):
    """Get AI answer for a specific question about comments"""
    # Get video count and names if analyzing multiple videos
    video_count = comments_df['Video Title'].nunique()
    if video_count > 1:
        video_names = comments_df['Video Title'].unique().tolist()
        video_info = f"Analyzing {video_count} videos: {', '.join(video_names[:3])}{'...' if len(video_names) > 3 else ''}"
    else:
        video_info = f"Analyzing video: {comments_df['Video Title'].iloc[0]}"
    
    # Sort comments by relevance to the question (using like count as a proxy for importance)
    relevant_comments = comments_df.nlargest(max_comments, 'Like Count')
    
    # Prepare a concise summary of the comments
    comments_summary = "\n".join([
        f"Video: {row['Video Title']}\nComment: {row['Comment'][:150]}...\nLikes: {row['Like Count']}\nSentiment: {row['Sentiment']}\n"
        for _, row in relevant_comments.iterrows()
    ])
    
    # Add summary statistics
    total_comments = len(comments_df)
    sentiment_dist = comments_df['Sentiment'].value_counts().to_dict()
    
    summary = f"""{video_info}
Total Comments Analyzed: {total_comments}
Sentiment Distribution: {sentiment_dist}

Relevant Comments:
{comments_summary}"""

    prompt = f"""Based on these comments, answer the following question: {question}

Comment Summary:
{summary}

Please provide a detailed and helpful answer focusing on the most relevant comments. If analyzing multiple videos, highlight any differences or patterns across videos."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes YouTube comments and provides insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting answer: {str(e)}"

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

def main():
    st.title("YouTube Comments Analysis Agent")
    
    # Main tabs
    tab_collect, tab_analyze = st.tabs(["Collect Comments", "Analyze Comments"])
    
    with tab_collect:
        collect_comments()
    
    with tab_analyze:
        # Sidebar for file selection
        st.sidebar.header("Data Selection")
        
        # Get list of comment files
        comment_files = glob.glob("data/comments_*/all_comments.csv")
        if not comment_files:
            st.warning("No comment files found. Please collect comments first using the 'Collect Comments' tab.")
            return
        
        selected_file = st.sidebar.selectbox(
            "Select a comment file",
            comment_files,
            format_func=lambda x: os.path.basename(os.path.dirname(x))
        )
        
        # Load and process data
        if selected_file:
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
                        answer = get_ai_answer(user_question, analysis_df, qa_max_comments)
                        st.write(answer)
                        
                        # Add a note about the analysis scope
                        if analysis_scope == "Single Video":
                            st.info(f"Analysis based on comments from: {selected_video}")
                        else:
                            st.info(f"Analysis based on comments from all {len(filtered_df['Video Title'].unique())} videos")

if __name__ == "__main__":
    main() 