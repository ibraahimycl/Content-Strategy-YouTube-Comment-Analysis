# 📊 YouTube Comment Analysis - AI-Powered Analytics Platform

## 🎥 Demo Video

[![Watch the demo](https://img.youtube.com/vi/1EvXgXP_BKV9lDnUOsECHEY14UGHHunI_/0.jpg)](https://drive.google.com/file/d/1EvXgXP_BKV9lDnUOsECHEY14UGHHunI_/view?usp=drive_link)

> **Click the image above to watch the demo video on Google Drive.**

A comprehensive YouTube comment analysis platform that combines multiple AI technologies to extract insights from YouTube comments using sentiment analysis, topic modeling, and natural language processing.

## 🚀 Features

### 🔍 Data Collection
- **YouTube API Integration**: Collect comments from channels and playlists
- **Date Range Filtering**: Analyze comments from specific time periods
- **Batch Processing**: Handle multiple videos simultaneously
- **Automatic Organization**: Save comments with video titles and metadata

### 🤖 AI-Powered Analysis
- **Sentiment Analysis**: Using Hugging Face Transformers (RoBERTa model)
- **Topic Modeling**: BERTopic for automatic topic discovery
- **GPT-4 Integration**: OpenAI-powered comment insights and Q&A
- **Advanced NLP**: Sentence transformers for semantic understanding

### 📈 Visualization & Insights
- **Interactive Dashboards**: Streamlit-based web interface
- **Sentiment Distribution**: Visual sentiment analysis results
- **Topic Clustering**: UMAP and HDBSCAN for topic visualization
- **Comment Analytics**: Like count analysis and engagement metrics

## 🛠️ Technologies Used

### 🤖 AI & Machine Learning
- **Hugging Face Transformers**: `cardiffnlp/twitter-roberta-base-sentiment` for sentiment analysis
- **OpenAI GPT-4**: Advanced comment analysis and insights generation
- **BERTopic**: Topic modeling and clustering
- **Sentence Transformers**: Semantic text embeddings

### 📊 Data Processing & Visualization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive charts and graphs
- **Streamlit**: Web application framework


### 🌐 APIs & Integration
- **Google API Python Client**: YouTube Data API v3 integration
- **OpenAI Python Client**: GPT-4 API integration

## 📁 Project Structure

```
YoutubeProject/
├── streamlit_app.py           # Main Streamlit web application
├── get_id.py                  # YouTube API utilities for video/channel extraction
├── comment.py                 # Comment collection and processing
├── channel_comments.py        # Channel-specific comment analysis
├── sentiment_analysis.py      # Sentiment analysis implementation
├── topic_analysis.py          # BERTopic-based topic modeling
├── topic_analysis_gpt.py      # GPT-4 powered topic analysis
├── requirements.txt           # Python dependencies
├── COMMENTS/                  # Comment data storage
├── comments_*/                # Timestamped comment collections
└── README.md                  # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- YouTube Data API v3 key
- OpenAI API key

### Setup
```bash
# Clone the repository
git clone https://github.com/ibraahimycl/Youtube-Comment-Analysis.git
cd Youtube-Comment-Analysis

# Install dependencies
pip install -r requirements.txt

# Set up API keys
# Add your YouTube API key and OpenAI API key to the application
```

## 🎮 Usage

### Running the Application
```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

### Features Available

#### 1. Comment Collection
- Enter YouTube channel or playlist URL
- Select date range for analysis
- Automatically collect and organize comments

#### 2. Sentiment Analysis
- Real-time sentiment classification using Hugging Face models
- Positive, negative, and neutral sentiment detection
- Sentiment distribution visualization

#### 3. Topic Modeling
- **BERTopic Analysis**: Automatic topic discovery using BERT embeddings
- **GPT-4 Topic Analysis**: AI-powered topic grouping and insights
- Interactive topic visualization with UMAP

#### 4. AI-Powered Insights
- Ask questions about your comment data
- Get AI-generated insights and summaries
- Analyze comment patterns and trends

## 🔧 Technical Implementation

### Sentiment Analysis Pipeline
```python
# Using Hugging Face Transformers
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Topic Modeling with BERTopic
```python
# BERTopic for automatic topic discovery
from bertopic import BERTopic
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(documents)
```

### GPT-4 Integration
```python
# OpenAI GPT-4 for advanced analysis
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

### YouTube API Integration
```python
# Google API Python Client for YouTube data
from googleapiclient.discovery import build
youtube = build('youtube', 'v3', developerKey=API_KEY)
```

## 📊 Analysis Capabilities

### Sentiment Analysis
- **Model**: RoBERTa-based sentiment classifier
- **Accuracy**: High accuracy for social media text
- **Output**: Positive, Negative, Neutral classifications

### Topic Modeling
- **BERTopic**: Automatic topic discovery
- **UMAP**: 2D visualization of topics
- **HDBSCAN**: Density-based clustering
- **GPT-4**: AI-powered topic analysis

### Comment Analytics
- **Engagement Metrics**: Like count analysis
- **Temporal Analysis**: Comments over time
- **Video Comparison**: Cross-video analysis
- **User Insights**: Comment patterns and trends

## 🎯 Use Cases

### Content Creators
- **Audience Sentiment**: Understand viewer reactions
- **Content Optimization**: Identify what resonates with audience
- **Engagement Analysis**: Track comment engagement patterns

### Marketing Teams
- **Brand Monitoring**: Track brand mentions and sentiment
- **Competitor Analysis**: Analyze competitor video comments
- **Campaign Effectiveness**: Measure campaign impact

### Researchers
- **Social Media Analysis**: Study online discourse patterns
- **Sentiment Trends**: Track public opinion over time
- **Topic Evolution**: Analyze trending topics and themes

## 🔮 Future Enhancements

- [ ] Multi-language sentiment analysis
- [ ] Real-time comment monitoring
- [ ] Advanced visualization dashboards
- [ ] Comment toxicity detection
- [ ] Influencer identification
- [ ] Automated report generation
- [ ] Integration with other social media platforms

## 📝 License

This project is developed for educational and research purposes.

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Enhancing AI models and analysis

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

## 🎯 Key Features Summary

- **Multi-Model AI**: Combines Hugging Face, OpenAI, and BERTopic
- **Real-time Analysis**: Live sentiment and topic analysis
- **Interactive Interface**: Streamlit-based web application
- **Comprehensive Analytics**: Sentiment, topics, engagement metrics
- **Scalable Architecture**: Handles large comment datasets
- **Professional Visualization**: Advanced charts and insights 
