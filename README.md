# 📊 YouTube Comment Analysis - AI-Powered Analytics Platform
This AI-powered analytics platform is designed to help content creators transform raw YouTube comments into strategic insights. By leveraging sentiment analysis,
topic modeling, and natural language processing, it provides the data-driven feedback needed to refine content and deepen audience engagement.

## 🎥 Demo Video

[![Watch the demo](https://img.youtube.com/vi/1EvXgXP_BKV9lDnUOsECHEY14UGHHunI_/0.jpg)](https://drive.google.com/file/d/1EvXgXP_BKV9lDnUOsECHEY14UGHHunI_/view?usp=drive_link)

> **Click the image above to watch the demo video on Google Drive.**


## Features

### Data Collection
- **YouTube API Integration**: Collect comments from channels and playlists
- **Date Range Filtering**: Analyze comments from specific time periods
- **Batch Processing**: Handle multiple videos simultaneously
- **Automatic Organization**: Save comments with video titles and metadata

### AI-Powered Analysis
- **Sentiment Analysis**: Using Hugging Face Transformers (RoBERTa model)
- **Topic Modeling**: BERTopic for automatic topic discovery
- **GPT-4 Integration**: OpenAI-powered comment insights and Q&A
- **Advanced NLP**: Sentence transformers for semantic understanding

### Visualization & Insights
- **Interactive Dashboards**: Streamlit-based web interface
- **Sentiment Distribution**: Visual sentiment analysis results
- **Topic Clustering**: UMAP and HDBSCAN for topic visualization
- **Comment Analytics**: Like count analysis and engagement metrics

## Technologies Used

### Data Processing & Visualization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Static visualizations
- **Streamlit**: Web application framework


### APIs & Integration
- **Google API Python Client**: YouTube Data API v3 integration
- **OpenAI Python Client**: GPT-4 API integration

## 📁 Project Structure

```
.
├── data/                    # Stores collected and processed comment data
│   └── comments_*/            # Timestamped folders for each data collection run
├── notebooks/               # Jupyter notebooks for analysis and exploration
│   └── readcsv.ipynb          # Example notebook for reading comment data
├── src/                     # Main source code for the application
│   ├── streamlit_app.py       # The core Streamlit web application
│   ├── get_id.py              # Utilities for the YouTube Data API
│   └── comment.py             # Logic for comment collection and processing
├── .gitignore               # Specifies files and folders for Git to ignore
├── README.md                # Project documentation (this file)
└── requirements.txt         # List of Python dependencies
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

## Usage

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

## Use Cases

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
- 
## 📝 License

This project is developed for educational and research purposes.


