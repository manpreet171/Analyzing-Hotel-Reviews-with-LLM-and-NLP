import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from typing import Dict, List
from collections import Counter
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from typing import Dict, List, Tuple
from collections import Counter


# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

class FeedbackProcessor:
    POSITIVE_THRESHOLD = 0.5
    TOP_WORDS_COUNT = 10

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.stop_words = set(stopwords.words('english'))

    def collect_feedback(self) -> pd.DataFrame:
        """
        Collect feedback and perform sentiment analysis on review text.
        
        Returns:
        pd.DataFrame: DataFrame with added sentiment column.
        """
        self.data['review_text'] = self.data['review_text'].astype(str)
        self.data['sentiment'] = self.data['review_text'].apply(self.analyze_sentiment)
        return self.data

    def analyze_sentiment(self, text: str) -> float:
        """
        Perform sentiment analysis on given text.
        
        Args:
        text (str): The text to analyze.
        
        Returns:
        float: Sentiment polarity score.
        """
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def report_feedback(self, analyzed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a report of feedback data.
        
        Args:
        analyzed_data (pd.DataFrame): DataFrame with sentiment analysis results.
        
        Returns:
        pd.DataFrame: Aggregated report of feedback data.
        """
        report = analyzed_data.groupby('hotel_name').agg({
            'rating': ['mean', 'count'],
            'sentiment': 'mean'
        }).reset_index()
        return report

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary of the entire dataset.
        
        Returns:
        pd.DataFrame: Statistical summary of the dataset.
        """
        return self.data.describe(include='all')

    def get_most_common_themes(self) -> pd.Series:
        """
        Get the most common themes (words) from all reviews.
        
        Returns:
        pd.Series: Series of most common words and their frequencies.
        """
        words = Counter(word.lower() for review in self.data['review_text'] 
                        for word in review.split() 
                        if word.lower() not in self.stop_words)
        return pd.Series(dict(words.most_common(self.TOP_WORDS_COUNT)))

    def get_hotel_summary(self, hotel_name: str) -> pd.DataFrame:
        """
        Get a summary of reviews for a specific hotel.
        
        Args:
        hotel_name (str): Name of the hotel.
        
        Returns:
        pd.DataFrame: Statistical summary of reviews for the specified hotel.
        """
        hotel_reviews = self.data[self.data['hotel_name'] == hotel_name]
        if hotel_reviews.empty:
            return pd.DataFrame({'error': [f"No reviews found for hotel: {hotel_name}"]})
        return hotel_reviews.describe(include='all')

    def compare_hotels(self, hotel1: str, hotel2: str) -> Dict[str, pd.DataFrame]:
        """
        Compare two hotels based on their reviews.
        
        Args:
        hotel1 (str): Name of the first hotel.
        hotel2 (str): Name of the second hotel.
        
        Returns:
        Dict[str, pd.DataFrame]: Dictionary containing summaries for both hotels.
        """
        if hotel1 not in self.data['hotel_name'].values or hotel2 not in self.data['hotel_name'].values:
            return {"error": f"One or both hotels not found: {hotel1}, {hotel2}"}
        hotel1_reviews = self.data[self.data['hotel_name'] == hotel1]
        hotel2_reviews = self.data[self.data['hotel_name'] == hotel2]
        return {
            hotel1: hotel1_reviews.describe(include='all'),
            hotel2: hotel2_reviews.describe(include='all')
        }

    def get_reviews_by_nationality(self, nationality: str) -> pd.DataFrame:
        """
        Get a summary of reviews from a specific nationality.
        
        Args:
        nationality (str): The nationality to filter reviews by.
        
        Returns:
        pd.DataFrame: Summary of reviews from the specified nationality.
        """
        reviews = self.data[self.data['nationality'] == nationality]
        return reviews.describe(include='all')

    def get_reviews_by_rating(self, rating: int) -> pd.DataFrame:
        """
        Get reviews with a specific rating.
        
        Args:
        rating (int): The rating to filter reviews by.
        
        Returns:
        pd.DataFrame: Reviews with the specified rating.
        """
        return self.data[self.data['rating'] == rating]

    def get_suggestions_for_improvement(self, hotel_name: str) -> pd.Series:
        """
        Get improvement suggestions for a specific hotel based on low-rated reviews.
        
        Args:
        hotel_name (str): Name of the hotel.
        
        Returns:
        pd.Series: Series of low-rated reviews for the specified hotel.
        """
        reviews = self.data[self.data['hotel_name'] == hotel_name]
        return reviews['review_text'][reviews['rating'] < 6]

    def get_positive_aspects(self, hotel_name: str) -> pd.Series:
        """
        Get positive aspects of a specific hotel based on highly-rated reviews.
        
        Args:
        hotel_name (str): Name of the hotel.
        
        Returns:
        pd.Series: Series of most common positive words in reviews for the specified hotel.
        """
        hotel_reviews = self.data[self.data['hotel_name'] == hotel_name]
        positive_reviews = hotel_reviews[hotel_reviews['sentiment'] > self.POSITIVE_THRESHOLD]
        words = Counter(word.lower() for review in positive_reviews['review_text'] 
                        for word in review.split() 
                        if word.lower() not in self.stop_words)
        return pd.Series(dict(words.most_common(self.TOP_WORDS_COUNT)))

    def get_key_insights(self) -> Dict[str, Dict[str, int]]:
        """
        Get key insights from all reviews, including top positive and negative aspects.
        
        Returns:
        Dict[str, Dict[str, int]]: Dictionary containing top positive and negative words.
        """
        positive_reviews = self.data[self.data['sentiment'] > self.POSITIVE_THRESHOLD]
        negative_reviews = self.data[self.data['sentiment'] <= self.POSITIVE_THRESHOLD]
        
        positive_words = Counter(word.lower() for review in positive_reviews['review_text'] 
                                 for word in review.split() 
                                 if word.lower() not in self.stop_words)
        negative_words = Counter(word.lower() for review in negative_reviews['review_text'] 
                                 for word in review.split() 
                                 if word.lower() not in self.stop_words)
        
        return {
            "positive_aspects": dict(positive_words.most_common(self.TOP_WORDS_COUNT)),
            "negative_aspects": dict(negative_words.most_common(self.TOP_WORDS_COUNT))
        }
    def plot_rating_distribution(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a bar chart of rating distribution.
        
        Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and Axes objects of the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        self.data['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
        ax.set_title('Distribution of Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        return fig, ax

    def plot_sentiment_distribution(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a histogram of sentiment distribution.
        
        Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and Axes objects of the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.data['sentiment'], kde=True, ax=ax)
        ax.set_title('Distribution of Sentiment Scores')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Count')
        return fig, ax

    def plot_top_hotels(self, n: int = 10) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a bar chart of top n hotels by average rating.
        
        Args:
        n (int): Number of top hotels to display.
        
        Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and Axes objects of the plot.
        """
        top_hotels = self.data.groupby('hotel_name')['rating'].mean().nlargest(n)
        fig, ax = plt.subplots(figsize=(12, 6))
        top_hotels.plot(kind='bar', ax=ax)
        ax.set_title(f'Top {n} Hotels by Average Rating')
        ax.set_xlabel('Hotel')
        ax.set_ylabel('Average Rating')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig, ax

    def plot_rating_vs_sentiment(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a scatter plot of rating vs sentiment.
        
        Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and Axes objects of the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.data['rating'], self.data['sentiment'], alpha=0.5)
        ax.set_title('Rating vs Sentiment')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Sentiment Score')
        return fig, ax

    def plot_word_cloud(self, hotel_name: str = None) -> plt.Figure:
        """
        Create a word cloud of review text.
        
        Args:
        hotel_name (str, optional): If provided, create word cloud for specific hotel.
        
        Returns:
        plt.Figure: Figure object of the word cloud.
        """
        from wordcloud import WordCloud
        
        if hotel_name:
            text = ' '.join(self.data[self.data['hotel_name'] == hotel_name]['review_text'])
        else:
            text = ' '.join(self.data['review_text'])
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        if hotel_name:
            ax.set_title(f'Word Cloud for {hotel_name}')
        else:
            ax.set_title('Word Cloud for All Reviews')
        return fig

    def plot_rating_over_time(self, hotel_name: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a line plot of average rating over time.
        
        Args:
        hotel_name (str, optional): If provided, plot for specific hotel.
        
        Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and Axes objects of the plot.
        """
        if 'review_date' not in self.data.columns:
            raise ValueError("Dataset does not contain 'review_date' column")
        
        self.data['review_date'] = pd.to_datetime(self.data['review_date'])
        
        if hotel_name:
            data = self.data[self.data['hotel_name'] == hotel_name]
        else:
            data = self.data
        
        daily_avg = data.groupby('review_date')['rating'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        daily_avg.plot(ax=ax)
        ax.set_title(f'Average Rating Over Time {"for " + hotel_name if hotel_name else ""}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Rating')
        plt.tight_layout()
        return fig, ax
def plot_top_hotels(self, n=10):
        top_hotels = self.data.groupby('hotel_name')['rating'].mean().nlargest(n)
        fig, ax = plt.subplots(figsize=(12, 6))
        top_hotels.plot(kind='bar', ax=ax)
        ax.set_title(f'Top {n} Hotels by Average Rating')
        ax.set_xlabel('Hotel')
        ax.set_ylabel('Average Rating')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig, ax

# Example usage (if running this file directly)
if __name__ == "__main__":
    data = pd.read_csv('reviews.csv')
    processor = FeedbackProcessor(data)
    collected_data = processor.collect_feedback()
    report = processor.report_feedback(collected_data)
    print(report)
    # Generate and save plots
    fig, ax = processor.plot_rating_distribution()
    fig.savefig('rating_distribution.png')
    
    fig, ax = processor.plot_sentiment_distribution()
    fig.savefig('sentiment_distribution.png')
    
    fig, ax = processor.plot_top_hotels()
    fig.savefig('top_hotels.png')
    
    fig, ax = processor.plot_rating_vs_sentiment()
    fig.savefig('rating_vs_sentiment.png')
    
    fig = processor.plot_word_cloud()
    fig.savefig('word_cloud.png')
    
    fig, ax = processor.plot_rating_over_time()
    fig.savefig('rating_over_time.png')