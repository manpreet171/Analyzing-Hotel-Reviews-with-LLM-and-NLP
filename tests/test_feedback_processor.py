import unittest
import pandas as pd
from src.feedback_processor import FeedbackProcessor

class TestFeedbackProcessor(unittest.TestCase):
    def setUp(self):
        data = {
            'hotel_name': ['Hotel A', 'Hotel B', 'Hotel A', 'Hotel B'],
            'review_text': [
                'Great stay, very comfortable!',
                'Not so good, had some issues.',
                'Excellent service and location.',
                'Terrible experience, will not return.'
            ],
            'rating': [9, 4, 10, 2],
            'nationality': ['USA', 'UK', 'USA', 'UK']
        }
        self.df = pd.DataFrame(data)
        self.processor = FeedbackProcessor(self.df)

    def test_collect_feedback(self):
        collected_data = self.processor.collect_feedback()
        self.assertTrue('sentiment' in collected_data.columns)

    def test_report_feedback(self):
        collected_data = self.processor.collect_feedback()
        report = self.processor.report_feedback(collected_data)
        self.assertEqual(len(report), 2)

    def test_get_most_common_themes(self):
        themes = self.processor.get_most_common_themes()
        self.assertTrue('great' in themes.index)

if __name__ == '__main__':
    unittest.main()
