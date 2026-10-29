"""Unit tests for the emotion-detection package."""

import unittest

from EmotionDetection.emotion_detection import emotion_detector


class EmotionDetectionTests(unittest.TestCase):
    """Test dominant-emotion results for representative statements."""

    def test_joy(self):
        """Check that a glad statement is classified as joy."""
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger(self):
        """Check that a mad statement is classified as anger."""
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust(self):
        """Check that a disgusted statement is classified as disgust."""
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_sadness(self):
        """Check that a sad statement is classified as sadness."""
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_fear(self):
        """Check that an afraid statement is classified as fear."""
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
