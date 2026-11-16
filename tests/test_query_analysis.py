import pytest

from cgraph.query_analysis import E3_DESCRIPTORS, QueryAnalyzer


class FakeExtractor:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, text, threshold):
        self.calls.append((text, threshold))
        return self.predictions


def test_unicode_spans_deduplication_and_labels():
    text = "Does β-cell use RNA?"
    label = E3_DESCRIPTORS["OBJECT"]
    fake = FakeExtractor([{"start": 5, "end": 11, "text": "β-cell", "label": label, "score": 0.4},
                          {"start": 5, "end": 11, "text": "β-cell", "label": label, "score": 0.8}])
    analyzer = QueryAnalyzer(fake, threshold=0.2)
    result = analyzer.analyze(text)
    assert result["mentions"][0]["surface_text"] == text[5:11]
    assert result["mentions"][0]["score"] == 0.8
    assert result["mentions"][0]["label"] == "OBJECT"
    assert result == analyzer.analyze(text)
    assert fake.calls[0] == (text, 0.2)


def test_empty_and_no_ner():
    fake = FakeExtractor([])
    analyzer = QueryAnalyzer(fake)
    assert analyzer.analyze("  ")["warnings"] == ["empty_question"]
    assert fake.calls == []
    assert analyzer.analyze("What happens?")["warnings"] == ["empty_ner"]


def test_invalid_span_is_rejected():
    fake = FakeExtractor([{"start": 0, "end": 1, "text": "z", "label": E3_DESCRIPTORS["OBJECT"], "score": 0.9}])
    with pytest.raises(ValueError):
        QueryAnalyzer(fake).analyze("abc")
