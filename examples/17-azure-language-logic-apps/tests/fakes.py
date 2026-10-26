"""Minimal stand-ins for the result objects of azure-ai-textanalytics."""

from types import SimpleNamespace as NS


class FakeLanguageClient:
    def __init__(self, language="en", sentiment="neutral", negative=0.1, phrases=(), entities=(), error=None):
        self.language, self.sentiment, self.negative = language, sentiment, negative
        self.phrases, self.entities, self.error = list(phrases), list(entities), error
        self.calls = []

    def _ok(self, **fields):
        if self.error:
            return [NS(is_error=True, error=NS(code=self.error, message="boom"))]
        return [NS(is_error=False, **fields)]

    def detect_language(self, docs, **kw):
        self.calls.append(("detect_language", kw))
        return self._ok(primary_language=NS(iso6391_name=self.language, name=self.language, confidence_score=0.99))

    def analyze_sentiment(self, docs, **kw):
        self.calls.append(("analyze_sentiment", kw))
        scores = NS(positive=max(0.0, 0.9 - self.negative), neutral=0.1, negative=self.negative)
        return self._ok(sentiment=self.sentiment, confidence_scores=scores)

    def recognize_entities(self, docs, **kw):
        self.calls.append(("recognize_entities", kw))
        return self._ok(entities=[NS(text=t, category=c, confidence_score=0.9) for t, c in self.entities])

    def extract_key_phrases(self, docs, **kw):
        self.calls.append(("extract_key_phrases", kw))
        return self._ok(key_phrases=self.phrases)


class FakeChatClient:
    def __init__(self, reply="Draft reply"):
        self.reply, self.requests = reply, []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return NS(choices=[NS(message=NS(content=self.reply))])
