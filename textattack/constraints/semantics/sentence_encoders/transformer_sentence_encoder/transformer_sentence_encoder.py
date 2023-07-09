"""
Transformer sentence encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from sentence_transformers import SentenceTransformer, util

class TransformerSentenceEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Transformer Sentence
    Encoder."""

    def __init__(self, threshold=0.8, large=True, metric="cosine", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        if large:
            model_name = 'all-distilroberta-v1'
        else:
            model_name = 'all-MiniLM-L6-v2'

        self.model = SentenceTransformer(model_name, device='cuda')

    def encode(self, sentences):
        encoding = self.model.encode(sentences,show_progress_bar=False)

        if isinstance(encoding, dict):
            encoding = encoding["outputs"]

        return encoding

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = None
