"""
TextFooler (Is BERT Really Robust?)
===================================================
A Strong Baseline for Natural Language Attack on Text Classification and Entailment)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import TransformerSentenceEncoder
from textattack.transformations import WordSwapEmbedding
from textattack.transformations.word_swaps.word_swap_masked_lm import WordSwapMaskedLM
from textattack.transformations import WordSwapWordNet
import time
import re
from textattack.shared.utils import words_from_text
from textattack.shared.attacked_text import AttackedText
import numpy as np
import random

def get_bae_info(tagger_type, tagset):
    transformation = WordSwapMaskedLM(
            masked_language_model="roberta-large",
            method="bae", max_candidates=40, min_confidence=5e-4
        )
    constraints = [
        PartOfSpeech(tagger_type=tagger_type, tagset=tagset, allow_verb_noun_swap=False),
    ]
    return transformation, constraints

class SynonymFunc:
    def __init__(self, method='bae'):
        self.use_stop_word = False if 'nosw' in method else True
        self.stop_word_constraint = StopwordModification()
        if method == 'bae' or method == 'bae_nosw':
            self.transformation, self.constraints = get_bae_info(tagger_type='nltk',tagset='universal')
        else:
            raise NotImplementedError
        self.method = method

    def get_sentence_candidates(self, sentence_, num_synonym_candidates=-1):
        if type(sentence_) == str:
            return_str = True
            sentence = AttackedText(sentence_)
        else:
            return_str = False
            sentence = sentence_
        idx_list = self.stop_word_constraint._get_modifiable_indices(sentence) if self.use_stop_word else list(range(sentence.num_words))
        if num_synonym_candidates != -1:
            # sample num_synonym_candidates words from idx_list and get candidates for sampled words
            if len(idx_list) > num_synonym_candidates:
                idx_list = random.sample(idx_list, num_synonym_candidates)

        sent_candidates = self.transformation._get_transformations(sentence, idx_list)
        for C in self.constraints:
            sent_candidates = C._check_constraint_many(sent_candidates, sentence)       
        sent_candidates = list(sent_candidates)
        sent_candidates.insert(0, sentence)
        if return_str:
            return_texts = [text.text for text in sent_candidates]
            del sent_candidates, sentence
            return return_texts
        else:
            return sent_candidates
