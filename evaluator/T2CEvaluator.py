# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
import json
import os
from typing import Dict

class T2CEvaluator:
    def __init__(self, max_order = 4, smooth = True,  verbose = False):
        self.verbose = verbose
        self.max_order = max_order
        self.smooth = smooth
        
    
    @classmethod
    def _preprocess_answers(cls, s):
        s = s.replace('\n', ' ')

        while '  ' in s:
            s = s.replace('  ', ' ')

        while s and s[-1]==' ':
            s = s[:-1]

        return s

    def calculate_metrics(self, fn_answers, fn_predictions) -> Dict:
        """
            return EM score, bleu metrics, and others
        """
        res  = {}
        preds = open(fn_predictions, "r", encoding='utf-8').readlines()
        gts   = open(fn_answers, "r", encoding='utf-8').readlines()

        assert len(preds) == len(gts), f"Samples of predictions and answers are not equal, {len(preds)}: {len(gts)}"

        total = len(gts)
        EM = 0.0

        translation_corpus = []
        reference_corpus = []

        #read the data in correct format
        for pred, gt in zip(preds, gts):
            pred = pred.strip()
            gt = json.loads(gt)["code"]
            reference_corpus.append([gt.split(' ')])
            translation_corpus.append(T2CEvaluator._preprocess_answers(pred).split(' '))
            if pred.split() == gt.split():
                EM += 1

        res['EM'] = EM

        # 3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
        #  precisions and brevity penalty.

        bleu_score, precisions, bp, ratio, translation_length, reference_length = self.compute_bleu(reference_corpus = reference_corpus, 
                                                                                         translation_corpus= translation_corpus, 
                                                                                         max_order = self.max_order,
                                                                                         smooth = self.smooth
                                                                                        )
        res['BLEU'] = bleu_score
        res['brevity_penalty'] = bp
        res['ratio'] = ratio
        res['translation_length'] = translation_length
        res['reference_length'] = reference_length
        for i in range(len(precisions)):
            res[f'precisions_{i}'] = precisions[i]

        if self.verbose:
            print(f"INFO:__main__:BLEU: {round(100 * bleu_score,2)}, EM: {round(EM/total*100, 2)}") 

        return res
    
    @classmethod
    def _get_ngrams(cls, segment, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.
        Args:
          segment: text segment from which n-grams will be extracted.
          max_order: maximum length in tokens of the n-grams returned by this
              methods.
        Returns:
          The Counter containing all n-grams upto max_order in segment
          with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    @classmethod
    def compute_bleu(cls, reference_corpus, translation_corpus, max_order=4,
                     smooth=False):
        """Computes BLEU score of translated segments against one or more references.
        Args:
          reference_corpus: list of lists of references for each translation. Each
              reference should be tokenized into a list of tokens.
          translation_corpus: list of translations to score. Each translation
              should be tokenized into a list of tokens.
          max_order: Maximum n-gram order to use when computing BLEU score.
          smooth: Whether or not to apply Lin et al. 2004 smoothing.
        Returns:
          3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
          precisions and brevity penalty.
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0
        for (references, translation) in zip(reference_corpus,
                                             translation_corpus):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= cls._get_ngrams(reference, max_order)
            translation_ngram_counts = cls._get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram)-1] += overlap[ngram]
            for order in range(1, max_order+1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order-1] += possible_matches

        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) /
                                 (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                                     possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(translation_length) / reference_length

        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp

        return (bleu, precisions, bp, ratio, translation_length, reference_length)