# Description: This file contains the functions used to evaluate the model.
# Author: Ahmed Magdi and Ahmed Magdy
# Last Modified: 23-06-2023
# ------------------------------------------------------------------------------

import string


def calculate_wer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()

    reference = reference.translate(str.maketrans('', '', string.punctuation))
    hypothesis = hypothesis.translate(
        str.maketrans('', '', string.punctuation))

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Counting the number of substitutions, deletions, and insertions
    # A substitution occurs when a word in the recognized or translated text is different from the corresponding word in the reference text.
    substitutions = sum(1 for ref, hyp in zip(
        ref_words, hyp_words) if ref != hyp)
    # A deletion occurs when a word in the reference text is missing from the recognized or translated text.
    deletions = len(ref_words) - len(hyp_words)
    # is the number of word insertions. An insertion occurs when an extra word is present in the recognized or translated text that is not in the reference text.
    insertions = len(hyp_words) - len(ref_words)
    # Total number of words in the reference text
    total_words = len(ref_words)
    # Calculating the Word Error Rate (WER) using the following Equation
    wer = (substitutions + deletions + insertions) / total_words
    return wer * 100
