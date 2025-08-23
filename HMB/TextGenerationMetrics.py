# Import all required libraries for metrics computation.
import torch, re, nltk, textstat, sys  # Core and NLP libraries.
import numpy as np  # For numerical operations.
import torch.nn as nn  # For deep learning metrics (if needed).
from rouge import Rouge  # For ROUGE metric computation.
from collections import Counter  # For token counting.
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # BLEU metric.
from nltk.translate.chrf_score import sentence_chrf  # CHRF metric.
from sklearn.metrics import accuracy_score, f1_score  # For accuracy and F1 metrics.

from HMB.Initializations import IncreaseSysRecursionLimit, DownloadNLTKPackages

# Increase system recursion limit to handle deep recursion in metrics calculations.
IncreaseSysRecursionLimit(10 ** 6)
# Download necessary NLTK packages for tokenization and other NLP tasks.
DownloadNLTKPackages()


# Define the TextGenerationMetrics class for evaluating text generation models.
class TextGenerationMetrics(object):
  r'''
  Encapsulates a comprehensive suite of text generation evaluation metrics for NLP tasks.
  Includes BLEU, ROUGE, METEOR, Edit Distance, Jaccard, Perplexity, F1, CHRF, and more.
  '''

  def __init__(self, tokenizer=None):
    '''
    Initializes the metrics class with an optional tokenizer.
    '''

    # Store the tokenizer if provided.
    self.tokenizer = tokenizer
    # Initialize ROUGE metric.
    self.rouge = Rouge()
    # Smoothing function for BLEU score.
    self.smoothing = SmoothingFunction().method1

  def CalculateBLEU(self, generatedText, referenceText, weights=(0.25, 0.25, 0.25, 0.25)):
    r'''
    Calculates BLEU score for generated text against reference text.
    BLEU measures n-gram precision with optional smoothing.

    .. math::
      BLEU = BP \times \exp\left(\sum_{n=1}^N w_n \times \log (p_n)\right)

    where:
      - :math:`BP` is the brevity penalty.
      - :math:`p_n` is the n-gram precision for n-grams of size n.
      - :math:`w_n` are the weights for each n-gram precision.

    Parameters:
      generatedText (str): Generated text to evaluate.
      referenceText (str): Reference text to compare against.
      weights (tuple): Weights for n-gram precision (default uniform for 1-4 grams).

    Returns:
      float: BLEU score.
    '''

    # Tokenize texts.
    generatedTokens = generatedText.split()
    referenceTokens = referenceText.split()
    # Calculate BLEU score with smoothing.
    bleuScore = sentence_bleu(
      [referenceTokens],  # Reference tokens as a list of lists.
      generatedTokens,  # Generated tokens.
      weights=weights,  # Weights for n-gram precision.
      smoothing_function=self.smoothing,  # Smoothing function for BLEU score.
    )
    # Return BLEU score.
    return bleuScore

  def CalculateROUGE(self, generatedText, referenceText):
    r'''
    Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for generated text against reference text.
    ROUGE measures n-gram recall and longest common subsequence.

    .. math::
      ROUGE_N = \frac{\sum_{gram_n \in ref} \min(count_{gen}(gram_n), count_{ref}(gram_n))}{\sum_{gram_n \in ref} count_{ref}(gram_n)}

    where:
      - :math:`gram_n` is an n-gram of size n.

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      dict: ROUGE scores.
    '''

    # Clean text for ROUGE calculation.
    def _CleanTextForRouge(text):
      '''Removes extra whitespace and ensures minimum length for ROUGE.'''
      # Remove extra whitespace and ensure minimum length.
      cleaned = re.sub(r"\s+", " ", text.strip())
      # Return cleaned text or a single space if empty.
      return cleaned if (len(cleaned) > 0) else " "

    # Clean texts for ROUGE calculation.
    cleanedGenText = _CleanTextForRouge(generatedText)
    cleanedRefText = _CleanTextForRouge(referenceText)
    # Calculate ROUGE scores.
    scores = self.rouge.get_scores(cleanedGenText, cleanedRefText)[0]
    # Return ROUGE scores as a dictionary.
    return {
      "rouge-1": scores["rouge-1"]["f"],
      "rouge-2": scores["rouge-2"]["f"],
      "rouge-l": scores["rouge-l"]["f"]
    }

  def CalculateMETEOR(self, generatedText, referenceText):
    r'''
    Calculates METEOR score for generated text against reference text.
    METEOR is based on unigram precision, recall, and F1.

    .. math::
      METEOR = \frac{1}{N} \times \sum_{i=1}^N \max(0, \frac{2 \cdot P_i \cdot R_i}{P_i + R_i})

    where:
      - :math:`P_i` is precision for unigram i.
      - :math:`R_i` is recall for unigram i.

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: METEOR score.
    '''

    # Approximate METEOR using F1 of unigram overlap.
    def _ApproximateMETEOR(genText, refText):
      '''Approximates METEOR using F1 of unigram overlap.'''
      # Get sets of words.
      genWords = set(genText.lower().split())
      refWords = set(refText.lower().split())
      # Handle empty cases.
      if (len(genWords) == 0 and len(refWords) == 0):
        return 1.0
      if (len(genWords) == 0 or len(refWords) == 0):
        return 0.0
      # Calculate intersection, precision, recall.
      intersection = genWords.intersection(refWords)
      precision = len(intersection) / len(genWords)
      recall = len(intersection) / len(refWords)
      # Handle zero division.
      if (precision + recall == 0):
        return 0.0
      # Calculate F1 score.
      f1 = 2.0 * (precision * recall) / (precision + recall)
      # Return F1 score as METEOR approximation.
      return f1

    # Return METEOR score.
    return _ApproximateMETEOR(generatedText, referenceText)

  def CalculateEditDistance(self, generatedText, referenceText):
    r'''
    Calculates normalized edit distance similarity between generated and reference text.
    Edit distance is the minimum number of operations to transform one text into another.

    .. math::
      Sim = 1 - \frac{D_{lev}(gen, ref)}{\max(|gen|, |ref|)}

    where:
      - :math:`D_{lev}(gen, ref)` is the Levenshtein distance between generated and reference text.
      - :math:`|gen|` and :math:`|ref|` are the lengths of generated and reference text.

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: Similarity score (1.0 identical, 0.0 completely different).
    '''

    # Tokenize texts.
    genTokens = generatedText.split()
    refTokens = referenceText.split()
    # Dynamic programming for edit distance.
    m, n = len(genTokens), len(refTokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize base cases.
    for i in range(m + 1):
      dp[i][0] = i
    for j in range(n + 1):
      dp[0][j] = j
    # Fill DP table.
    for i in range(1, m + 1):
      for j in range(1, n + 1):
        # Check if tokens match.
        if (genTokens[i - 1] == refTokens[j - 1]):
          dp[i][j] = dp[i - 1][j - 1]
        else:
          dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    # Normalize by maximum length.
    maxLen = max(m, n)
    # Handle empty case.
    if (maxLen == 0):
      return 1.0
    # Calculate normalized distance.
    normalizedDistance = dp[m][n] / maxLen
    # Return similarity instead of distance.
    return 1.0 - normalizedDistance

  def CalculateSemanticSimilarity(self, generatedText, referenceText):
    r'''
    Calculates Jaccard similarity based on word overlap between generated and reference text.

    .. math::
      Jaccard = \frac{|gen \cap ref|}{|gen \cup ref|}

    where:
      - :math:`|gen \cap ref|` is the size of the intersection of words.
      - :math:`|gen \cup ref|` is the size of the union of words.

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: Jaccard similarity score.
    '''

    # Get sets of words.
    genWords = set(generatedText.lower().split())
    refWords = set(referenceText.lower().split())
    # Handle empty cases.
    if (len(genWords) == 0 and len(refWords) == 0):
      return 1.0
    if (len(genWords) == 0 or len(refWords) == 0):
      return 0.0
    # Jaccard similarity.
    intersection = len(genWords.intersection(refWords))
    union = len(genWords.union(refWords))
    jaccard = intersection / union if (union > 0) else 0.0
    # Return Jaccard similarity.
    return jaccard

  def CalculateLengthRatio(self, generatedText, referenceText):
    r'''
    Calculates the ratio of generated text length to reference text length.

    .. math::
      LengthRatio = \frac{|gen|}{|ref|}

    where:
      - :math:`|gen|` is the length of the generated text.
      - :math:`|ref|` is the length of the reference text.

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: Length ratio (1.0 if equal, < 1.0 if generated is shorter, > 1.0 if longer).
    '''

    # Get lengths.
    genLen = len(generatedText.split())
    refLen = len(referenceText.split())
    # Handle division by zero.
    if (refLen == 0):
      return 0.0 if (genLen == 0) else float("inf")
    # Return length ratio.
    return genLen / refLen

  def CalculatePerplexity(self, generatedTokens, referenceTokens):
    r'''
    Calculates the perplexity of generated tokens against reference tokens.
    Perplexity measures how well the generated text predicts the reference text.

    .. math::
      Perplexity = \exp\left(-\frac{1}{N} \times \sum_{i=1}^N \log P(token_i)\right)

    where:
      - :math:`N` is the number of generated tokens.
      - :math:`P(token_i)` is the probability of token i in the reference text.

    Parameters:
      generatedTokens (list): Tokens generated by the model.
      referenceTokens (list): Tokens in the reference text.

    Returns:
      float: Perplexity score.
    '''

    # Handle empty reference.
    if (len(referenceTokens) == 0):
      return float("inf")
    # Count occurrences of each token in the reference text.
    refCounter = Counter(referenceTokens)
    totalRefTokens = len(referenceTokens)
    # Calculate probabilities for each generated token.
    logProbs = []
    for token in generatedTokens:
      prob = refCounter[token] / totalRefTokens if (token in refCounter) else 1e-10
      # Add log probability if prob > 0.
      if (prob > 0):
        logProbs.append(np.log(prob))
      else:
        return float("inf")
    # Handle empty logProbs.
    if (len(logProbs) == 0):
      return float("inf")
    avgLogProb = np.mean(logProbs)
    perplexity = np.exp(-avgLogProb)
    perplexity = perplexity / len(generatedTokens)
    # Return perplexity score.
    return perplexity

  def CalculateAccuracy(self, generatedText, referenceText):
    r'''
    Calculates accuracy of generated text against reference text.
    Accuracy is the proportion of matching tokens.

    .. math::
      Accuracy = \frac{|\{gen \cap ref\}|}{|ref|}

    where:
      - :math:`|\{gen \cap ref\}|` is the number of matching tokens.
      - :math:`|ref|` is the total number of tokens in the reference text.

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: Accuracy score (1.0 perfect match, 0.0 no match).
    '''

    # Tokenize texts.
    genTokens = generatedText.split()
    refTokens = referenceText.split()
    # Handle empty reference.
    if (len(refTokens) == 0):
      return 1.0 if (len(genTokens) == 0) else 0.0
    # Calculate accuracy as the proportion of matching tokens.
    correct = sum(1 for g, r in zip(genTokens, refTokens) if (g == r))
    accuracy = correct / len(refTokens)
    # Return accuracy score.
    return accuracy

  def CalculateF1Score(self, generatedText, referenceText):
    r'''
    Calculates F1 score of generated text against reference text.
    F1 is the harmonic mean of precision and recall.

    .. math::
      F1 = \frac{2 \times P \times R}{P+R}

    where:
      - :math:`P` is precision (proportion of generated tokens in reference).
      - :math:`R` is recall (proportion of reference tokens in generated).

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: F1 score.
    '''

    # Tokenize texts.
    genTokens = generatedText.split()
    refTokens = referenceText.split()
    # Handle empty cases.
    if (len(genTokens) == 0 or len(refTokens) == 0):
      return 0.0
    # Count matches.
    matches = sum(1 for g in genTokens if (g in refTokens))
    precision = matches / len(genTokens)
    recall = matches / len(refTokens)
    # Handle zero division.
    if ((precision + recall) == 0):
      return 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall)
    # Return F1 score.
    return f1

  def CalculateCHRF(self, generatedText, referenceText):
    r'''
    Calculates CHRF score for generated text against reference text.
    CHRF is the character n-gram F-score.

    .. math::
      CHRF = \frac{2 \cdot P \cdot R}{P + R}

    where:
      - :math:`P` is precision (proportion of n-grams in generated text).
      - :math:`R` is recall (proportion of n-grams in reference text).

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: CHRF score.
    '''

    # Calculate CHRF score.
    chrfScore = sentence_chrf(generatedText, referenceText)
    # Return CHRF score.
    return chrfScore

  def CalculateRepetitionRate(self, generatedText, n=3):
    r'''
    Calculates the repetition rate of n-grams in the generated text.
    Repetition rate is the proportion of n-grams that are repeated.

    .. math::
      RepetitionRate = 1 - \frac{|unique\ ngrams|}{|total\ ngrams|}

    where:
      - :math:`|unique\ ngrams|` is the number of unique n-grams.
      - :math:`|total\ ngrams|` is the total number of n-grams.

    Parameters:
      generatedText (str): Generated text.
      n (int): Size of n-grams to consider (default 3).

    Returns:
      float: Repetition rate (0.0 no repetition, 1.0 all n-grams repeated).
    '''

    # Tokenize text.
    tokens = generatedText.split()
    # Handle short text.
    if (len(tokens) < n):
      return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    uniqueNgrams = set(ngrams)
    # Handle empty ngrams.
    if (len(ngrams) == 0):
      return 0.0
    repetitionRate = 1.0 - (len(uniqueNgrams) / len(ngrams))
    # Return repetition rate.
    return repetitionRate

  def CalculateLexicalDiversity(self, generatedText):
    r'''
    Calculates lexical diversity of the generated text.
    Lexical diversity is the ratio of unique words to total words.

    .. math::
      LexicalDiversity = \frac{|unique\ words|}{|total\ words|}

    where:
      - :math:`|unique\ words|` is the number of unique words.
      - :math:`|total\ words|` is the total number of words.

    Parameters:
      generatedText (str): Generated text.

    Returns:
      float: Lexical diversity (0.0 no diversity, 1.0 all unique words).
    '''

    # Tokenize text.
    tokens = generatedText.lower().split()
    # Handle empty text.
    if (len(tokens) == 0):
      return 0.0
    uniqueTokens = set(tokens)
    # Return lexical diversity.
    return len(uniqueTokens) / len(tokens)

  def CalculateReadabilityScore(self, generatedText):
    r'''
    Calculates the Flesch-Kincaid grade level of the generated text.
    Lower score = easier readability.

    .. math::
      FleschKincaid = 0.39 \cdot \frac{total\ words}{total\ sentences} + 11.8 \cdot \frac{total\ syllables}{total\ words} - 15.59

    where:
      - :math:`total\ words` is the number of words.
      - :math:`total\ sentences` is the number of sentences.
      - :math:`total\ syllables` is the number of syllables.

    Parameters:
      generatedText (str): Generated text.

    Returns:
      float: Flesch-Kincaid grade level (lower is easier).
    '''

    # Compute readability score.
    return textstat.flesch_kincaid_grade(generatedText)

  def CalculateInformationDensity(self, generatedText):
    r'''
    Calculates information density of the generated text.
    Information density is the ratio of content words to total words.

    .. math::
      InformationDensity = \frac{|content\ words|}{|tokens|}

    where:
      - :math:`|content\ words|` is the number of content words (nouns, verbs, adjectives, adverbs).
      - :math:`|tokens|` is the total number of tokens.

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text (not used here, but could be for comparison).

    Returns:
      float: Information density (0.0 no content words, 1.0 all content words).
    '''

    # Tokenize and POS tag.
    tokens = nltk.word_tokenize(generatedText.lower())
    posTags = nltk.pos_tag(tokens)
    # Content words: nouns, verbs, adjectives, adverbs.
    contentWords = [word for word, pos in posTags if (pos.startswith(('NN', 'VB', 'JJ', 'RB')))]
    # Handle empty tokens.
    if (len(tokens) == 0):
      return 0.0
    # Return information density.
    return len(contentWords) / len(tokens)

  def CalculateHallucinationRate(self, generatedText, referenceText):
    r'''
    Calculates hallucination rate of generated text against reference text.
    Hallucination rate is the proportion of words in the generated text not in the reference text.

    .. math::
      HallucinationRate = \frac{|gen \setminus ref|}{|gen|}

    where:
      - :math:`|gen \setminus ref|` is the number of words in the generated text not in the reference text.
      - :math:`|gen|` is the total number of words in the generated text (not counting hallucinations).

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: Hallucination rate (0.0 no hallucination, 1.0 all hallucination).
    '''

    # Get sets of words.
    genWords = set(generatedText.lower().split())
    refWords = set(referenceText.lower().split())
    # Handle empty generated text.
    if (len(genWords) == 0):
      return 0.0
    hallucinatedWords = genWords - refWords
    # Return hallucination rate.
    return len(hallucinatedWords) / len(genWords)

  def CalculateOmissionRate(self, generatedText, referenceText):
    r'''
    Calculates omission rate of generated text against reference text.
    Omission rate is the proportion of words in the reference text not in the generated text.

    .. math::
      OmissionRate = \frac{|ref \setminus gen|}{|ref|}

    where:
      - :math:`|ref \setminus gen|` is the number of words in the reference text not in the generated text.
      - :math:`|ref|` is the total number of words in the reference text.

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: Omission rate (0.0 no omissions, 1.0 all words omitted).
    '''

    # Get sets of words.
    genWords = set(generatedText.lower().split())
    refWords = set(referenceText.lower().split())
    # Handle empty reference text.
    if (len(refWords) == 0):
      return 0.0
    omittedWords = refWords - genWords
    # Return omission rate.
    return len(omittedWords) / len(refWords)

  def CalculateFactualityScore(self, generatedText, referenceText):
    r'''
    Calculates factuality score based on hallucination and omission rates.
    Factuality combines precision and recall of content overlap.
    Factuality score is 1.0 for perfect factuality (no hallucinations or omissions).

    .. math::
      F1 = \frac{2 \times P \times R}{P+R}

    where:
      - :math:`P` is precision (1.0 - hallucination rate).
      - :math:`R` is recall (1.0 - omission rate).

    Parameters:
      generatedText (str): Generated text.
      referenceText (str): Reference text.

    Returns:
      float: Factuality score (1.0 perfect factuality).
    '''

    # Calculate hallucination and omission rates.
    hallucinationRate = self.CalculateHallucinationRate(generatedText, referenceText)
    omissionRate = self.CalculateOmissionRate(generatedText, referenceText)
    precision = 1.0 - hallucinationRate
    recall = 1.0 - omissionRate
    # Handle zero division.
    if (precision + recall == 0):
      return 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall)
    # Return factuality score.
    return f1

  def CalculateAllMetrics(self, generatedText, referenceText):
    r'''
    Calculates all metrics for generated text against reference text.
    Returns a dictionary of all computed metrics.

    Parameters:
      generatedText (str): Generated text to evaluate.
      referenceText (str): Reference text to compare against.

    Returns:
      dict: Dictionary containing all computed metrics.
    '''

    # Compute all metrics and return as dictionary.
    return {
      "BLEU"              : self.CalculateBLEU(generatedText, referenceText),
      "ROUGE"             : self.CalculateROUGE(generatedText, referenceText),
      "METEOR"            : self.CalculateMETEOR(generatedText, referenceText),
      "EditDistance"      : self.CalculateEditDistance(generatedText, referenceText),
      "SemanticSimilarity": self.CalculateSemanticSimilarity(generatedText, referenceText),
      "LengthRatio"       : self.CalculateLengthRatio(generatedText, referenceText),
      "Perplexity"        : self.CalculatePerplexity(generatedText.split(), referenceText.split()),
      "Accuracy"          : self.CalculateAccuracy(generatedText, referenceText),
      "F1Score"           : self.CalculateF1Score(generatedText, referenceText),
      "CHRF"              : self.CalculateCHRF(generatedText, referenceText),
      "RepetitionRate"    : self.CalculateRepetitionRate(generatedText),
      "LexicalDiversity"  : self.CalculateLexicalDiversity(generatedText),
      "ReadabilityScore"  : self.CalculateReadabilityScore(generatedText),
      "InformationDensity": self.CalculateInformationDensity(generatedText),
      "HallucinationRate" : self.CalculateHallucinationRate(generatedText, referenceText),
      "OmissionRate"      : self.CalculateOmissionRate(generatedText, referenceText),
      "FactualityScore"   : self.CalculateFactualityScore(generatedText, referenceText),
    }


# Main block for running example metric calculations.
if __name__ == "__main__":
  # Initialize metrics calculator.
  metrics = TextGenerationMetrics()  # Create an instance of the metrics class.
  # Define example pairs of generated and reference texts for evaluation.
  results = [
    ("Invasive ductal carcinoma grade 2", "Invasive ductal carcinoma grade 2"),  # Identical texts.
    ("Invasive ductal carcinoma grade 2", "Invasive ductal carcinoma grade 3"),  # Slightly different texts.
    ("The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog."),
    # Identical texts.
  ]
  # Iterate over each pair and compute all metrics.
  for genText, refText in results:
    # Compute BLEU score.
    bleuScore = metrics.CalculateBLEU(genText, refText)
    # Compute ROUGE scores.
    rougeScores = metrics.CalculateROUGE(genText, refText)
    # Compute METEOR score.
    meteorScore = metrics.CalculateMETEOR(genText, refText)
    # Compute edit distance similarity.
    editDistance = metrics.CalculateEditDistance(genText, refText)
    # Compute semantic similarity.
    semanticSimilarity = metrics.CalculateSemanticSimilarity(genText, refText)
    # Compute length ratio.
    lengthRatio = metrics.CalculateLengthRatio(genText, refText)
    # Compute perplexity.
    perplexity = metrics.CalculatePerplexity(genText.split(), refText.split())
    # Compute accuracy.
    accuracy = metrics.CalculateAccuracy(genText, refText)
    # Compute F1 score.
    f1Score = metrics.CalculateF1Score(genText, refText)
    # Compute CHRF score.
    chrfScore = metrics.CalculateCHRF(genText, refText)
    # Compute repetition rate.
    repetitionRate = metrics.CalculateRepetitionRate(genText)
    # Compute lexical diversity.
    lexicalDiversity = metrics.CalculateLexicalDiversity(genText)
    # Compute readability score.
    readabilityScore = metrics.CalculateReadabilityScore(genText)
    # Compute information density.
    informationDensity = metrics.CalculateInformationDensity(genText)
    # Compute hallucination rate.
    hallucinationRate = metrics.CalculateHallucinationRate(genText, refText)
    # Compute omission rate.
    omissionRate = metrics.CalculateOmissionRate(genText, refText)
    # Compute factuality score.
    factualityScore = metrics.CalculateFactualityScore(genText, refText)

    # Print all computed metrics for the current text pair.
    print(f"BLEU Score: {bleuScore:.4f}")  # BLEU metric output.
    print(
      f"ROUGE-1: {rougeScores['rouge-1']:.4f}, ROUGE-2: {rougeScores['rouge-2']:.4f}, "
      f"ROUGE-L: {rougeScores['rouge-l']:.4f}"
    )  # ROUGE metrics output.
    print(f"METEOR Score: {meteorScore:.4f}")  # METEOR metric output.
    print(f"Edit Distance Similarity: {editDistance:.4f}")  # Edit distance output.
    print(f"Semantic Similarity: {semanticSimilarity:.4f}")  # Semantic similarity output.
    print(f"Length Ratio: {lengthRatio:.4f}")  # Length ratio output.
    print(f"Perplexity: {perplexity:.4f}")  # Perplexity output.
    print(f"Accuracy: {accuracy:.4f}")  # Accuracy output.
    print(f"F1 Score: {f1Score:.4f}")  # F1 score output.
    print(f"CHRF Score: {chrfScore:.4f}")  # CHRF metric output.
    print(f"Repetition Rate: {repetitionRate:.4f}")  # Repetition rate output.
    print(f"Lexical Diversity: {lexicalDiversity:.4f}")  # Lexical diversity output.
    print(f"Readability Score: {readabilityScore:.4f}")  # Readability score output.
    print(f"Information Density: {informationDensity:.4f}")  # Information density output.
    print(f"Hallucination Rate: {hallucinationRate:.4f}")  # Hallucination rate output.
    print(f"Omission Rate: {omissionRate:.4f}")  # Omission rate output.
    print(f"Factuality Score: {factualityScore:.4f}")  # Factuality score output.
    print("=" * 80)  # Separator for readability.
