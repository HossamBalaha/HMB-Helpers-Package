# Import all required libraries for metrics computation.
import torch, re, math
import numpy as np  # For numerical operations.

from HMB.Initializations import IncreaseSysRecursionLimit, DownloadNLTKPackages

# Increase system recursion limit to handle deep recursion in metrics calculations.
IncreaseSysRecursionLimit(10 ** 6)
# Download necessary NLTK packages for tokenization and other NLP tasks.
DownloadNLTKPackages()


class TextGenerationMetrics(object):
  r'''
  Encapsulates a comprehensive suite of text generation evaluation metrics for NLP tasks.
  Includes BLEU, ROUGE, METEOR, Edit Distance, Jaccard, Perplexity, F1, CHRF, and more.

  Examples
  --------
  .. code-block:: python

    from HMB.TextGenerationMetrics import TextGenerationMetrics

    # Initialize the metrics object.
    metrics = TextGenerationMetrics()

    # Example usage for BLEU score.
    generatedText = "The cat sat on the mat."
    referenceText = "The cat is sitting on the mat."

    bleuScore = metrics.CalculateBLEU(generatedText, referenceText)
    print(f"BLEU Score: {bleuScore:.4f}")
  '''

  def __init__(self, tokenizer=None):
    '''
    Initializes the metrics class with an optional tokenizer.

    Parameters:
      tokenizer (Optional): A tokenizer object for text preprocessing. If not provided, default tokenization methods will be used.
    '''

    from rouge import Rouge  # For ROUGE metric computation.
    from nltk.translate.bleu_score import SmoothingFunction

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

    from nltk.translate.bleu_score import sentence_bleu

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

  def CalculateJaccardSimilarity(self, generatedText, referenceText):
    r'''
    Calculates Jaccard similarity based on word overlap between generated and reference text.

    This function employs a purely statistical, lexical approach. It treats text as a "bag of words"
    and calculates the ratio of shared unique words to the total unique words across both texts.
    It is highly sensitive to exact word matches and completely ignores word order, syntax, and
    semantic meaning. Consequently, it will assign a low score to texts that use different synonyms
    to express the exact same meaning, unlike embedding-based approaches.

    Architectural Correlation Analysis:
      - CalculateJaccardSimilarity (Lexical): High score only if exact words match. Fails on synonyms/paraphrases.
      - CalculateSemanticSimilarity (Embedding): High score if meaning is preserved, regardless of word choice.
      - Conclusion: Use Jaccard for exact duplication checks; use Semantic for meaning preservation.

    .. math::
      Jaccard = \frac{|gen \cap ref|}{|gen \cup ref|}

    where:
      - :math:`|gen \cap ref|` is the size of the intersection of words.
      - :math:`|gen \cup ref|` is the size of the union of words.

    Parameters:
      generatedText (str): The generated text to evaluate.
      referenceText (str): The reference text to compare against.

    Returns:
      float: The Jaccard similarity score between 0.0 and 1.0.

    Examples
    --------
    .. code-block:: python

      from HMB.TextGenerationMetrics import TextGenerationMetrics

      # Initialize the metrics object.
      obj = TextGenerationMetrics()

      # Calculate Jaccard similarity between two texts.
      text1 = "The study was conducted on January 15, 2023, and included 150 participants."
      text2 = "The research took place on the fifteenth of January, 2023, with a total of one hundred fifty subjects."
      similarity = obj.CalculateJaccardSimilarity(text1, text2)
      # Expected output: Jaccard Similarity: 0.1200 (example value, actual may vary).
      print(f"Jaccard Similarity: {similarity:.4f}")
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

  def CalculateSemanticSimilarity(self, semanticModel, text1: str, text2: str) -> float:
    r'''
    Calculates the cosine similarity between two text embeddings using a sentence transformer model.

    This function utilizes a deep learning approach, mapping texts into a high-dimensional semantic
    space where geometric proximity indicates meaning equivalence. Unlike lexical overlap methods,
    it is highly context-aware and robust to paraphrasing, capturing syntactic structures and
    long-range dependencies. It will assign a high score to texts that use completely different
    words but convey the same underlying meaning, making it ideal for evaluating humanized text
    where vocabulary diversity is intentionally increased.

    Architectural Correlation Analysis:
      - CalculateJaccardSimilarity (Lexical): High score only if exact words match. Fails on synonyms/paraphrases.
      - CalculateSemanticSimilarity (Embedding): High score if meaning is preserved, regardless of word choice.
      - Conclusion: Use Jaccard for exact duplication checks; use Semantic for meaning preservation.

    Parameters:
      semanticModel (sentence_transformers.SentenceTransformer): A pre-initialized sentence transformer model.
      text1 (str): The first text string.
      text2 (str): The second text string.

    Returns:
      float: The cosine similarity score between 0.0 and 1.0. Returns 0.05 on failure.

    Examples
    --------
    .. code-block:: python

      from sentence_transformers import SentenceTransformer
      from HMB.TextGenerationMetrics import TextGenerationMetrics

      # Load a pre-trained sentence transformer model.
      model = SentenceTransformer("all-MiniLM-L6-v2")

      # Initialize the metrics object.
      obj = TextGenerationMetrics()

      # Calculate semantic similarity between two texts.
      text1 = "The study was conducted on January 15, 2023, and included 150 participants."
      text2 = "The research took place on the fifteenth of January, 2023, with a total of one hundred fifty subjects."
      similarity = obj.CalculateSemanticSimilarity(model, text1, text2)
      # Expected output: Semantic Similarity: 0.7649 (example value, actual may vary).
      print(f"Semantic Similarity: {similarity:.4f}")
    '''

    # Validate that the semantic model is available for inference.
    if (semanticModel is not None):
      # Attempt to calculate similarity with comprehensive error handling.
      try:
        # Encode the first text into a high-dimensional tensor embedding.
        embedding1 = semanticModel.encode(text1, convert_to_tensor=True)

        # Encode the second text into a high-dimensional tensor embedding.
        embedding2 = semanticModel.encode(text2, convert_to_tensor=True)

        # Ensure the first embedding is 2D for the cosine similarity calculation.
        if (len(embedding1.shape) == 1):
          # Reshape the 1D tensor to 2D by adding a batch dimension.
          embedding1 = embedding1.unsqueeze(0)

        # Ensure the second embedding is 2D for the cosine similarity calculation.
        if (len(embedding2.shape) == 1):
          # Reshape the 1D tensor to 2D by adding a batch dimension.
          embedding2 = embedding2.unsqueeze(0)

        # Calculate the cosine similarity between the two embeddings along the feature dimension.
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1)

        # Extract the similarity value as a standard Python float.
        similarityValue = similarity.item()

        # Return the successfully calculated semantic similarity score.
        return similarityValue

      # Catch any unexpected runtime exceptions during model inference.
      except Exception as e:
        # Print a diagnostic warning containing the exception details.
        print(f"⚠️ Semantic similarity calculation failed: {e}")

        # Return a low fallback score to reject invalid variations.
        return 0.05

    # Return a low fallback score if the model is unavailable to reject variations.
    return 0.05

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
    Calculates the perplexity of generated tokens against reference tokens using a statistical approach.

    Architectural Paradigm & Differences:
    This function employs a statistical, frequency-based approach, relying on empirical token counts
    from a provided reference corpus. It is context-agnostic; it treats tokens as independent events
    (a bag-of-words model) and calculates probabilities based solely on global frequency distributions,
    ignoring word order entirely.

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

    Examples
    --------
    .. code-block:: python

      from HMB.TextGenerationMetrics import TextGenerationMetrics

      # Example reference and generated tokens.
      referenceTokens = "The cat sat on the mat".split()
      generatedTokens = "The cat sat on the mat".split()

      # Initialize the metrics object.
      obj = TextGenerationMetrics()

      # Calculate perplexity.
      perplexity = obj.CalculatePerplexity(generatedTokens, referenceTokens)
      print(f"Statistical Perplexity: {perplexity:.4f}")
    '''

    from collections import Counter  # For token counting.

    # Handle empty reference.
    if (len(referenceTokens) == 0):
      # Return infinity for empty reference.
      return float("inf")

    # Count occurrences of each token in the reference text.
    refCounter = Counter(referenceTokens)

    # Calculate total reference tokens.
    totalRefTokens = len(referenceTokens)

    # Initialize list for log probabilities.
    logProbs = []

    # Iterate through each generated token.
    for token in generatedTokens:
      # Calculate probability with smoothing.
      prob = refCounter[token] / totalRefTokens if (token in refCounter) else 1e-10

      # Add log probability if prob > 0.
      if (prob > 0):
        # Append the natural log of the probability.
        logProbs.append(np.log(prob))
      else:
        # Return infinity if probability is zero.
        return float("inf")

    # Handle empty logProbs.
    if (len(logProbs) == 0):
      # Return infinity for empty log probabilities.
      return float("inf")

    # Calculate average log probability.
    avgLogProb = np.mean(logProbs)

    # Calculate perplexity from average log probability.
    perplexity = np.exp(-avgLogProb)

    # Normalize perplexity by the number of generated tokens.
    perplexity = perplexity / len(generatedTokens)

    # Return perplexity score.
    return perplexity

  def CalculateNeuralPerplexity(self, critTokenizer, critModel, text, context=""):
    r'''
    Calculates the perplexity score of the provided text, conditioned on an optional context, using a deep learning approach.

    Architectural Paradigm & Differences:
    This function utilizes a deep learning approach, specifically a Transformer-based causal language model
    (such as GPT-2), to compute perplexity. It is highly context-aware. By processing the entire sequence
    through a neural network, it captures syntactic structures, word order, and long-range dependencies.

    Perplexity is defined as the exponent of the average negative log-likelihood of the tokens.
    In this implementation, the loss is computed exclusively over the target text portion by
    masking the context tokens. This ensures the score strictly reflects the predictability of
    the new text given the preceding context, rather than rewarding predictable context tokens.

    Heuristic Interpretation:
      - Lower scores (e.g., < 20.0) often indicate AI-generated or highly predictable text.
      - Moderate scores (e.g., 20.0 to 50.0) suggest natural human-like variability.
      - Extremely high scores (e.g., > 150.0) may indicate incoherent, fragmented, or gibberish text.

    Note: Perplexity is inherently model-dependent and architecture-specific. It should be
    evaluated alongside complementary metrics such as semantic similarity, fluency scoring,
    and stylistic analysis to ensure robust text quality detection.

    Parameters:
      critTokenizer (transformers.PreTrainedTokenizer): A pre-initialized tokenizer aligned
          with the critic model for consistent tokenization.
      critModel (transformers.PreTrainedModel): A pre-initialized causal language model
          used to compute the cross-entropy loss and subsequent perplexity.
      text (str): The target input text for which to calculate the perplexity score.
      context (str): An optional preceding context string to condition the calculation.

    Returns:
      float: The calculated perplexity score. Returns 0.0 for empty inputs or invalid states.

    Examples
    --------
    .. code-block:: python

      from transformers import GPT2LMHeadModel, GPT2Tokenizer
      from HMB.TextGenerationMetrics import TextGenerationMetrics

      # Define the critic model name (e.g., GPT-2).
      critModelName = "gpt2"
      # Initialize the tokenizer from the pretrained model.
      critTokenizer = GPT2Tokenizer.from_pretrained(critModelName)
      # Initialize the model from the pretrained weights.
      critModel = GPT2LMHeadModel.from_pretrained(critModelName)

      # Ensure the tokenizer has a pad token to avoid errors.
      if (critTokenizer.pad_token is None):
        # Set the pad token to the end of sequence token.
        critTokenizer.pad_token = critTokenizer.eos_token

      text ="The cat sat on the mat."
      context = "The cat sat on the mat. The cat is happy."

      # Execute the context-aware neural function with the current strings.
      obj = TextGenerationMetrics()
      neuralPPL = obj.CalculateNeuralPerplexity(critTokenizer, critModel, text, context)

      print(f"Neural Context-Aware Perplexity: {neuralPPL:.4f}")
    '''

    # Validate that the target text is not empty or purely whitespace.
    if (not text or not text.strip()):
      # Return zero immediately for invalid or empty input sequences.
      return 0.0

    # Construct the full input sequence by combining context and target text.
    if (context and context.strip()):
      # Concatenate the stripped context and text with a separating space.
      fullInput = f"{context.strip()} {text.strip()}"
    else:
      # Use only the target text when no context is provided.
      fullInput = text.strip()
      # Normalize the context variable to an empty string for subsequent logic.
      context = ""

    # Determine the maximum allowed sequence length from the model configuration.
    maxLength = getattr(critModel.config, "max_position_embeddings", 512)

    # Tokenize the full combined input sequence with truncation safety.
    encodings = critTokenizer(
      fullInput,
      return_tensors="pt",
      truncation=True,
      max_length=maxLength,
    ).to(critModel.device)

    # Clone the input token IDs to initialize the label tensor for loss computation.
    labels = encodings.input_ids.clone()

    # Mask the context tokens to exclude them from the perplexity calculation.
    if (context):
      # Tokenize the context independently to determine its exact token span.
      contextEncodings = critTokenizer(
        context.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=maxLength,
      ).to(critModel.device)

      # Extract the sequence length of the isolated context tokens.
      contextLen = contextEncodings.input_ids.shape[1]

      # Extract the total sequence length of the combined input tokens.
      inputLen = encodings.input_ids.shape[1]

      # Calculate a safe masking boundary to prevent tensor indexing errors.
      maskLength = min(contextLen, inputLen)

      # Assign -100 to the context label positions to ignore them during loss calculation.
      labels[:, :maskLength] = -100

    # Disable gradient computation to optimize memory usage during inference.
    with torch.inference_mode():
      # Attempt to execute the forward pass with the prepared inputs and labels.
      try:
        outputs = critModel(input_ids=encodings.input_ids, labels=labels)

        # Verify that the computed loss is a valid numerical value and not NaN.
        if (outputs.loss is None or torch.isnan(outputs.loss)):
          # Return zero if the loss value is invalid or undefined.
          return 0.0

        # Transform the cross-entropy loss into an interpretable perplexity score.
        perplexity = torch.exp(outputs.loss).item()

        # Return the successfully calculated perplexity value.
        return perplexity

      # Catch any unexpected runtime exceptions that occur during model execution.
      except Exception as e:
        # Print a diagnostic warning message containing the exception details.
        print(f"⚠️ Error during perplexity calculation: {e}")

        # Return zero as a safe fallback value upon encountering an error.
        return 0.0

  def CalculateFluencyScore(self, genTokenizer, genModel, text: str, context: str = "") -> float:
    r'''
    Calculates a normalized fluency score for the provided text using model-based perplexity.

    Architectural Paradigm & Heuristics:
    This function converts raw perplexity into a bounded fluency score between 0.0 and 1.0
    by applying a multi-stage heuristic pipeline. Unlike raw perplexity, which is unbounded
    and model-dependent, this score provides an interpretable, normalized metric suitable
    for ranking candidate text variations during the humanization process.

    Heuristics applied:
      1. Hard Length Rejection: Texts with fewer than 3 words receive a minimal score.
      2. Linear Perplexity Inversion: Maps typical perplexity ranges to a 0.0–1.0 range.
      3. Aggressive Short-Text Penalty: Caps the maximum score for texts under 8 words.
      4. Valid Token Scaling: Requires approximately 15 tokens for full score confidence.
      5. Repetition Penalty: Detects and penalizes texts with excessive word repetition.
      6. Punctuation Penalty: Reduces the score for texts lacking terminal punctuation.
      7. Capitalization Penalty: Reduces the score for texts not starting with a capital letter.

    Parameters:
      genTokenizer (transformers.PreTrainedTokenizer): A pre-initialized tokenizer
        aligned with the model for consistent tokenization.
      genModel (transformers.PreTrainedModel): A pre-initialized language model
        used to compute the cross-entropy loss for perplexity estimation.
      text (str): The candidate text to evaluate for fluency.
      context (str): Optional preceding context to condition the loss calculation.

    Returns:
      float: A normalized fluency score between 0.0 (incoherent) and 1.0 (perfectly fluent).

    Examples
    --------
    .. code-block:: python

      from transformers import GPT2LMHeadModel, GPT2Tokenizer
      from HMB.TextGenerationMetrics import TextGenerationMetrics

      # Load a pre-trained GPT-2 model and tokenizer.
      modelName = "gpt2"
      tokenizer = GPT2Tokenizer.from_pretrained(modelName)
      model = GPT2LMHeadModel.from_pretrained(modelName)

      # Ensure the tokenizer has a pad token to avoid errors.
      if (tokenizer.pad_token is None):
        tokenizer.pad_token = tokenizer.eos_token

      # Initialize the metrics object.
      metrics = TextGenerationMetrics()

      # Calculate fluency score for a candidate text with optional context.
      candidateText = "The cat sat on the mat."
      contextText = "In the living room, the cat was very comfortable."
      fluencyScore = metrics.CalculateFluencyScore(tokenizer, model, candidateText, contextText)
      print(f"Fluency Score: {fluencyScore:.4f}")
    '''

    # Validate that the input text is a non-empty string.
    if (not text or not isinstance(text, str)):
      # Return a minimal score for invalid or empty input.
      return 0.1

    # Strip leading and trailing whitespace from the input text.
    text = text.strip()

    # Strip leading and trailing whitespace from the optional context.
    context = context.strip() if (context) else ""

    # Calculate the number of words in the stripped text.
    wordCount = len(text.split())

    # Apply a hard rejection for extremely short text that is too noisy for perplexity.
    if (wordCount < 3):
      # Return a minimal score for text that is too short to evaluate reliably.
      return 0.1

    # Attempt to calculate the fluency score with comprehensive error handling.
    try:
      # Construct the full input sequence by combining context and target text.
      fullText = (context + " " + text) if (context) else text

      # Tokenize the full combined input sequence with truncation safety.
      encoding = genTokenizer(
        fullText,
        return_tensors="pt",
        truncation=True,
        max_length=512,
      )

      # Move the input token IDs to the model's compute device.
      inputIds = encoding.input_ids.to(genModel.device)

      # Clone the input token IDs to initialize the label tensor for loss computation.
      labels = inputIds.clone()

      # Initialize the count of valid (unmasked) tokens.
      countValidTokens = inputIds.shape[1]

      # Mask the context tokens to exclude them from the perplexity calculation.
      if (context):
        # Tokenize the context independently to determine its exact token span.
        contextEncoding = genTokenizer(
          context,
          return_tensors="pt",
          truncation=True,
          max_length=512,
        )

        # Extract the sequence length of the isolated context tokens.
        contextLen = contextEncoding.input_ids.shape[1]

        # Check if the context is shorter than the full input to apply masking.
        if (contextLen < inputIds.shape[1]):
          # Assign -100 to the context label positions to ignore them during loss.
          labels[:, :contextLen] = -100

          # Calculate the number of valid (unmasked) target tokens.
          countValidTokens = inputIds.shape[1] - contextLen
        else:
          # Set valid token count to zero if context covers the entire input.
          countValidTokens = 0

      # Apply a safety check to reject inputs where all tokens are masked.
      if (countValidTokens < 2):
        # Return a minimal score when there are insufficient valid tokens.
        return 0.1

      # Disable gradient computation to optimize memory usage during inference.
      with torch.inference_mode():
        # Execute the forward pass with the prepared inputs and labels.
        outputs = genModel(inputIds, labels=labels)

        # Extract the computed loss from the model outputs.
        loss = outputs.loss

        # Verify that the computed loss is a valid numerical value and not NaN.
        if (loss is None or torch.isnan(loss)):
          # Return a minimal score if the loss value is invalid or undefined.
          return 0.05

        # Transform the cross-entropy loss into a raw perplexity value.
        perplexity = torch.exp(loss).item()

      # --- Scoring Formula ---
      # Base Score: Linear inversion mapped to a 0-250 perplexity range.
      # Good text PPL: ~20-50 -> Score ~0.8-0.9
      # Moderate text PPL: ~100 -> Score ~0.6
      # Gibberish PPL: ~250+ -> Score ~0.0
      baseScore = max(0.0, 1.0 - (perplexity / 250.0))

      # Length Penalty & Capping
      # Problem: Short gibberish (3-4 words) often has artificially low PPL.
      # Solution: Cap the maximum possible score for very short texts, but allow normal sentences to score well.
      if (wordCount < 6):
        # Max score scales from 0.30 (3 words) to 0.50 (6 words).
        maxPossibleScore = 0.30 + (wordCount * 0.05)
        baseScore = min(baseScore, maxPossibleScore)

      # Calculate a length confidence factor based on the number of valid tokens.
      lengthFactor = min(1.0, countValidTokens / 15.0)

      # Blend the base score with the length confidence factor to produce a preliminary score.
      finalScore = baseScore * (0.6 + 0.4 * lengthFactor)

      # Apply a penalty for texts with a high ratio of duplicated words (common in repetitive gibberish).
      wordFrequencies = {}

      # Iterate through each word in the text to count occurrences, stripping punctuation.
      for word in text.lower().split():
        # Clean punctuation from word for accurate counting.
        cleanWord = word.strip(".,!?;:\"'()[]{}")

        # Increment the frequency count for the current cleaned word.
        if (cleanWord):
          wordFrequencies[cleanWord] = wordFrequencies.get(cleanWord, 0) + 1

      # Check the diversity of the vocabulary to detect repetitive gibberish.
      if (wordFrequencies):
        # Count how many unique words appear more than once in the text.
        duplicatedUniqueWords = sum(1 for count in wordFrequencies.values() if count > 1)

        # Calculate the total number of unique words.
        totalUniqueWords = len(wordFrequencies)

        # Calculate the ratio of duplicated unique words to total unique words.
        duplicateRatio = duplicatedUniqueWords / totalUniqueWords

        # Apply a heavy penalty if more than 30% of the unique vocabulary is repeated.
        if (duplicateRatio > 0.3):
          # Reduce the final score proportionally to the vocabulary repetition.
          finalScore *= (1.0 - duplicateRatio)

      # Apply a punctuation penalty for texts lacking proper terminal punctuation.
      if (text[-1] not in ".!?"):
        # Reduce the score by ten percent for missing terminal punctuation.
        finalScore *= 0.9

      # Apply a capitalization penalty for texts not starting with a capital letter.
      if (not text[0].isupper()):
        # Reduce the score by five percent for missing initial capitalization.
        finalScore *= 0.95

      # Clamp the final score to the valid range of 0.0 to 1.0.
      finalScore = max(0.0, min(1.0, finalScore))

      # Return the fully adjusted and clamped fluency score.
      return round(finalScore, 4)

    # Catch any unexpected runtime exceptions during the fluency calculation.
    except Exception as e:
      # Print a diagnostic warning message containing the exception details.
      print(f"⚠️ Fluency scoring failed: {e}")

      # Return a minimal fallback score upon encountering an error.
      return 0.05

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

    from nltk.translate.chrf_score import sentence_chrf  # CHRF metric.

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

    import textstat

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

    import nltk

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
      "SemanticSimilarity": self.CalculateJaccardSimilarity(generatedText, referenceText),
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
    semanticSimilarity = metrics.CalculateJaccardSimilarity(genText, refText)
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
