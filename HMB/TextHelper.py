# Import the required libraries.
import contractions, spacy, uuid, nltk, re
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Dict, Tuple, Any, Optional, Set


class HuggingFaceTextSummarizer(object):
  r'''
  Flexible text summarization using Hugging Face Transformers.

  This class provides a convenient interface for abstractive text summarization using pre-trained transformer models
  (e.g., BART, T5) via the Hugging Face pipeline. It automatically handles device selection (CPU/GPU), input chunking
  for long texts, and dynamic adjustment of summary length to avoid common warnings. The class is suitable for both
  short and long documents, and can be reused for multiple summarization tasks without reloading the model.

  Features:
    - Supports any Hugging Face summarization model (default: facebook/bart-large-cnn).
    - Automatically uses GPU if available, otherwise falls back to CPU.
    - Handles long texts by splitting into manageable chunks and summarizing each chunk.
    - Dynamically sets max_length and min_length based on tokenized input length to avoid warnings.
    - Returns a single concatenated summary for the entire input.
    - Easy to customize summary length and chunk size.

  Parameters:
    modelName (str): Name of the Hugging Face model to use for summarization. Default is "facebook/bart-large-cnn".
    maxLength (int): Maximum length of the summary (in tokens). Default is 130.
    minLength (int): Minimum length of the summary (in tokens). Default is 30.
    maxInputLength (int): Maximum input length (in characters) before chunking. Default is 1024.

  Notes:
    - For best results, choose a model appropriate for your language and domain.
    - If your input text is very long, the class will split it into chunks and summarize each chunk separately.
    - The final summary is a concatenation of all chunk summaries.
    - maxLength and minLength are automatically adjusted to avoid warnings about input length.
    - You can customize chunk size by changing maxInputLength.
    - The class is thread-safe for repeated use.

  Examples
  --------
  .. code-block:: python

    from HMB.TextHelper import HuggingFaceTextSummarizer

    text = "Your long text to summarize goes here..."
    summarizer = HuggingFaceTextSummarizer(
      modelName="facebook/bart-large-cnn",  # Use a pre-trained summarization model from Hugging Face.
      maxLength=130,  # Maximum summary length in tokens.
      minLength=30,  # Minimum summary length in tokens.
      maxInputLength=1024  # Maximum input length in characters before chunking.
    )
    summary = summarizer.Summarize(text)
    print(summary)
  '''

  def __init__(
    self,
    modelName="facebook/bart-large-cnn",
    maxLength=130,
    minLength=30,
    maxInputLength=1024,
  ):
    r'''
    Initialize the Summarizer with the specified model and parameters.

    Parameters:
      modelName (str): Hugging Face model name for summarization (e.g., "facebook/bart-large-cnn", "t5-base").
      maxLength (int): Maximum summary length in tokens (will be capped by input length).
      minLength (int): Minimum summary length in tokens (will be capped by input length).
      maxInputLength (int): Maximum input length in characters before chunking. Longer texts are split into chunks.

    Notes:
      - The model is loaded once and reused for all summarization calls.
      - Device selection is automatic: uses GPU if available, otherwise CPU.
    '''

    import torch
    from transformers import pipeline

    self.modelName = modelName
    self.maxLength = maxLength
    self.minLength = minLength
    self.maxInputLength = maxInputLength
    self.summarizer = pipeline(
      "summarization",
      model=modelName,
      device=0 if (torch.cuda.is_available()) else -1,
    )

  def Summarize(self, text):
    r'''
    Summarize the input text using the loaded transformer model.

    Parameters:
      text (str): The text to summarize. Can be short or long (long texts are chunked automatically).

    Returns:
      str: The summarized text. For long inputs, returns a concatenation of chunk summaries.

    Behavior:
      - If the input text exceeds maxInputLength, it is split into chunks of 1000 characters.
      - Each chunk is summarized separately, with maxLength and min_length set based on tokenized input length.
      - Warnings about maxLength/inputLength mismatch are avoided by dynamic adjustment.
      - If a chunk is too short, it is returned as-is without summarization.
      - The final output is a single string containing all chunk summaries.

    Example:
    --------
    .. code-block:: python

        summarizer = HuggingFaceTextSummarizer()
        summary = summarizer.Summarize("Very long text ...")
        print(summary)

    Notes:
      - For best results, clean your input text before summarization.
      - You can customize summary length by changing maxLength and minLength.
      - The method is robust to empty or very short inputs.
    '''

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(self.modelName)

    if (len(text) > self.maxInputLength):
      # Split text into chunks of maxInputLength characters.
      chunks = [text[i:i + self.maxInputLength] for i in range(0, len(text), self.maxInputLength)]
      # Collect summaries for each chunk.
      summaries = []
      # Summarize each chunk separately.
      for chunk in chunks:
        # Tokenize the chunk to get input length.
        chunkTokens = tokenizer(chunk, return_tensors="pt")
        # Get the length of the tokenized input.
        inputLength = len(chunkTokens["input_ids"][0])

        # If input is too short, skip summarization.
        if (inputLength == 0):
          continue
        elif (inputLength < self.minLength):
          summaries.append(chunk)
          continue

        # Dynamically set maxLength to avoid warnings.
        if (inputLength > 1):
          maxLen = min(self.maxLength, inputLength - 1)
        else:
          maxLen = 1

        # Generate summary for the chunk.
        summary = self.summarizer(
          chunk,
          max_length=maxLen,
          min_length=min(self.minLength, maxLen),
          do_sample=False,
        )[0]["summary_text"]
        # Append the chunk summary to the list.
        summaries.append(summary)

      # Return the concatenated summaries for all chunks.
      return " ".join(summaries)
    else:
      # Tokenize the entire text to get input length.
      textTokens = tokenizer(text, return_tensors="pt")
      inputLength = len(textTokens["input_ids"][0])

      # If input is too short, return as-is.
      if (inputLength == 0):
        return ""
      elif (inputLength < self.minLength):
        return text

      # Dynamically set maxLength to avoid warnings.
      if (inputLength > 1):
        maxLen = min(self.maxLength, inputLength - 1)
      else:
        maxLen = 1

      # Generate summary for the entire text.
      summary = self.summarizer(
        text,
        max_length=maxLen,
        min_length=min(self.minLength, maxLen),
        do_sample=False,
      )[0]["summary_text"]
      return summary


class TextHelper:
  r'''
  TextHelper: Small collection of common text-processing utilities.

  This class provides thin wrappers around NLTK, gensim and scikit-learn
  helpers for tokenization, stopword removal, stemming, lemmatization,
  simple POS/chunking helpers, and common vectorization examples.

  Notes:
    - Methods are convenience wrappers and try to preserve original behavior.
    - Many methods assume NLTK corpora (punkt, stopwords) are available.
    - You need to pip install nltk, gensim, wordcloud, gtts, and scikit-learn to use this class.
  '''

  def SentenceTokeize(self, document):
    r'''
    Tokenize a document into word tokens using NLTK's word_tokenize.

    Parameters:
      document (str): Input text to tokenize.

    Returns:
      list: Word tokens.
    '''

    document = word_tokenize(document)
    return document

  def WordTokenize(self, document):
    r'''
    Split a document into sentences using NLTK's sent_tokenize.

    Note: method name is historical; it returns sentence-level tokens.

    Parameters:
      document (str): Input text.

    Returns:
      list: Sentence strings.
    '''

    document = sent_tokenize(document)
    return document

  def RemoveStopWords(self, document):
    r'''
    Remove English stopwords from an iterable of tokens.

    Parameters:
      document (iterable): Iterable of token strings.

    Returns:
      list: Tokens with stopwords removed.
    '''

    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))

    filteredSentence = [w for w in document if not w in stop_words]
    return filteredSentence

  def Stemming(self, document):
    r'''
    Return Porter stems for each token in the input.

    Parameters:
      document (iterable): Iterable of token strings.

    Returns:
      list of tuple: Each tuple is (token, stem).
    '''

    from nltk.stem import PorterStemmer

    ps = PorterStemmer()
    return [(w, ps.stem(w)) for w in document]

  def Lemmatization(self, document):
    r'''
    Return WordNet lemma for each token.

    Parameters:
      document (iterable): Iterable of token strings.

    Returns:
      list of tuple: Each tuple is (token, lemma).
    '''

    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    return [(w, lemmatizer.lemmatize(w)) for w in document]

  def POS(self, document):
    r'''
    Part-of-speech tagging using NLTK's averaged perceptron tagger.

    Parameters:
      document (iterable): Iterable of token strings.

    Returns:
      list of tuple: POS tags as returned by nltk.pos_tag.
    '''

    import nltk

    nltk.download("averaged_perceptron_tagger")

    return nltk.pos_tag(document)

  def Chunking(self, document):
    r'''
    Named-entity chunking helper using NLTK's ne_chunk.

    Parameters:
      document (iterable): Iterable of token strings (POS-tagged preferred).

    Returns:
      nltk.Tree: Chunked parse tree produced by ne_chunk.
    '''

    import nltk
    from nltk import ne_chunk

    # Try to ensure required resources are available; be tolerant if a specific chunker table is missing.
    try:
      nltk.download("maxent_ne_chunker")
      nltk.download("words")
    except Exception:
      # Downloads may fail in restricted environments; we'll try to proceed and catch LookupError below.
      pass

    try:
      return ne_chunk(document)
    except LookupError:
      # Some NLTK installs are missing the compiled chunker table; try the more specific resource name then retry.
      try:
        nltk.download("maxent_ne_chunker_tab")
        return ne_chunk(document)
      except Exception:
        # As a graceful fallback, return the input tokens when chunking can't be performed.
        return document

  def BagOfWords(self, document):
    r'''
    Vectorize documents using CountVectorizer and return the vectorizer and dense matrix.

    Parameters:
      document (list of str): Iterable of documents.

    Returns:
      (vectorizer, numpy.ndarray): The fitted CountVectorizer instance and the dense array.
    '''

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(document)
    return vectorizer, X.toarray()

  def TFIDF(self, document):
    r'''
    Vectorize documents using TfidfVectorizer and return the vectorizer and dense matrix.

    Parameters:
      document (list of str): Iterable of documents.

    Returns:
      (vectorizer, numpy.ndarray): The fitted TfidfVectorizer instance and the dense array.
    '''

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(document)
    return vectorizer, X.toarray()

  def Word2Vec(self, document):
    r'''
    Train a Word2Vec model and return the model and vocabulary list.

    Parameters:
      document (list of list of str): Tokenized sentences.

    Returns:
      (model, list): Trained gensim Word2Vec model and list of vocabulary words.
    '''

    from gensim.models import Word2Vec

    # Use modern gensim API: explicitly pass sentences and vector_size, then build vocab and train.
    try:
      model = Word2Vec(sentences=document, vector_size=20, min_count=1, epochs=50)
      # In some gensim versions, training occurs during initialization when 'sentences' passed; ensure wv exists.
      try:
        words = list(model.wv.index_to_key)
      except Exception:
        # Older gensim: fallback to vocab attribute
        words = list(model.wv.vocab)
    except Exception as e:
      # Fallback: create an untrained model and attempt to build vocab/train safely
      model = Word2Vec(vector_size=20, min_count=1)
      model.build_vocab(document)
      model.train(document, total_examples=model.corpus_count, epochs=50)
      try:
        words = list(model.wv.index_to_key)
      except Exception:
        words = list(model.wv.vocab)

    return model, words

  def Doc2Vec(self, document, saveModel=False, savePath="d2v.model", epochs=100, vecSize=20, alpha=0.025):
    r'''
    Train a Doc2Vec model on provided documents and return the model.

    Parameters:
      document (list of str): Iterable of documents (strings).

    Returns:
      model: Trained gensim Doc2Vec model.
    '''
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    tagged_data = [
      TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
      for i, _d in enumerate(document)
    ]
    # Use modern gensim parameter names (vector_size instead of size)
    model = Doc2Vec(vector_size=vecSize, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)
    model.build_vocab(tagged_data)

    # Train in small epoch increments to allow alpha decay similar to older examples.
    try:
      for epoch in range(epochs):
        model.train(tagged_data, total_examples=model.corpus_count, epochs=1)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
    except Exception:
      # Fallback: single train call
      model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs)

    if (saveModel):
      model.save(savePath)
    return model

  def LDA(self, document):
    r'''
    Fit a simple LDA model using gensim's LdaModel and return the model and topics.

    Parameters:
      document (list of list of str): Tokenized documents.

    Returns:
      (model, list): LdaModel instance and printed topics list.
    '''

    from gensim import corpora
    from gensim.models.ldamodel import LdaModel

    dictionary = corpora.Dictionary(document)
    docTermMatrix = [dictionary.doc2bow(doc) for doc in document]

    Lda = LdaModel
    ldamodel = Lda(docTermMatrix, num_topics=3, id2word=dictionary, passes=50)
    topics = ldamodel.print_topics(num_topics=3, num_words=3)
    return ldamodel, topics

  def LSA(self, document):
    r'''
    Fit a simple LSI model using gensim and return the model and topics.

    Parameters:
      document (list of list of str): Tokenized documents.

    Returns:
      (model, list): LsiModel instance and printed topics list.
    '''

    from gensim import corpora, models

    dictionary = corpora.Dictionary(document)
    docTermMatrix = [dictionary.doc2bow(doc) for doc in document]
    lsi = models.LsiModel(docTermMatrix, num_topics=3, id2word=dictionary)
    topics = lsi.print_topics(num_topics=3, num_words=3)
    return lsi, topics

  def TextSummarization(self, document):
    r'''
    Extractive summarization wrapper using gensim.summarization.summarize.

    Parameters:
      document (str): Text to summarize.

    Returns:
      str: Summary string.
    '''

    try:
      from gensim.summarization import summarize
      return summarize(document)
    except Exception:
      # gensim.summarization may be unavailable in newer gensim installs; fallback to a simple heuristic.
      try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(document)
        return sents[0] if sents else ""
      except Exception:
        return ""

  def TextRank(self, document):
    r'''
    Alias for TextSummarization (uses gensim.summarization.summarize).

    Parameters:
      document (str): Text to summarize.

    Returns:
      str: Summary string.
    '''

    try:
      from gensim.summarization import summarize
      return summarize(document)
    except Exception:
      # Fallback to first sentence
      try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(document)
        return sents[0] if sents else ""
      except Exception:
        return ""

  def TopicModeling(self, document):
    r'''
    Extract keywords from a document using gensim's keywords function.

    Parameters:
      document (str): Input text.

    Returns:
      str: Keywords extracted from the document.
    '''

    try:
      from gensim.summarization import keywords
      return keywords(document)
    except Exception:
      # Fallback: return top-n frequent words as a simple keyword approximation.
      try:
        words = [w.lower() for w in re.findall(r"\w+", document)]
        freq = {}
        for w in words:
          freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return " ".join([w for w, _ in top])
      except Exception:
        return ""

  def CosineSimilarity(self, text1, text2):
    r'''
    Demonstrate cosine similarity between two example texts and return the matrix.

    Parameters:
      text1 (str): First input text.
      text2 (str): Second input text.

    Returns:
      numpy.ndarray: Cosine similarity matrix.
    '''
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    text = [text1, text2]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)
    return cosine_similarity(count_matrix)

  def JaccardSimilarity(self, text1, text2):
    r'''
    Demonstrate Jaccard similarity between two example texts and return the score.

    Parameters:
      text1 (str): First input text.
      text2 (str): Second input text.

    Returns:
      float: Jaccard similarity score between example texts (binary presence).
    '''

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import jaccard_score

    text = [text1, text2]
    cv = CountVectorizer()
    countMatrix = cv.fit_transform(text).toarray()
    # Convert to binary presence vectors and compute Jaccard on flattened arrays.
    a = (countMatrix[0] > 0).astype(int)
    b = (countMatrix[1] > 0).astype(int)

    try:
      return jaccard_score(a, b)
    except Exception:
      # Fallback: compute set-based Jaccard.
      sa = set([w for w, v in zip(cv.get_feature_names_out(), a) if v])
      sb = set([w for w, v in zip(cv.get_feature_names_out(), b) if v])
      if (not sa and not sb):
        return 0.0
      return len(sa & sb) / float(len(sa | sb))

  def EuclideanDistance(self, text1, text2):
    r'''
    Demonstrate Euclidean distance between two example texts and return the matrix.

    Parameters:
      text1 (str): First input text.
      text2 (str): Second input text.

    Returns:
      numpy.ndarray: Euclidean distance matrix.
    '''

    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.feature_extraction.text import CountVectorizer

    text = [text1, text2]
    cv = CountVectorizer()
    countMatrix = cv.fit_transform(text)
    return euclidean_distances(countMatrix)

  def ManhattanDistance(self, text1, text2):
    r'''
    Demonstrate Manhattan (L1) distance between two example texts and return the matrix.

    Parameters:
      text1 (str): First input text.
      text2 (str): Second input text.

    Returns:
      numpy.ndarray: Manhattan distance matrix.
    '''

    from sklearn.metrics.pairwise import manhattan_distances
    from sklearn.feature_extraction.text import CountVectorizer

    text = [text1, text2]
    cv = CountVectorizer()
    countMatrix = cv.fit_transform(text)
    return manhattan_distances(countMatrix)

  def WordCloud(self, document):
    r'''
    Generate a word cloud object for the given text and return it.

    Parameters:
      document (str): Input text from which to build the word cloud.

    Returns:
      WordCloud: The generated WordCloud instance (from wordcloud package).
    '''

    from wordcloud import WordCloud

    wordcloud = WordCloud().generate(document)
    return wordcloud

  def TextToSpeech(self, document, outFile="good.mp3"):
    r'''
    Convert text to speech using gTTS, save to an MP3 file, and return filename.

    Parameters:
      document (str): Input text to convert to speech.
      outFile (str): Output file path for MP3.

    Returns:
      str: Path to the saved MP3 file.
    '''

    from gtts import gTTS
    tts = gTTS(text=document, lang="en")
    tts.save(outFile)
    return outFile

  def SentimentAnalysis(self, document):
    r'''
    Perform sentiment analysis using TextBlob and return the sentiment object.

    Parameters:
      document (str): Input text.

    Returns:
      object: TextBlob sentiment result (polarity, subjectivity).
    '''

    from textblob import TextBlob

    testimonial = TextBlob(document)
    return testimonial.sentiment

  def LanguageDetection(self, document):
    r'''
    Detect the language of the input text using langdetect.

    Parameters:
      document (str): Input text.

    Returns:
      str: Detected language code.
    '''

    from langdetect import detect
    return detect(document)

  def SpellingCorrection(self, document):
    r'''
    Perform spelling correction using TextBlob and return corrected string.

    Parameters:
      document (str): Input text.

    Returns:
      str: Corrected text.
    '''

    from textblob import TextBlob

    return str(TextBlob(document).correct())

  def LanguageTranslation(self, document, to="es"):
    r'''
    Translate text using TextBlob's translation wrapper and return the result.

    Parameters:
      document (str): Input text.
      to (str): Target language code (default "es").

    Returns:
      str: Translated text.
    '''

    from textblob import TextBlob

    try:
      # TextBlob.translate may not be available or may require extra dependencies; attempt and fallback.
      return str(TextBlob(document).translate(to=to))
    except Exception:
      # Fallback: return original text (no translation available in environment)
      return document

  def Tokenization(self, document):
    r'''
    Return sentence and word tokenization using NLTK utilities.

    Parameters:
      document (str): Input text.

    Returns:
      (list, list): Tuple of (sentences, words).
    '''

    from nltk.tokenize import sent_tokenize, word_tokenize
    return sent_tokenize(document), word_tokenize(document)

  def NER(self, document):
    r'''
    Named-entity recognition using spaCy's small English model; returns list of entities.

    Parameters:
      document (str): Input text.

    Returns:
      list of tuple: Each tuple is (text, start_char, end_char, label).
    '''

    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(document)
    return [
      (ent.text, ent.start_char, ent.end_char, ent.label_)
      for ent in doc.ents
    ]


class TextSemanticShield:
  r'''
  Extracts named entities, domain-specific terms, and noun chunks via spaCy, then replaces them with high-entropy
  cryptographic tokens («K_<uuid>») so downstream transformations cannot corrupt technical terminology.

  Examples
  --------
  .. code-block:: python

    from HMB.TextHelper import TextSemanticShield

    text = "Alice works at OpenAI in San Francisco."
    shield = TextSemanticShield("en_core_web_sm")
    maskedText, entityMap = shield.ApplyCryptographicMask(text)
    print(maskedText)  # e.g., "«K_1a2b3c4d» works at «K_5e6f7g8h» in «K_9i0j1k2l»."
    restoredText = shield.RestoreEntities(maskedText, entityMap)
    print(restoredText)  # "Alice works at OpenAI in San Francisco."
  '''

  def __init__(self, nlpModelName: str = "en_core_web_sm"):
    r'''
    Initialize the SemanticShield with a spaCy language model for entity extraction and masking.

    Parameters:
      - nlpModelName (str): The name of the spaCy language model to load. Defaults to "en_core_web_sm".
    '''

    # Attempt to load the spaCy language model for dependency parsing and NER.
    try:
      # Load the spaCy model directly from the installed packages.
      self.nlpModel = spacy.load(nlpModelName)
    except OSError:
      # Import subprocess to download the missing spaCy model automatically.
      import subprocess
      # Run the spaCy download command for the specified model.
      subprocess.run(["python", "-m", "spacy", "download", nlpModelName])
      # Retry loading the model after successful download.
      self.nlpModel = spacy.load(nlpModelName)

    # Initialize the dictionary that maps cryptographic tokens to original entities.
    self.entityMap: Dict[str, str] = {}

    # Define the regex pattern for detecting domain-specific technical terms.
    self.customPattern = r"\b([A-Z][a-zA-Z0-9-]*\b|\b[A-Z]{2,}\b|\b[a-zA-Z]+-[a-zA-Z0-9]+\b)"

    # Define the minimum entity length threshold to filter out tiny artifacts.
    self.minEntityLength = 2

  def ApplyCryptographicMask(self, text: str) -> Tuple[str, Dict[str, str]]:
    r'''
    Apply cryptographic masking to named entities, technical terms, and noun chunks in the input text.
    This method replaces detected entities with unique high-entropy tokens to prevent downstream transformations
    from altering critical terminology. It returns the masked text along with a mapping of tokens to original entities for restoration.

    Parameters:
      - text (str): The input text to be processed and masked.

    Returns:
      - Tuple[str, Dict[str, str]]: A tuple containing the masked text and a dictionary mapping cryptographic tokens to their corresponding original entities.
    '''

    # Parse the input text using the spaCy language model pipeline.
    doc = self.nlpModel(text)

    # Initialize the masked text variable with the original input text.
    maskedText = text

    # Clear the entity mapping dictionary to prepare for a new masking session.
    self.entityMap.clear()

    # Extract named entities detected by the spaCy NER pipeline.
    entitiesToProtect: List[str] = [ent.text for ent in doc.ents]

    # Find all custom technical entities matching the domain-specific regex pattern.
    customEntities: List[str] = re.findall(self.customPattern, text)

    # Extend the protection list with the custom technical entities.
    entitiesToProtect.extend(customEntities)

    # Iterate through noun chunks to capture multi-word logical phrases.
    for chunk in doc.noun_chunks:
      # Check if the noun chunk text is longer than three characters.
      if (len(chunk.text.strip()) > 3):
        # Add the noun chunk text to the protection list.
        entitiesToProtect.append(chunk.text.strip())

    # Remove duplicates and sort by length descending to prevent partial masking.
    entitiesToProtect = sorted(list(set(entitiesToProtect)), key=len, reverse=True)

    # Iterate through each unique entity to apply a unique cryptographic mask.
    for entity in entitiesToProtect:
      # Check if the entity exceeds the minimum length threshold.
      if (len(entity) > self.minEntityLength):
        # Generate a high-entropy cryptographic placeholder string using uuid.
        cryptoToken = f"«K_{uuid.uuid4().hex[:8]}»"

        # Map the generated placeholder to the original entity text.
        self.entityMap[cryptoToken] = entity

        # Escape special regex characters in the entity to prevent matching errors.
        escapedEntity = re.escape(entity)

        # Compile a regex pattern with word boundaries for exact whole-word matching.
        pattern = r"\b" + escapedEntity + r"\b"

        # Replace all occurrences of the entity with the cryptographic placeholder.
        maskedText = re.sub(pattern, cryptoToken, maskedText)

    # Return the fully masked text and the restoration mapping dictionary.
    return maskedText, self.entityMap

  # Restore the original entities from the cryptographic placeholders.
  def RestoreEntities(self, text: str, entityMap: Dict[str, str]) -> str:
    r'''
    Restore the original entities in the text by replacing cryptographic placeholders with their corresponding original entity texts.
    This method uses the mapping of cryptographic tokens to original entities to reconstruct the original text from the masked version.

    Parameters:
      - text (str): The masked text containing cryptographic placeholders.
      - entityMap (Dict[str, str]): A dictionary mapping cryptographic tokens to their corresponding original entity texts.

    Returns:
      - str: The fully restored text with all original entities intact.
    '''

    # Initialize the restored text variable with the masked text input.
    restoredText = text

    # Iterate through the placeholder mapping to restore each original entity.
    for token, originalEntity in entityMap.items():
      # Replace the cryptographic placeholder with the original entity text.
      restoredText = restoredText.replace(token, originalEntity)

    # Return the fully restored text with all entities intact.
    return restoredText


# Define a function to clean and normalize text based on several options.
def CleanText(
  text,  # The raw input text to be cleaned.
  removeNonAscii=True,  # Whether to remove non-ASCII characters.
  lowercase=True,  # Whether to convert text to lowercase.
  removeSpecialChars=True,  # Whether to remove special characters and punctuation.
  normalizeWhitespace=True,  # Whether to replace multiple spaces with a single space.
  handleContractions=True,  # Whether to expand contractions (e.g., "don't" → "do not").
  lemmatize=False,  # Whether to lemmatize words (reduce to base form).
  removeStopwords=False,  # Whether to remove common stop words.
  removeCommonWords=False,  # Whether to remove common words.
  numOfCommonWords=10,  # Number of common words to remove if removeCommonWords is True.
  removeNonEnglishWords=False,  # Whether to remove non-English words.
):
  r'''
  Cleans the input text based on specified options. It applies multiple text normalization techniques including (1)
  removing non-ASCII characters, (2) converting to lowercase, (3) removing special characters and punctuation,
  (4) normalizing whitespace, (5) expanding contractions, (6) lemmatizing words, (7) removing stopwords,
  (8) removing common words, and (9) removing non-English words.

  Parameters:
    text (str): The raw input text to be cleaned.
    removeNonAscii (bool): Whether to remove non-ASCII characters. Default is True.
    lowercase (bool): Whether to convert text to lowercase. Default is True.
    removeSpecialChars (bool): Whether to remove special characters and punctuation. Default is True.
    normalizeWhitespace (bool): Whether to replace multiple spaces with a single space. Default is True.
    handleContractions (bool): Whether to expand contractions (e.g., "don't" → "do not"). Default is True.
    lemmatize (bool): Whether to lemmatize words (reduce to base form). Default is False.
    removeStopwords (bool): Whether to remove common stop words. Default is False.
    removeCommonWords (bool): Whether to remove common words. Default is False.
    numOfCommonWords (int): Number of common words to remove if removeCommonWords is True. Default is 10.
    removeNonEnglishWords (bool): Whether to remove non-English words. Default is False.

  Returns:
    str: The cleaned and normalized text.

  Examples
  --------
  .. code-block:: python

    import HMB.TextHelper as th

    raw = "I can't believe it's not butter!   "
    cleaned = th.CleanText(
      raw,
      removeNonAscii=True,
      lowercase=True,
      removeSpecialChars=True,
      normalizeWhitespace=True,
      handleContractions=True,
      lemmatize=True,
      removeStopwords=True,
      removeCommonWords=False,
      numOfCommonWords=10,
      removeNonEnglishWords=False,
    )
    print(cleaned)
  '''

  # Remove empty lines and replace newlines with spaces.
  cleanedText = " ".join([
    line.replace("\n", " ").strip()  # Replace newlines with spaces and strip whitespace.
    for line in text.splitlines()  # Split text into lines.
    if (line.strip() != "")  # Ignore empty lines.
  ])

  # Remove non-ASCII characters if specified.
  if (removeNonAscii):
    cleanedText = cleanedText.encode("ascii", "ignore").decode("ascii")  # Keep only ASCII characters.

  # Convert text to lowercase if specified.
  if (lowercase):
    cleanedText = cleanedText.lower()  # Lowercase all characters.

  # Expand contractions if specified.
  if (handleContractions):
    cleanedText = contractions.fix(cleanedText)  # Expand contractions (e.g., don't → do not).

  # Remove special characters and punctuation if specified.
  if (removeSpecialChars):
    # Preserve basic sentence punctuation .,!? while removing others
    cleanedText = re.sub(r"[^a-zA-Z0-9\s\.!?,]", "", cleanedText)

  # If not removing non-ASCII, restore original non-ASCII from input best-effort
  # This block ensures that when removeNonAscii=False, non-ASCII characters like 'é' are preserved
  if (not removeNonAscii):
    # Rebuild cleanedText from original by filtering only removed categories
    # Keep letters, digits, spaces, and common punctuation
    cleanedText = "".join([ch for ch in text if re.match(r"[\w\s\.!?,]", ch) or ord(ch) > 127])
    if lowercase:
      cleanedText = cleanedText.lower()

  # Normalize whitespace around punctuation.
  # Remove extra spaces before punctuation.
  cleanedText = re.sub(r"\s+([?.!,])", r"\1", cleanedText)

  # Lemmatize words if specified.
  if (lemmatize):
    from nltk.stem import WordNetLemmatizer  # Import WordNetLemmatizer from NLTK.
    lemmatizer = WordNetLemmatizer()  # Create a lemmatizer instance.
    cleanedText = " ".join([lemmatizer.lemmatize(word) for word in cleanedText.split()])  # Lemmatize each word.

  # Remove stopwords if specified.
  # Stopwords are common words that may not add significant meaning to the text.
  if (removeStopwords):
    from nltk.corpus import stopwords  # Import stopwords from NLTK.
    stopWords = set(stopwords.words("english"))  # Get English stopwords.
    cleanedText = " ".join([word for word in cleanedText.split() if word.lower() not in stopWords])  # Remove stopwords.

  # Replace multiple spaces with a single space if specified.
  if (normalizeWhitespace):
    # Normalize whitespace.
    cleanedText = " ".join(cleanedText.split())

  # Ensure terminal punctuation remains if originally present
  cleanedText = re.sub(r"\s+([\.!?,])", r"\1", cleanedText)

  # Remove common words if specified.
  if (removeCommonWords):
    # Get word frequency distribution.
    wordFreq = nltk.FreqDist(cleanedText.split())
    # Get the N most common words.
    commonWords = set([word for word, freq in wordFreq.most_common(numOfCommonWords)])
    # Remove common words.
    cleanedText = " ".join([word for word in cleanedText.split() if word not in commonWords])

  # Remove non-English words if specified.
  if (removeNonEnglishWords):
    from nltk.corpus import words  # Import words corpus from NLTK.
    # Get a set of valid English words.
    englishWords = set(words.words())
    # Keep only English words.
    cleanedText = " ".join([word for word in cleanedText.split() if word.lower() in englishWords])

    # Return the cleaned text.
  return cleanedText


def IsSyntacticallyValid(nlpModel, text: str) -> bool:
  r'''
  Perform a robust heuristic check to determine the syntactic validity of a given text.

  This function evaluates whether a string resembles a grammatically correct English
  sentence by applying a series of structural and lexical heuristics. It is designed
  to filter out gibberish, repetitive loops, and nonsensical word salads that might
  otherwise pass basic length checks. The validation pipeline includes checks for
  proper capitalization, terminal punctuation, the presence of core grammatical
  components (verbs and subjects), structural anomaly detection, excessive word
  repetition, and a baseline stop word ratio to ensure natural language flow.

  Heuristics applied:
    1. Length Check: The text must contain at least three words.
    2. Capitalization: The text must start with a capital letter.
    3. Terminal Punctuation: The text must end with a single standard punctuation mark.
    4. Verb Presence: The text must contain at least one verb or auxiliary verb.
    5. Subject Presence: The text must contain a nominal subject (active, passive, or clausal).
    6. Word Salad Detection: Checks for nonsensical sequences like noun-determiner-preposition.
    7. Repetition Check: Flags text where a single word dominates (>40% of tokens).
    8. Stop Word Ratio: Ensures a reasonable proportion of common stop words in longer texts.

  Parameters:
    nlpModel (spacy.Language): A loaded spaCy language model instance.
    text (str): The input text string to evaluate.

  Returns:
    bool: True if the text passes all checks, False otherwise.

  Raises:
    ValueError: If the provided nlpModel is None.

  Examples
  --------
  .. code-block:: python

    import spacy
    from HMB.TextHelper import IsSyntacticallyValid

    # Load a spaCy English model (e.g., "en_core_web_sm").
    nlp = spacy.load("en_core_web_sm")

    # Example text to validate.
    text = "The quick brown fox jumps over the lazy dog."

    # Check if the text is syntactically valid.
    isValid = IsSyntacticallyValid(nlp, text)
    print(f"Is the text syntactically valid? {isValid}")  # Expected output: True.
    isValid = IsSyntacticallyValid(nlp, "Mat the on sat cat the.")
    print(f"Is the text syntactically valid? {isValid}")  # Expected output: False.
    isValid = IsSyntacticallyValid(nlp, "The the the the the cat.")
    print(f"Is the text syntactically valid? {isValid}")  # Expected output: False.
    isValid = IsSyntacticallyValid(nlp, "sat on the mat.")
    print(f"Is the text syntactically valid? {isValid}")  # Expected output: False.
    isValid = IsSyntacticallyValid(nlp, "The cat sat on the mat..")
    print(f"Is the text syntactically valid? {isValid}")  # Expected output: False.
  '''

  # Validate that the provided NLP model is not None.
  if (nlpModel is None):
    # Raise an error if the model is invalid.
    raise ValueError("The nlpModel parameter must be a valid spaCy language model instance.")

  # Check if the text is empty or contains fewer than three words.
  if (not text or len(text.split()) < 3):
    # Return False for invalid length or empty text.
    return False

  # Verify that the text starts with a capital letter.
  if (not text[0].isupper()):
    # Return False if the capitalization rule is violated.
    return False

  # Verify that the text ends with valid terminal punctuation.
  if (text[-1] not in ".!?"):
    # Return False if the punctuation rule is violated.
    return False

  # Check for anomalous terminal punctuation like multiple dots or mixed marks.
  if (len(text) > 1 and text[-2:] in ["..", "??", "!!", ".?", "?.", "!?", "?!"]):
    # Return False if the text ends with gibberish punctuation.
    return False

  # Parse the input text using the provided spaCy language model.
  doc = nlpModel(text)

  # Determine if the document contains at least one verb or auxiliary verb.
  hasVerb = any(token.pos_ == "VERB" or token.pos_ == "AUX" for token in doc)

  # Check whether a verb was found in the parsed document.
  if (not hasVerb):
    # Return False if no verb is present.
    return False

  # Determine if the document contains a valid subject dependency.
  hasSubject = any(token.dep_ in ["nsubj", "nsubjpass", "csubj"] for token in doc)

  # Check whether a subject was found in the parsed document.
  if (not hasSubject):
    # Return False if no subject is present.
    return False

  # Convert the parsed document into a list of tokens for sequential analysis.
  tokens = list(doc)

  # Ensure there are at least three tokens to perform the word salad check.
  if (len(tokens) >= 3):
    # Extract the first three tokens for structural pattern matching.
    t0, t1, t2 = tokens[0], tokens[1], tokens[2]

    # Detect gibberish patterns where a noun is incorrectly followed by a determiner and a preposition.
    if (t0.pos_ in ["NOUN", "PROPN"] and t1.pos_ == "DET" and t2.pos_ == "ADP"):
      # Return False if the nonsensical structural pattern is detected.
      return False

  # Initialize a dictionary to calculate word frequencies for repetition detection.
  wordFrequencies = {}

  # Iterate through each token to count word occurrences.
  for token in tokens:
    # Skip punctuation and spaces for frequency counting.
    if (not token.is_punct and not token.is_space):
      # Get the lowercase text of the token.
      wordText = token.text.lower()

      # Increment the count for the current word.
      wordFrequencies[wordText] = wordFrequencies.get(wordText, 0) + 1

  # Check if any single word dominates the sentence to detect repetitive gibberish.
  if (wordFrequencies):
    # Find the maximum frequency of any single word.
    maxFrequency = max(wordFrequencies.values())

    # Calculate the total number of valid words.
    totalWords = sum(wordFrequencies.values())

    # Check if the most frequent word appears too often.
    if (totalWords > 0 and (maxFrequency / totalWords) > 0.4):
      # Return False if the text is overly repetitive.
      return False

  # Define a set of common English stop words for ratio calculation.
  commonStopwords = {
    "the", "a", "an", "is", "are", "was", "were", "on",
    "in", "at", "to", "for", "of", "and", "but", "with",
    "or", "as", "by", "from", "that", "this", "it", "its",
    "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must",
    "if", "then", "else", "when", "where", "while",
    "which", "who", "whom", "whose", "what", "how", "why",
    "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "now"
  }

  # Create a set of unique lowercase words from the input text.
  words = set(w.lower() for w in text.split())

  # Calculate the number of unique stop words present in the text.
  stopCount = len(words.intersection(commonStopwords))

  # Check if the text lacks stop words despite having more than four unique words.
  if (stopCount == 0 and len(words) > 4):
    # Return False if the stop word ratio is unreasonably low for the given length.
    return False

  # Return True as the text has passed all syntactic validity checks.
  return True


def CalculateDynamicTemperature(nlpModel, text, baseTemp=1.0):
  r'''
  Analyzes sentence content and returns the optimal dynamic temperature for generation.

  This function evaluates the text to determine if it is factual, mixed, or descriptive,
  and adjusts the base temperature accordingly to prevent hallucinations in factual text
  while encouraging creativity in descriptive text.

  Heuristics applied:
    1. Numeric Presence: Detects numbers and quantities.
    2. Temporal Entities: Detects dates and times.
    3. Proper Noun Density: Measures the ratio of proper nouns to total tokens.
    4. Lexical Density: Measures the ratio of nouns and adjectives to total tokens.
    5. Passive Voice: Detects the presence of passive voice constructions.
    6. Average Word Length: Technical texts typically use longer, more complex words.
    7. Punctuation Density: Formal texts often contain more complex punctuation marks.

  Parameters:
    nlpModel (spacy.Language): A loaded spaCy language model instance.
    text (str): The input text to analyze for content type.
    baseTemp (float): The baseline temperature to adjust from. Default is 1.0.

  Returns:
    float: The dynamically adjusted temperature based on the content analysis.

  Raises:
    ValueError: If the provided nlpModel is None.

  Examples
  --------
  .. code-block:: python

    import spacy
    from HMB.TextHelper import CalculateDynamicTemperature

    # Load a spaCy English model (e.g., "en_core_web_sm").
    nlp = spacy.load("en_core_web_sm")

    # Example text to analyze.
    text = "The study was conducted on January 15, 2023, and included 150 participants."
    # Calculate the dynamic temperature for the text.
    dynamicTemp = CalculateDynamicTemperature(nlp, text, baseTemp=1.0)
    print(f"Dynamic temperature: {dynamicTemp:.2f}")  # Expected output: < 1.0 for factual content.
    text = "Once upon a time, in a land far away, there lived a brave knight."
    dynamicTemp = CalculateDynamicTemperature(nlp, text, baseTemp=1.0)
    print(f"Dynamic temperature: {dynamicTemp:.2f}")  # Expected output: > 1.0 for descriptive content.
  '''

  # Validate that the input text is a non-empty string.
  if (not text or not isinstance(text, str)):
    # Return the unadjusted base temperature for invalid input.
    return baseTemp

  if (nlpModel is None):
    # Raise an error if the model is invalid.
    raise ValueError("The nlpModel parameter must be a valid spaCy language model instance.")

  # Parse the input text using the provided spaCy language model.
  doc = nlpModel(text)

  # Guard against empty parsed documents to prevent division by zero.
  if (len(doc) == 0):
    # Return the unadjusted base temperature for empty documents.
    return baseTemp

  # Initialize the factual score accumulator.
  factualScore = 0.0

  # Calculate the total number of tokens in the document.
  totalTokens = len(doc)

  # Detect the presence of numeric tokens or quantities.
  hasNumbers = any(token.like_num or token.pos_ == "NUM" for token in doc)

  # Add to the factual score if numbers are detected.
  if (hasNumbers):
    # Increase the factual score for numeric content.
    factualScore += 0.3

  # Detect the presence of date, time, or quantity entities.
  hasDate = any(ent.label_ in ["DATE", "TIME", "CARDINAL", "QUANTITY"] for ent in doc.ents)

  # Add to the factual score if temporal or quantity entities are detected.
  if (hasDate):
    # Increase the factual score for temporal content.
    factualScore += 0.3

  # Count the total number of proper nouns in the document.
  properNouns = sum(1 for token in doc if token.pos_ == "PROPN")

  # Add to the factual score based on the density of proper nouns.
  if (properNouns > 0 and totalTokens > 0):
    # Increase the factual score proportionally to proper noun density, capped at 0.4.
    factualScore += min(0.4, (properNouns / totalTokens) * 2)

  # Count the total number of nouns and adjectives for lexical density.
  nounAdjCount = sum(1 for token in doc if token.pos_ in ["NOUN", "ADJ"])

  # Calculate the ratio of nouns and adjectives to total tokens.
  nounAdjRatio = nounAdjCount / totalTokens if (totalTokens > 0) else 0

  # Add to the factual score if the lexical density is exceptionally high.
  if (nounAdjRatio > 0.6):
    # Increase the factual score for highly dense technical text.
    factualScore += 0.2

  # Count the tokens indicating passive voice constructions.
  passiveCount = sum(1 for token in doc if token.dep_ == "auxpass" or token.tag_ == "VBN")

  # Determine if the text has a significant amount of passive voice.
  hasPassive = (passiveCount > 0 and (passiveCount / totalTokens) > 0.1)

  # Add to the factual score if passive voice is frequently used.
  if (hasPassive):
    # Increase the factual score for academic or formal passive structures.
    factualScore += 0.2

  # Calculate the average length of alphabetic words in the text.
  wordLengths = [len(token.text) for token in doc if token.is_alpha]

  # Compute the mean word length, defaulting to zero if no alphabetic tokens exist.
  avgWordLength = sum(wordLengths) / len(wordLengths) if (wordLengths) else 0

  # Add to the factual score if the average word length indicates complex vocabulary.
  if (avgWordLength > 5.5):
    # Increase the factual score for longer, more technical words.
    factualScore += 0.15

  # Count the total number of punctuation marks in the document.
  punctCount = sum(1 for token in doc if token.is_punct)

  # Calculate the ratio of punctuation marks to total tokens.
  punctDensity = punctCount / totalTokens if (totalTokens > 0) else 0

  # Add to the factual score if the punctuation density indicates complex sentence structures.
  if (punctDensity > 0.08):
    # Increase the factual score for texts with high punctuation density.
    factualScore += 0.1

  # Determine the content category and adjust the temperature based on the factual score.
  if (factualScore > 0.6):
    # Set the category to Factual for high factual scores.
    category = "Factual"
    # Calculate the adjusted temperature for conservative generation.
    adjustedTemp = max(0.5, baseTemp * 0.5)
  # Check if the factual score indicates mixed content.
  elif (factualScore > 0.3):
    # Set the category to Mixed for moderate factual scores.
    category = "Mixed"
    # Calculate the adjusted temperature for balanced generation.
    adjustedTemp = baseTemp * 0.8
  # Default to descriptive content for low factual scores.
  else:
    # Set the category to Descriptive for low factual scores.
    category = "Descriptive"
    # Calculate the adjusted temperature for creative generation.
    adjustedTemp = min(1.5, baseTemp * 1.2)

  # Apply a global clamp to prevent the temperature from reaching extreme values.
  finalTemp = max(0.1, min(2.0, adjustedTemp))

  # Print a diagnostic message detailing the calculation results.
  print(
    f"Dynamic Temp Calculation: {category} content detected (Score: {factualScore:.2f}). "
    f"Base Temp: {baseTemp}, Adjusted Temp: {finalTemp:.2f}"
  )

  # Return the final calculated temperature.
  return finalTemp


def CalculateContextualSimilarity(
  semanticModel,
  originalToken: str,
  synonym: str,
  context: str
) -> float:
  r'''
  Calculates how semantically similar a synonym is to the surrounding context.

  This function uses a sentence transformer model to encode the context with the original token
  and with the synonym, and then computes the cosine similarity between the two embeddings.
  The context should contain a placeholder "_____" where the token is replaced, allowing for
  a direct comparison of how well the original token and the synonym fit within the same context.
  If the placeholder is not found, the function will construct simple sentences for comparison,
  but this may lead to less accurate similarity scores.

  Heuristics applied:
    1. Identity Check: Returns 1.0 immediately if the original token and synonym are identical.
    2. Input Validation: Returns a neutral score for empty inputs or missing models.
    3. Placeholder Replacement: Uses the provided placeholder to maintain exact syntactic structure.
    4. Fallback Construction: Constructs a neutral sentence if the placeholder is missing.
    5. Embedding Normalization: Ensures tensors are 2D before computing cosine similarity.

  Parameters:
    semanticModel (sentence_transformers.SentenceTransformer): A pre-initialized sentence transformer model.
    originalToken (str): The original word or phrase that is being replaced.
    synonym (str): The candidate synonym that is being evaluated for contextual fit.
    context (str): The surrounding text containing a placeholder "_____" where the original token is located.
      This allows the method to compare the semantic fit of the original token and the synonym within the same
      context. If the placeholder is not present, the method will attempt a fallback comparison, but for best
      results, the context should include "_____" to indicate where the token is being replaced.

  Returns:
    float: A similarity score between 0.0 and 1.0 indicating how well the synonym fits semantically within the context compared to the original token. Higher scores suggest that the synonym is more consistent with the surrounding text, while lower scores indicate that the synonym may not fit as well within the context. This score can be used to filter or rank synonym replacements to ensure they maintain the intended meaning and coherence of the text.

  Examples
  --------
  .. code-block:: python

    from sentence_transformers import SentenceTransformer
    from HMB.TextHelper import CalculateContextualSimilarity

    # Load a pre-trained sentence transformer model.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Define the original token, synonym, and context with a placeholder.
    originalToken = "happy"
    synonym = "joyful"
    context = "She felt very _____ after receiving the good news."

    # Calculate the contextual similarity score.
    similarityScore = CalculateContextualSimilarity(model, originalToken, synonym, context)
    # Expected output: A float value between 0.0 and 1.0 indicating the semantic similarity.
    print(f"Contextual similarity score: {similarityScore:.4f}")
  '''

  import torch

  # Validate that the semantic model is available for inference.
  if (semanticModel is None):
    # Return a neutral score if the model is not available.
    return 0.5

  # Validate that the original token and synonym are not empty.
  if (not originalToken or not synonym):
    # Return a neutral score for invalid inputs.
    return 0.5

  # Check if the original token and synonym are identical to save compute.
  if (originalToken.lower() == synonym.lower()):
    # Return a perfect score for identical tokens.
    return 1.0

  # Attempt to calculate the similarity with comprehensive error handling.
  try:
    # Initialize the context strings for encoding.
    originalContext = ""
    synonymContext = ""

    # Check if the context contains the designated placeholder.
    if ("_____" in context):
      # Replace the placeholder with the original token for the first context.
      originalContext = context.replace("_____", originalToken)
      # Replace the placeholder with the synonym for the second context.
      synonymContext = context.replace("_____", synonym)
    else:
      # Construct a neutral fallback sentence for the original token.
      originalContext = f"The {originalToken} is relevant here."
      # Construct a neutral fallback sentence for the synonym.
      synonymContext = f"The {synonym} is relevant here."
      # Print a warning message for debugging purposes.
      # print("⚠️ No placeholder '_____' found in context. Using fallback construction.")

    # Encode the original context into a tensor embedding.
    originalEmbedding = semanticModel.encode(originalContext, convert_to_tensor=True)
    # Encode the synonym context into a tensor embedding.
    synonymEmbedding = semanticModel.encode(synonymContext, convert_to_tensor=True)

    # Ensure the original embedding is 2D for the cosine similarity calculation.
    if (len(originalEmbedding.shape) == 1):
      # Reshape the 1D tensor to 2D by adding a batch dimension.
      originalEmbedding = originalEmbedding.unsqueeze(0)

    # Ensure the synonym embedding is 2D for the cosine similarity calculation.
    if (len(synonymEmbedding.shape) == 1):
      # Reshape the 1D tensor to 2D by adding a batch dimension.
      synonymEmbedding = synonymEmbedding.unsqueeze(0)

    # Calculate the cosine similarity between the two embeddings along the feature dimension.
    similarity = torch.nn.functional.cosine_similarity(originalEmbedding, synonymEmbedding, dim=1)
    # Extract the similarity value as a standard Python float.
    similarityValue = similarity.item()

    # Return the successfully calculated contextual similarity score.
    return similarityValue

  # Catch any unexpected runtime exceptions during model inference.
  except Exception as e:
    # Print a diagnostic warning message containing the exception details.
    print(f"⚠️ Contextual similarity calculation failed: {e}")
    # Return a neutral fallback score upon encountering an error.
    return 0.5


def GetContextualSynonym(nlpModel, maskPipeline, stemmer, word: str, context: str) -> str:
  r'''
  Retrieves a contextual synonym for a specified word based on the provided surrounding text.

  This function employs a multi-stage strategy to ensure high-quality synonym replacement:
    1. Validation: Verifies the target word (or its morphological variant) exists within the context.
    2. Inference: Utilizes a Masked Language Model (e.g., RoBERTa) to predict semantically appropriate substitutions.
    3. Filtering: Excludes exact matches and morphological variants using lemmatization and stemming.
    4. Fallback: If the model fails to identify a distinct synonym, a lexical synonym is retrieved from WordNet.

  Parameters:
    nlpModel (spacy.Language): A loaded spaCy language model instance for tokenization and lemmatization.
    maskPipeline (transformers.Pipeline): A pre-initialized masked language model pipeline for inference.
    stemmer (nltk.stem.PorterStemmer): A pre-initialized stemmer for morphological filtering.
    word (str): The target word for which a synonym is required.
    context (str): The sentence or phrase containing the target word.

  Returns:
    str: A string representing the top predicted contextual synonym.

  Examples
  --------
  .. code-block:: python

    import spacy
    from transformers import pipeline
    from nltk.stem import PorterStemmer
    from HMB.TextHelper import GetContextualSynonym

    # Load a spaCy English model (e.g., "en_core_web_sm").
    nlp = spacy.load("en_core_web_sm")
    # Initialize a masked language model pipeline (e.g., RoBERTa).
    maskPipeline = pipeline("fill-mask", model="roberta-base")
    # Initialize a Porter stemmer.
    stemmer = PorterStemmer()

    # Define the target word and context.
    targetWord = "happy"
    contextSentence = "She felt very happy after receiving the good news."

    # Retrieve a contextual synonym for the target word.
    synonym = GetContextualSynonym(nlp, maskPipeline, stemmer, targetWord, contextSentence)
    print(f"Contextual synonym for '{targetWord}': {synonym}")
  '''

  from nltk.corpus import wordnet

  # Return the original word immediately if the context is empty.
  if (not context):
    # Return the original word as no context is available for inference.
    return word

  # Convert the target word to lowercase for case-insensitive matching.
  wordLower = word.lower()

  # Parse the context text using the provided spaCy language model.
  doc = nlpModel(context)

  # Initialize the target token variable to None.
  targetToken = None

  # Iterate through each token in the parsed document to find a lemma match.
  for token in doc:
    # Skip tokens that are punctuation or whitespace.
    if (token.is_punct or token.is_space):
      # Continue to the next token.
      continue

    # Check if the token's lemma matches the target word.
    if (token.lemma_.lower() == wordLower):
      # Assign the matched token as the target token.
      targetToken = token
      # Break the loop as the target has been found.
      break

  # Check if the target token was not found via lemma matching.
  if (targetToken is None):
    # Compute the stem of the target word for secondary matching.
    wordStem = stemmer.stem(wordLower)

    # Iterate through each token in the parsed document to find a stem match.
    for token in doc:
      # Skip tokens that are punctuation or whitespace.
      if (token.is_punct or token.is_space):
        # Continue to the next token.
        continue

      # Compute the stem of the current token's text.
      tokenStem = stemmer.stem(token.text.lower())

      # Check if the token's stem matches the target word's stem.
      if (tokenStem == wordStem):
        # Assign the matched token as the target token.
        targetToken = token
        # Break the loop as the target has been found.
        break

  # Check if the target token is still None after both matching attempts.
  if (targetToken is None):
    # Return the original word as it was not found in the context.
    return word

  # Retrieve the mask token string from the masked language model's tokenizer.
  maskToken = maskPipeline.tokenizer.mask_token

  # Determine the start character index of the target token in the context.
  startIdx = targetToken.idx

  # Determine the end character index of the target token in the context.
  endIdx = startIdx + len(targetToken.text)

  # Construct the masked context by replacing the target token with the mask token.
  maskedContext = context[:startIdx] + maskToken + context[endIdx:]

  # Attempt to perform masked language model inference and filtering.
  try:
    # Request the top 200 predictions from the masked language model pipeline.
    predictions = maskPipeline(maskedContext, top_k=200)

    # Extract the lowercase lemma of the original target token.
    originalLemma = targetToken.lemma_.lower()

    # Extract the stem of the original target token's text.
    originalStem = stemmer.stem(targetToken.text.lower())

    # Iterate through each prediction returned by the masked language model.
    for prediction in predictions:
      # Extract the predicted token string and strip leading/trailing whitespace.
      candidate = prediction["token_str"].strip()

      # Remove any leading or trailing non-word characters from the candidate.
      candidate = re.sub(r"^[^\w]|[^\w]$", "", candidate)

      # Check if the candidate string is empty after cleaning.
      if (not candidate):
        # Continue to the next prediction.
        continue

      # Convert the candidate string to lowercase for comparison.
      candidateLower = candidate.lower()

      # Check if the candidate is an exact match for the original word.
      if (candidateLower == wordLower):
        # Continue to the next prediction to exclude exact matches.
        continue

      # Attempt to parse the candidate using the spaCy model for morphological filtering.
      try:
        # Parse the candidate text into a spaCy document.
        candidateDoc = nlpModel(candidate)

        # Check if the parsed document contains at least one token.
        if (candidateDoc):
          # Extract the lowercase lemma of the first token in the candidate document.
          candidateLemma = candidateDoc[0].lemma_.lower()

          # Extract the stem of the candidate string.
          candidateStem = stemmer.stem(candidateLower)

          # Check if the candidate's lemma matches the original lemma.
          if (candidateLemma == originalLemma):
            # Continue to the next prediction to exclude morphological variants.
            continue

          # Check if the candidate's stem matches the original stem.
          if (candidateStem == originalStem):
            # Continue to the next prediction to exclude morphological variants.
            continue

      # Handle any exceptions that occur during spaCy processing of the candidate.
      except Exception:
        # Fallback to stem check if spaCy processing fails.
        if (stemmer.stem(candidateLower) == originalStem):
          # Continue to the next prediction.
          continue

      # Calculate the character difference between the candidate and the original word.
      charDifference = sum(1 for a, b in zip(candidateLower, wordLower) if a != b)

      # Check if the candidate is likely an orthographic spelling variant.
      if (len(candidate) == len(wordLower) and charDifference <= 2):
        # Continue to the next prediction to exclude spelling variants.
        continue

      # Check if the candidate contains the original word as a substring or vice versa.
      if (wordLower in candidateLower or candidateLower in wordLower):
        # Check if the lengths are different to allow exact length matches that are distinct.
        if (len(candidate) != len(wordLower)):
          # Continue to the next prediction to exclude substrings.
          continue

      # Return the first valid contextual synonym found.
      return candidate

  # Handle any unexpected exceptions during masked language model inference.
  except Exception:
    # Proceed to the lexical fallback if inference fails.
    pass

  # Attempt to retrieve a lexical synonym from WordNet as a fallback.
  try:
    # Retrieve all synsets for the original word from WordNet.
    synsets = wordnet.synsets(wordLower)

    # Iterate through each synset associated with the word.
    for syn in synsets:
      # Iterate through each lemma in the current synset.
      for lemma in syn.lemmas():
        # Extract the synonym name and replace underscores with spaces.
        synonym = lemma.name().replace("_", " ")

        # Check if the synonym is different from the original word.
        if (synonym.lower() != wordLower):
          # Return the first valid WordNet synonym found.
          return synonym

  # Handle any exceptions that occur during WordNet lookup.
  except Exception:
    # Proceed to the final fallback if WordNet lookup fails.
    pass

  # Return the original word if all synonym retrieval methods fail.
  return word


def BackTranslateDeepRestructure(
  sourceToPivotTokenizer,
  sourceToPivotModel,
  pivotToSourceTokenizer,
  pivotToSourceModel,
  text: str,
  device: str = "cpu"
) -> str:
  r'''
  Performs deep restructuring via back-translation through an intermediate pivot language.

  This function takes the input text, translates it from the source language to a pivot
  language using a provided translation model, and then translates it back to the source
  language using a second translation model. This process can create more significant
  changes in sentence structure and word choice compared to simple paraphrasing, as the
  translation models may rephrase sentences in ways that are more natural in the pivot
  language and then re-translate them back with a different structure. This is a powerful
  technique for generating more human-like variations of the original text while still
  preserving the core meaning.

  Parameters:
    sourceToPivotTokenizer (transformers.PreTrainedTokenizer): Tokenizer for the source to pivot model.
    sourceToPivotModel (transformers.PreTrainedModel): Model for translating source to pivot language.
    pivotToSourceTokenizer (transformers.PreTrainedTokenizer): Tokenizer for the pivot to source model.
    pivotToSourceModel (transformers.PreTrainedModel): Model for translating pivot to source language.
    text (str): The input text to be back-translated.
    device (str): The compute device to use for model inference.

  Returns:
    str: The back-translated text, which has been processed through the pivot language.

  Examples
  --------
  .. code-block:: python

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from HMB.TextHelper import BackTranslateDeepRestructure

    # Load the source to pivot translation model and tokenizer (e.g., English to French).
    sourceToPivotTokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    sourceToPivotModel = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

    # Load the pivot to source translation model and tokenizer (e.g., French to English).
    pivotToSourceTokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    pivotToSourceModel = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

    # Define the input text to be back-translated.
    inputText = "The quick brown fox jumps over the lazy dog."

    # Perform back-translation through the pivot language.
    backTranslatedText = BackTranslateDeepRestructure(
        sourceToPivotTokenizer,
        sourceToPivotModel,
        pivotToSourceTokenizer,
        pivotToSourceModel,
        inputText,
        device="cpu"
    )
    # Print the back-translated text, which may have a different structure and word choice.
    # Example output: Back-translated text: The fast brown fox jumps on the lazy dog.
    print("Back-translated text:", backTranslatedText)
  '''

  import torch

  # Validate that the input text is a non-empty string.
  if (not text or not isinstance(text, str)):
    # Return the original text immediately if the input is invalid.
    return text

  # Guard against pure whitespace to avoid unnecessary model inference.
  # Translation models (like MarianMT) use tokenizers that strip pure whitespace,
  # so we short-circuit here to preserve the original string and save compute.
  if (not text.strip()):
    # Return the original whitespace string.
    return text

  # Attempt the first translation step with comprehensive error handling.
  try:
    # Disable gradient computation to optimize memory usage during inference.
    with torch.inference_mode():
      # Tokenize the input text for the source to pivot model.
      sourceInputs = sourceToPivotTokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
      ).to(device)

      # Generate the intermediate translation in the pivot language.
      pivotTransIds = sourceToPivotModel.generate(
        **sourceInputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
      )

      # Decode the generated token IDs into the pivot language text.
      pivotText = sourceToPivotTokenizer.decode(pivotTransIds[0], skip_special_tokens=True)

    # Guard against empty intermediate translation results.
    if (not pivotText or not pivotText.strip()):
      # Print a warning message indicating the first translation step failed.
      print("⚠️ Back-translation failed at source to pivot step. Returning original text.")

      # Return the original text as the intermediate step yielded no output.
      return text

    # Attempt the second translation step to return to the source language.
    with torch.inference_mode():
      # Tokenize the pivot text for the pivot to source model.
      pivotInputs = pivotToSourceTokenizer(
        pivotText,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
      ).to(device)

      # Generate the final translation back in the source language.
      sourceTransIds = pivotToSourceModel.generate(
        **pivotInputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
      )

      # Decode the generated token IDs into the final source language text.
      finalText = pivotToSourceTokenizer.decode(sourceTransIds[0], skip_special_tokens=True)

    # Guard against empty final translation results.
    if (not finalText or not finalText.strip()):
      # Print a warning message indicating the second translation step failed.
      print("⚠️ Back-translation failed at pivot to source step. Returning original text.")

      # Return the original text as the final step yielded no output.
      return text

    # Return the successfully back-translated text.
    return finalText

  # Catch any unexpected runtime exceptions during the translation process.
  except Exception as e:
    # Print a diagnostic warning message containing the exception details.
    print(f"⚠️ Back-translation failed with error: {e}")

    # Return the original text as a safe fallback upon encountering an error.
    return text


def CalculateMaskedLikelihoodDelta(maskPipeline, context, originalWord, replacementWord, offset, errorLength):
  r'''
  Calculates the probability delta between an original word and a replacement word
  at a specific masked position in the context using a Masked Language Model.

  Parameters:
    maskPipeline (transformers.Pipeline): The initialized fill-mask pipeline.
    context (str): The full sentence containing the error.
    originalWord (str): The original erroneous word.
    replacementWord (str): The suggested replacement word.
    offset (int): The character index where the error starts.
    errorLength (int): The character length of the error span.

  Returns:
    float: The probability delta (P(Replacement) - P(Original)). A positive value indicates the replacement is more natural.

  Examples
  --------
  .. code-block:: python

    from transformers import pipeline
    from HMB.TextHelper import CalculateMaskedLikelihoodDelta

    # Initialize a masked language model pipeline (e.g., RoBERTa).
    maskPipeline = pipeline("fill-mask", model="roberta-base")

    # Define the context, original word, replacement word, and error position.
    context = "She felt very happy after receiving the good news."
    originalWord = "happy"
    replacementWord = "joyful"
    offset = context.index(originalWord)
    errorLength = len(originalWord)

    # Calculate the likelihood delta for the replacement.
    delta = CalculateMaskedLikelihoodDelta(maskPipeline, context, originalWord, replacementWord, offset, errorLength)
    print(f"Likelihood delta for replacement: {delta:.6f}")
  '''

  # Check if the context is empty.
  if (len(context.strip()) <= 0):
    return 0.0

  # Construct the masked context string by replacing the error span with the mask token.
  maskedText = context[:offset] + maskPipeline.tokenizer.mask_token + context[offset + errorLength:]

  import torch

  try:
    # Determine the device the model is currently running on.
    device = maskPipeline.device

    # Tokenize the masked text and move tensors to the correct device.
    inputs = maskPipeline.tokenizer(maskedText, return_tensors="pt").to(device)

    # Extract the mask token ID from the tokenizer.
    maskTokenId = maskPipeline.tokenizer.mask_token_id

    # Find the column indices of the mask token in the input sequence.
    maskIndices = torch.where(inputs["input_ids"] == maskTokenId)[1]

    # Return zero if the mask token was not found in the tokenized input.
    if (len(maskIndices) == 0):
      # Return zero as a safe fallback when masking fails.
      return 0.0

    # Extract the scalar integer index of the first mask token.
    maskIndex = maskIndices[0].item()

    # Disable gradient computation to optimize memory during inference.
    with torch.no_grad():
      # Execute the forward pass through the model.
      outputs = maskPipeline.model(**inputs)

      # Isolate the 1D logits tensor at the mask token position.
      logits = outputs.logits[0, maskIndex, :]

    # Apply softmax to convert raw logits into a probability distribution.
    probabilities = torch.softmax(logits, dim=-1)

    # Define a helper to calculate the aggregate probability of a multi-token word.
    def _calculateWordProbability(word):
      # Encode the word using the tokenizer to get its BPE token IDs.
      # Prepend a space to simulate mid-sentence tokenization behavior.
      encodedTokens = maskPipeline.tokenizer.encode(" " + word.strip(), add_special_tokens=False)

      # Return zero if the word produces no valid tokens.
      if (not encodedTokens):
        # Return zero for empty token sequences.
        return 0.0

      # For single-token words, return the direct probability.
      if (len(encodedTokens) == 1):
        # Extract the probability of the single token.
        return probabilities[encodedTokens[0]].item()

      # For multi-token words, return the product of all subword probabilities.
      # This approximates the joint probability P(w) = P(t1) * P(t2) * ... * P(tn).
      jointProbability = 1.0

      # Iterate through each subword token ID.
      for tokenId in encodedTokens:
        # Multiply the joint probability by the current subword probability.
        jointProbability *= probabilities[tokenId].item()

      # Return the aggregated joint probability.
      return jointProbability

    # Calculate the probability of the original word.
    originalProb = _calculateWordProbability(originalWord)

    # Calculate the probability of the replacement word.
    replacementProb = _calculateWordProbability(replacementWord)

    # Calculate the delta score representing the likelihood improvement.
    deltaScore = replacementProb - originalProb

    # Return the calculated delta score to the caller.
    return deltaScore

  except Exception as e:
    # Print a diagnostic warning if the likelihood calculation fails.
    print(f"⚠️ Masked likelihood calculation failed: {e}")

    # Return 0.0 as a safe fallback.
    return 0.0


def GetInflectedForm(token, targetTag):
  r'''
  Retrieves a dynamically inflected morphological form of a given spaCy token.

  Parameters:
    token (spacy.tokens.Token): The spaCy token to be inflected.
    targetTag (str): The target morphological tag (e.g., "VBD", "VBN").

  Returns:
    str: The inflected word, or the original word if inflection fails.
  '''

  # Define a fallback dictionary for common irregular verbs in case pyinflect fails.
  irregularFallbacks = {
    "VBD": {  # Simple Past
      "go": "went", "buy": "bought", "do": "did", "have": "had",
      "is": "was", "are": "were", "goes": "went", "has": "had"
    },
    "VBN": {  # Past Participle
      "go": "gone", "buy": "bought", "do": "done", "have": "had",
      "is": "been", "are": "been", "goes": "gone", "has": "had"
    }
  }

  # Attempt to dynamically inflect the token using pyinflect.
  try:
    inflectedForms = token._.inflect(targetTag, inflect_oov=True)
    if (inflectedForms and len(inflectedForms) > 0):
      inflectedWord = inflectedForms[0] if isinstance(inflectedForms, tuple) else inflectedForms
      if (token.text[0].isupper() and len(inflectedWord) > 0):
        inflectedWord = inflectedWord[0].upper() + inflectedWord[1:]
      return inflectedWord
  except Exception:
    # pyinflect failed or is not loaded. Proceed to fallback.
    pass

  # --- FALLBACK MECHANISM ---
  # If pyinflect failed, check our hardcoded dictionary for irregular verbs.
  lemma = token.lemma_.lower()
  if (targetTag in irregularFallbacks and lemma in irregularFallbacks[targetTag]):
    fallbackWord = irregularFallbacks[targetTag][lemma]
    # Preserve capitalization
    if (token.text[0].isupper()):
      fallbackWord = fallbackWord[0].upper() + fallbackWord[1:]
    return fallbackWord

  # Ultimate fallback: return original text.
  return token.text


def GetPastParticipleForm(token):
  r'''
  Retrieves the past participle form (VBN) of a given spaCy token.

  Parameters:
    token (spacy.tokens.Token): The spaCy token to be inflected.

  Returns:
    str: The past participle form of the word.
  '''

  # Retrieve the past participle form by delegating to the generic inflection function.
  return GetInflectedForm(token, "VBN")


def GetPastTenseForm(token):
  r'''
  Retrieves the simple past tense form (VBD) of a given spaCy token.

  Parameters:
    token (spacy.tokens.Token): The spaCy token to be inflected.

  Returns:
    str: The simple past tense form of the word.
  '''

  # Retrieve the simple past tense form by delegating to the generic inflection function.
  return GetInflectedForm(token, "VBD")


def CalculateEnglishCorrectnessScore(match, currentText, maskPipeline):
  r'''
  Calculates a robust correctness score for a specific grammar match.

  The score is derived from LanguageTool metadata (category, error length, confidence)
  and contextual checks (plural/singular agreement, masked likelihood delta).

  Parameters:
    match (language_tool_python.Match): The grammar match object.
    currentText (str): The current state of the text being evaluated.
    maskPipeline (transformers.Pipeline): The masked language model pipeline.

  Returns:
    float: The calculated correctness score.
  '''

  # Initialize the base correctness score using LanguageTool metadata.
  score = 0.0
  # Extract the grammatical category from the match metadata.
  category = getattr(match, "category", "")
  # Assign high priority to strict grammatical errors.
  if (category == "GRAMMAR"):
    # Add maximum weight for structural grammar issues.
    score += 100.0
  # Assign moderate priority to typographical and spelling errors.
  elif (category == "TYPOS"):
    # Add moderate weight for spelling mistakes.
    score += 50.0
  # Assign lower priority to punctuation and stylistic suggestions.
  elif (category in ["PUNCTUATION", "TYPOGRAPHY", "STYLE"]):
    # Add minimal weight for subjective or formatting issues.
    score += 10.0
  # Extract the length of the detected error span.
  errorLength = getattr(match, "errorLength", getattr(match, "error_length", 0))
  # Increase score for longer error spans as they are usually more specific.
  score += errorLength * 2.0
  # Extract the list of suggested replacements.
  replacements = getattr(match, "replacements", [])
  # Increase score if the tool is highly confident (exactly one replacement).
  if (len(replacements) == 1):
    # Add confidence bonus for unambiguous corrections.
    score += 20.0
  # Decrease score if the tool is uncertain (many replacement options).
  elif (len(replacements) > 5):
    # Subtract confidence penalty for ambiguous stylistic suggestions.
    score -= 10.0
  # Verify that the proposed replacement doesn't create a plural/singular mismatch.
  if (replacements):
    # Get the primary suggested replacement.
    replacement = replacements[0].lower()
    # Define sets for plural pronouns and singular determiners.
    pluralPronouns = {"these", "those", "we", "they", "our", "their"}
    singularDeterminers = {" a ", " an ", " this ", " that ", " my ", " his ", " her ", " its "}
    # Check if the replacement is a plural pronoun.
    if (replacement in pluralPronouns):
      # Extract the text immediately following the error span.
      afterText = currentText[match.offset + errorLength:].lower()
      # If a singular determiner appears shortly after, penalize the score heavily.
      if (any(det in afterText for det in singularDeterminers)):
        # Apply a heavy penalty for number mismatch.
        score -= 50.0
  # Use the MLM to verify contextual naturalness if the pipeline is provided.
  if (maskPipeline is not None and replacements):
    # Extract the primary suggested replacement.
    replacement = replacements[0]
    # Extract the original text span from the current context.
    originalSpan = currentText[match.offset: match.offset + errorLength]
    # Attempt to calculate the MLM likelihood delta.
    try:
      # Calculate the likelihood delta to verify contextual naturalness.
      likelihoodDelta = CalculateMaskedLikelihoodDelta(
        maskPipeline,
        currentText,
        originalSpan,
        replacement,
        match.offset,
        errorLength
      )
      # Add the likelihood delta to the score (scaled to match the base weights).
      score += (likelihoodDelta * 500.0)
    # Handle any exceptions during the MLM calculation.
    except Exception:
      # Ignore MLM errors and rely on metadata scoring.
      pass
  # Return the final calculated correctness score.
  return score


def ApplyLanguageToolCorrections(inputText, languageToolInstance, maskPipeline):
  r'''
  Iteratively applies the highest-scoring grammar corrections to the input text.

  Parameters:
    inputText (str): The text to be corrected.
    languageToolInstance (language_tool_python.LanguageTool): The initialized tool.
    maskPipeline (transformers.Pipeline): The masked language model pipeline.

  Returns:
    str: The fully corrected input text.

  Examples
  --------
  .. code-block:: python

    import language_tool_python
    from transformers import pipeline
    from HMB.TextHelper import ApplyLanguageToolCorrections

    # Initialize the LanguageTool instance for English.
    languageToolInstance = language_tool_python.LanguageTool("en-US")
    # Initialize a masked language model pipeline (e.g., RoBERTa).
    maskPipeline = pipeline("fill-mask", model="roberta-base")

    # Define the input text with grammatical errors.
    inputText = "She go to the market yesterday and buy some fruits."

    # Apply the grammar corrections iteratively.
    correctedText = ApplyLanguageToolCorrections(inputText, languageToolInstance, maskPipeline)
    # Print the fully corrected text.
    print("Corrected text:", correctedText)
  '''

  # Retrieve the initial list of grammar matches from the LanguageTool instance.
  matches = languageToolInstance.check(inputText)
  # Continue processing as long as there are actionable matches.
  while (matches):
    # Filter out matches that do not have any suggested replacements.
    actionableMatches = [m for m in matches if (getattr(m, "replacements", []))]
    # Break the loop if no actionable matches remain to prevent infinite loops.
    if (not actionableMatches):
      # Exit the while loop.
      break
    # Sort the actionable matches by their English correctness score in descending order.
    actionableMatches.sort(key=lambda m: CalculateEnglishCorrectnessScore(m, inputText, maskPipeline), reverse=True)
    # Select the match with the highest correctness score.
    bestMatch = actionableMatches[0]
    # Extract the first suggested replacement from the best match.
    replacement = bestMatch.replacements[0]
    # Extract the error length safely across different library versions.
    errorLength = getattr(bestMatch, "errorLength", getattr(bestMatch, "error_length", 0))
    # Apply the correction by slicing the input text at the match offset.
    inputText = inputText[:bestMatch.offset] + replacement + inputText[bestMatch.offset + errorLength:]
    # Re-check the corrected text for any remaining grammar issues.
    matches = languageToolInstance.check(inputText)
  # Return the fully corrected input text.
  return inputText


def EnforceTemporalConsistency(inputText, nlpModel):
  r'''
  Dynamically enforces past tense for verbs when past time markers are detected.

  Parameters:
    inputText (str): The text to be processed.
    nlpModel (spacy.Language): The initialized spaCy model.

  Returns:
    str: The temporally corrected text.
  '''

  import re
  from datetime import datetime

  # Parse the text with spaCy to access entities and morphological inflection.
  doc = nlpModel(inputText)
  # Initialize the flag to track if a past time marker is detected.
  hasPastMarker = False
  # Get the current year to evaluate specific year entities.
  currentYear = datetime.now().year
  # Define a set of absolute past adverbs for quick lookup.
  pastAdverbs = {"yesterday", "previously", "formerly", "earlier", "then", "once", "back"}
  # Iterate through all named entities detected by the spaCy pipeline.
  for ent in doc.ents:
    # Check if the entity is classified as a DATE or TIME by spaCy.
    if (ent.label_ in ["DATE", "TIME"]):
      # Extract the lowercase text of the date entity.
      dateText = ent.text.lower().strip()
      # Check 1: Does the entity contain the word "ago"?
      if (re.search(r"\bago\b", dateText)):
        # Mark that a past time marker has been found.
        hasPastMarker = True
        # Break the loop as we have confirmed the past tense requirement.
        break
      # Check 2: Does the entity start with the word "last"?
      if (re.search(r"\blast\b", dateText)):
        # Mark that a past time marker has been found.
        hasPastMarker = True
        # Break the loop to proceed with verb conjugation.
        break
      # Check 3: Is the entity an explicit past adverb or phrase?
      if (dateText in pastAdverbs or any(word in dateText.split() for word in pastAdverbs)):
        # Mark that a past time marker has been found.
        hasPastMarker = True
        # Break the loop to proceed with verb conjugation.
        break
      # Check 4: Does the entity contain a specific year in the past?
      yearMatch = re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", dateText)
      # Check if a valid year pattern was successfully matched.
      if (yearMatch):
        # Extract the integer value of the matched year.
        extractedYear = int(yearMatch.group(1))
        # Check if the extracted year is strictly less than the current year.
        if (extractedYear < currentYear):
          # Mark that a past time marker has been found.
          hasPastMarker = True
          # Break the loop to proceed with verb conjugation.
          break
      # Check 5: Does the entity refer to historical or past eras?
      if (re.search(r"\b(?:past|previous|former|ancient|medieval)\b", dateText)):
        # Mark that a past time marker has been found.
        hasPastMarker = True
        # Break the loop to proceed with verb conjugation.
        break
  # Return the original text immediately if no past markers are found.
  if (not hasPastMarker):
    # Return the unmodified text to save compute.
    return inputText
  # Initialize a list to hold the reconstructed tokens.
  reconstructedTokens = []
  # Iterate through each token in the parsed document.
  for token in doc:
    # Check if the token is a main verb in present tense or base form.
    if (token.pos_ == "VERB" and token.tag_ in ["VBP", "VBZ", "VB"]):
      # Retrieve the dynamically conjugated past tense verb.
      pastVerb = GetPastTenseForm(token)
      # Append the dynamically conjugated past tense verb with its original whitespace.
      reconstructedTokens.append(pastVerb + token.whitespace_)
    else:
      # Append non-verb tokens exactly as they appeared.
      reconstructedTokens.append(token.text + token.whitespace_)
  # Join the tokens back into a single string and strip trailing whitespace.
  return "".join(reconstructedTokens).strip()


def CorrectGrammar(text, maskPipeline=None):
  r'''
  Applies comprehensive grammar correction using LanguageTool with advanced
  contextual correctness scoring and dynamic temporal consistency enforcement.

  Parameters:
    text (str): The input text to be grammar-checked.
    maskPipeline (transformers.Pipeline, optional): A fill-mask pipeline for MLM scoring.

  Returns:
    str: The grammar-corrected text.

  Examples
  --------
  .. code-block:: python

    from transformers import pipeline
    from HMB.TextHelper import CorrectGrammar

    # Initialize a masked language model pipeline (e.g., RoBERTa).
    maskPipeline = pipeline("fill-mask", model="roberta-base")

    # Define the text to be corrected.
    text = "He go to the store yesterday and buyed some milk."

    # Execute the correct grammar function to fix the text.
    correction = CorrectGrammar(text, maskPipeline)
    print(f"Corrected text: {correction}")
  '''

  # Import the language tool python library for comprehensive grammar checking.
  import language_tool_python
  # Import the spaCy library for natural language processing.
  import spacy

  # Validate that the input text is a non-empty string.
  if (not text or not isinstance(text, str)):
    # Return the original input immediately if it is invalid.
    return text
  # Initialize the LanguageTool instance for US English.
  languageToolInstance = language_tool_python.LanguageTool("en-US")
  # Use a global cache to avoid reloading the heavy spaCy model on every call.
  global _nlpModelCache
  # Check if the NLP model cache is not yet initialized.
  if ("_nlpModelCache" not in globals()):
    # Load the English spaCy model and cache it globally.
    _nlpModelCache = spacy.load("en_core_web_sm")
  # Assign the cached model to a local variable for processing.
  nlpModel = _nlpModelCache
  # Initialize the corrected text variable with the original input.
  correctedText = text
  # Execute the iterative grammar tool on the current text.
  toolOutput = ApplyLanguageToolCorrections(correctedText, languageToolInstance, maskPipeline)
  # Check if the tool returned a valid, non-empty string.
  if (toolOutput and isinstance(toolOutput, str) and len(toolOutput.strip()) > 0):
    # Update the corrected text with the tool's output.
    correctedText = toolOutput
  # Apply the dynamic temporal consistency post-processor to fix long-distance tense errors.
  correctedText = EnforceTemporalConsistency(correctedText, nlpModel)
  # Check if the final corrected text is different from the original input.
  if (correctedText != text):
    # Print a success message indicating that corrections were applied.
    print("✅ External grammar check applied successfully.")
  # Return the final corrected text.
  return correctedText


if __name__ == "__main__":
  # Lightweight demo and tests for TextHelper and CleanText.
  sampleText = '''
  This is an example text! It includes various elements:
  - Contractions like don't and it's.
  - Special characters: @#$%^&*()!
  - Multiple     spaces and newlines.

  Let's see how well the cleaning function works.
  '''

  # Call the CleanText function with all options enabled except common word removal and non-English word removal.
  cleaned = CleanText(
    sampleText,  # The raw text to clean.
    removeNonAscii=True,  # Remove non-ASCII characters.
    lowercase=True,  # Convert text to lowercase.
    removeSpecialChars=True,  # Remove special characters and punctuation.
    normalizeWhitespace=True,  # Replace multiple spaces with a single space.
    handleContractions=True,  # Expand contractions (e.g., don't → do not).
    lemmatize=True,  # Lemmatize words to their base form.
    removeStopwords=True,  # Remove common stop words.
    removeCommonWords=False,  # Do not remove most common words.
    numOfCommonWords=10,  # Number of common words to remove if enabled.
    removeNonEnglishWords=False,  # Do not remove non-English words.
  )

  # Print the original text before cleaning.
  print("Original Text:\n", sampleText)
  # Print the cleaned text after processing.
  print("\nCleaned Text:\n", cleaned)


  # SafeCall helper used to call methods and gracefully report failures.
  def SafeCall(name, fn, *args, **kwargs):
    try:
      res = fn(*args, **kwargs)
      print(f"{name} ->", res)
      print("-" * 40)
      return res
    except Exception as e:
      print(f"{name} raised {type(e).__name__}:", e)
      print("-" * 40)
      return None


  # Summarizer demo (guarded because it requires transformers/torch and model weights).
  sampleLongText = (
    "This is an example text! "
    "The quick brown fox jumps over the lazy dog. "
    "This sentence contains every letter of the English alphabet. "
    "It's often used to test fonts and keyboard layouts. "
    "In addition to its practical uses, it has a playful tone that makes it memorable. "
    "The fox is known for its cunning and agility, while the dog represents loyalty and patience. "
    "Together, they create a vivid image that captures the imagination."
  )
  try:
    summarizer = HuggingFaceTextSummarizer(
      modelName="facebook/bart-large-cnn",
      maxLength=150,
      minLength=25,
      maxInputLength=1024,
    )
    try:
      summary = summarizer.Summarize(sampleLongText)
      print("Length of Original Text:", len(sampleLongText))
      print("Length of Summary:", len(summary))
      print("\nSummary:\n", summary)
    except Exception as e:
      print("Summarizer Summarize failed:", type(e).__name__, e)
  except Exception as e:
    print("Summarizer init skipped:", type(e).__name__, e)

  th = TextHelper()

  # Simple tests for tokenizer and basic helpers.
  sampleSentence = "Hello world. This is a test."
  SafeCall("SentenceTokeize", th.SentenceTokeize, sampleSentence)
  SafeCall("WordTokenize", th.WordTokenize, sampleSentence)
  SafeCall("Tokenization", th.Tokenization, sampleSentence)

  # Stopwords removal demonstration on token list.
  tokens = th.SentenceTokeize("This is a simple sentence for testing stopwords removal.")
  SafeCall("RemoveStopWords", th.RemoveStopWords, tokens)

  # N-grams demo (bigrams).
  # SafeCall("Ngrams (bigrams)", th.Ngrams, ["this", "is", "a", "test"])

  # Stemming and Lemmatization.
  SafeCall("Stemming", th.Stemming, ["running", "jumps", "easily"])
  SafeCall("Lemmatization", th.Lemmatization, ["running", "jumps", "easily"])

  # POS tagging and Chunking (try to get POS tags first, fall back to token list).
  try:
    posTokens = th.POS(word_tokenize("Apple is looking at buying U.K. startup for $1 billion"))
  except Exception:
    posTokens = None
  SafeCall("POS", th.POS, ["This", "is", "a", "test"])
  SafeCall("Chunking", th.Chunking, posTokens if posTokens else ["Apple", "is", "buying", "startup"])

  # Vectorizers and embedding demos (use small safe inputs).
  SafeCall("BagOfWords", th.BagOfWords, ["this is a doc", "this is another doc"])
  SafeCall("TFIDF", th.TFIDF, ["this is a doc", "this is another doc"])
  SafeCall("Word2Vec", th.Word2Vec, [["this", "is"], ["another", "doc"]])
  SafeCall("Doc2Vec", th.Doc2Vec, ["This is doc one", "This is doc two"], False)

  # Topic models (small inputs) - these may be slow or require gensim.
  SafeCall("LDA", th.LDA, [["apple", "banana"], ["banana", "carrot"]])
  SafeCall("LSA", th.LSA, [["apple", "banana"], ["banana", "carrot"]])

  # Summarization / TextRank / TopicModeling (may require gensim).
  SafeCall("TextSummarization", th.TextSummarization,
           "This is a short text. It has multiple sentences. It might be too short for gensim.summarize.")
  SafeCall("TextRank", th.TextRank,
           "This is a short text. It has multiple sentences. It might be too short for gensim.summarize.")
  SafeCall("TopicModeling", th.TopicModeling, sampleLongText)

  # Similarity and distances - provide valid strings instead of None.
  SafeCall("CosineSimilarity", th.CosineSimilarity, "hello world", "hello")
  SafeCall("JaccardSimilarity", th.JaccardSimilarity, "apple banana", "banana carrot")
  SafeCall("EuclideanDistance", th.EuclideanDistance, "hello world", "hello")
  SafeCall("ManhattanDistance", th.ManhattanDistance, "hello world", "hello")

  # WordCloud demo (may require pillow and wordcloud).
  try:
    SafeCall("WordCloud", th.WordCloud, "hello world hello")
  except Exception as e:
    print("WordCloud demo skipped:", type(e).__name__, e)

  # Text to speech (guarded because it creates a file and requires gTTS).
  try:
    SafeCall("TextToSpeech", th.TextToSpeech, "Hello world from HMB TextHelper", "hmb_test_tts.mp3")
  except Exception as e:
    print("TextToSpeech demo skipped:", type(e).__name__, e)

  # TextBlob / langdetect demos.
  SafeCall("SentimentAnalysis", th.SentimentAnalysis, "I love this product. It's great!")
  SafeCall("LanguageDetection", th.LanguageDetection, "Bonjour tout le monde")
  SafeCall("SpellingCorrection", th.SpellingCorrection, "I havv goood speling")
  SafeCall("LanguageTranslation", th.LanguageTranslation, "Hello world", to="es")

  # NER (spaCy - may require model).
  SafeCall("NER", th.NER, "Apple is looking at buying U.K. startup for $1 billion")

  print("TextHelper tests completed.")
