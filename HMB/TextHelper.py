# Import the required libraries.
import os, re, contractions, nltk
from nltk.tokenize import sent_tokenize, word_tokenize


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
    # Remove non-alphanumeric characters.
    cleanedText = re.sub(r"[^a-zA-Z0-9\s]", "", cleanedText)

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


class Summarizer(object):
  r'''
  Summarizer: Flexible text summarization using Hugging Face Transformers.

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

  Example Usage:
  --------------
  .. code-block:: python

      from HMB.TextHelper import Summarizer

      text = "Your long text to summarize goes here..."
      summarizer = Summarizer(
          modelName="facebook/bart-large-cnn",
          maxLength=130,
          minLength=30,
          maxInputLength=1024,
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

        summarizer = Summarizer()
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


if __name__ == "__main__":
  # Lightweight demo and tests for TextHelper and CleanText.
  sampleText = """
  This is an example text! It includes various elements:
  - Contractions like don't and it's.
  - Special characters: @#$%^&*()!
  - Multiple     spaces and newlines.

  Let's see how well the cleaning function works.
  """

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
    summarizer = Summarizer(
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
