import unittest
import torch
from unittest.mock import patch, MagicMock

# Import after setting up patches in tests
from HMB.EmbeddingsToTextHelper import EmbeddingsToTextModel


class TestEmbeddingsToTextHelper(unittest.TestCase):
  '''
  Unit tests for EmbeddingsToTextHelper with heavy dependencies mocked.
  Tests model initialization, forward, and generate flows with synthetic tensors.
  '''

  @patch("HMB.EmbeddingsToTextHelper.T5ForConditionalGeneration")
  @patch("HMB.EmbeddingsToTextHelper.T5Tokenizer")
  def test_model_init(self, mockTokenizerCls, mockT5Cls):
    # Mock tokenizer behavior
    mockTokenizer = MagicMock()
    mockTokenizer.pad_token = None
    mockTokenizer.eos_token = "<eos>"
    mockTokenizer.pad_token_id = None
    mockTokenizer.eos_token_id = 1
    mockTokenizerCls.from_pretrained.return_value = mockTokenizer

    # Mock T5 model
    mockT5 = MagicMock()
    mockT5.config = MagicMock(d_model=16)
    mockT5Cls.from_pretrained.return_value = mockT5

    # Instantiate model with small dims
    model = EmbeddingsToTextModel(
      tokenizeModelName="t5-small",
      inputFeatureDim=32,
      hiddenDim=8,
      generationMaxLength=16,
      dropoutRatio=0.0,
      numPromptTokens=3,
    )

    # Verify tokenizer pad token set
    self.assertEqual(model.tokenizer.pad_token, "<eos>")
    self.assertEqual(model.tokenizer.pad_token_id, 1)
    self.assertEqual(model.t5.config.d_model, 16)

  @patch("HMB.EmbeddingsToTextHelper.T5ForConditionalGeneration")
  @patch("HMB.EmbeddingsToTextHelper.T5Tokenizer")
  def test_forward(self, mockTokenizerCls, mockT5Cls):
    # Mock tokenizer
    mockTokenizer = MagicMock()
    mockTokenizer.pad_token = "<pad>"
    mockTokenizer.pad_token_id = 0
    mockTokenizerCls.from_pretrained.return_value = mockTokenizer

    # Mock T5 output
    mockOutput = MagicMock()
    mockT5 = MagicMock()
    mockT5.config = MagicMock(d_model=16)
    mockT5.return_value = mockOutput
    mockT5Cls.from_pretrained.return_value = mockT5

    model = EmbeddingsToTextModel(
      inputFeatureDim=32,
      hiddenDim=8,
      numPromptTokens=4,
      generationMaxLength=16,
    )

    # Run forward with synthetic features
    features = torch.randn(2, 32)
    out = model.forward(features)
    self.assertIsNotNone(out)
    # Ensure T5 was called
    self.assertTrue(mockT5.called)

  @patch("HMB.EmbeddingsToTextHelper.T5ForConditionalGeneration")
  @patch("HMB.EmbeddingsToTextHelper.T5Tokenizer")
  def test_generate(self, mockTokenizerCls, mockT5Cls):
    # Mock tokenizer
    mockTokenizer = MagicMock()
    mockTokenizer.pad_token = "<pad>"
    mockTokenizer.pad_token_id = 0
    mockTokenizer.batch_decode.return_value = ["hello world"]
    mockTokenizerCls.from_pretrained.return_value = mockTokenizer

    # Mock T5 generate
    mockT5 = MagicMock()
    mockT5.config = MagicMock(d_model=16)
    mockT5.generate.return_value = torch.ones((2, 5), dtype=torch.long)
    mockT5Cls.from_pretrained.return_value = mockT5

    model = EmbeddingsToTextModel(
      inputFeatureDim=32,
      hiddenDim=8,
      numPromptTokens=4,
      generationMaxLength=16,
    )

    features = torch.randn(2, 32)
    ids = model.generate(features)  # Do not pass max_length to avoid duplicate arg
    self.assertIsInstance(ids, torch.Tensor)
    self.assertEqual(ids.shape[1], 5)

  @patch("HMB.EmbeddingsToTextHelper.T5ForConditionalGeneration")
  @patch("HMB.EmbeddingsToTextHelper.T5Tokenizer")
  def test_forward_invalid_shape_raises(self, mockTokenizerCls, mockT5Cls):
    mockTokenizer = MagicMock()
    mockTokenizer.pad_token_id = 0
    mockTokenizerCls.from_pretrained.return_value = mockTokenizer
    mockT5 = MagicMock()
    mockT5.config = MagicMock(d_model=16)
    mockT5Cls.from_pretrained.return_value = mockT5

    model = EmbeddingsToTextModel(inputFeatureDim=32, hiddenDim=8, numPromptTokens=2)
    with self.assertRaises(Exception):
      _ = model.forward(torch.randn(32))

  @patch("HMB.EmbeddingsToTextHelper.T5ForConditionalGeneration")
  @patch("HMB.EmbeddingsToTextHelper.T5Tokenizer")
  def test_generate_empty_batch(self, mockTokenizerCls, mockT5Cls):
    mockTokenizer = MagicMock()
    mockTokenizer.pad_token_id = 0
    mockTokenizer.batch_decode.return_value = []
    mockTokenizerCls.from_pretrained.return_value = mockTokenizer

    mockT5 = MagicMock()
    mockT5.config = MagicMock(d_model=16)
    mockT5.generate.return_value = torch.empty((0, 0), dtype=torch.long)
    mockT5Cls.from_pretrained.return_value = mockT5

    model = EmbeddingsToTextModel(inputFeatureDim=32, hiddenDim=8, numPromptTokens=2)
    features = torch.empty((0, 32))
    ids = model.generate(features)
    self.assertEqual(tuple(ids.shape), (0, 0))


if __name__ == "__main__":
  unittest.main()
