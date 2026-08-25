import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_provider import (  # noqa: E402
    DEFAULT_OPENAI_MODEL,
    create_llm_client,
    model_or_default,
)


class OpenAIProviderTests(unittest.TestCase):
    def test_default_model(self):
        self.assertEqual(model_or_default(None), DEFAULT_OPENAI_MODEL)
        self.assertEqual(model_or_default("requested-model"), "requested-model")

    def test_api_key_is_required(self):
        with self.assertRaisesRegex(ValueError, "API-Key"):
            create_llm_client(api_key=None)

    @patch("llm_provider.OpenAI")
    def test_client_receives_api_key(self, openai):
        client = create_llm_client(api_key="test-key")
        self.assertIs(client, openai.return_value)
        openai.assert_called_once_with(api_key="test-key")


if __name__ == "__main__":
    unittest.main()
