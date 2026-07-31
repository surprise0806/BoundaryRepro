from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv_loader import read_env


class DefaultEncodingTests(unittest.TestCase):
    def test_default_decodes_utf8_when_platform_locale_is_ascii(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_bytes("ACCOUNT_NAME=上海\n".encode("utf-8"))
            with patch(
                "dotenv_loader.locale.getpreferredencoding",
                return_value="ascii",
            ):
                self.assertEqual(read_env(dotenv), "ACCOUNT_NAME=上海\n")


if __name__ == "__main__":
    unittest.main()
