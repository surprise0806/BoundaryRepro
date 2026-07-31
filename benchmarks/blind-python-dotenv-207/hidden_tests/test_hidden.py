from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv_loader import read_env


class HiddenEncodingTests(unittest.TestCase):
    def test_default_handles_non_ascii_utf8_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_bytes("GREETING=你好🙂\n".encode("utf-8"))
            with patch(
                "dotenv_loader.locale.getpreferredencoding",
                return_value="cp1252",
            ):
                self.assertEqual(read_env(dotenv), "GREETING=你好🙂\n")

    def test_explicit_utf16_still_wins_over_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_bytes("VALUE=snowman-☃\n".encode("utf-16"))
            self.assertEqual(
                read_env(dotenv, encoding="utf-16"),
                "VALUE=snowman-☃\n",
            )

    def test_invalid_utf8_is_not_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_bytes(b"VALUE=\xff\n")
            with self.assertRaises(UnicodeDecodeError):
                read_env(dotenv)


if __name__ == "__main__":
    unittest.main()
