from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dotenv_loader import read_env


class ExplicitEncodingRegressionTests(unittest.TestCase):
    def test_explicit_non_utf8_encoding_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_bytes("CITY=café\n".encode("latin-1"))
            self.assertEqual(
                read_env(dotenv, encoding="latin-1"),
                "CITY=café\n",
            )


if __name__ == "__main__":
    unittest.main()
