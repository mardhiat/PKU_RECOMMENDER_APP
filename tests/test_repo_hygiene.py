import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FORBIDDEN_NAMES = {
    "ratingsappdata - Sheet1.csv",
    "user_responses.csv",
    "user_nutritional_limits.csv",
    "data_user_food_ratings.csv",
    "data_user_nutritional_limits.csv",
    "data_train_ratings.csv",
    "data_test_ratings.csv",
    "data_test_users.csv",
    "usda_cache.json",
}


class RepoHygieneTests(unittest.TestCase):
    def test_participant_exports_are_gone(self):
        present = [path.name for path in ROOT.rglob("*") if path.name in FORBIDDEN_NAMES and ".git" not in path.parts]
        self.assertEqual(present, [])

    def test_data_directory_has_no_email_addresses(self):
        import re

        pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
        hits = []
        for path in DATA.rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                hits.append(str(path.relative_to(ROOT)))
        self.assertEqual(hits, [])

    def test_synthetic_fixture_uses_demo_ids(self):
        fixture = (DATA / "fixtures" / "synthetic_ratings.csv").read_text(encoding="utf-8")
        self.assertIn("user_id", fixture.splitlines()[0])
        self.assertNotIn("email", fixture.splitlines()[0].lower())
        self.assertIn("demo-01", fixture)


if __name__ == "__main__":
    unittest.main()
