import unittest

import pandas as pd

from pku.ratings import rating_for, recommendation_universe


class RatingLookupTests(unittest.TestCase):
    def test_lookup_does_not_reuse_another_users_row_index(self):
        # User B's rating for rice sits on index 0. Filtering to rice first
        # and then using the original index would miss user A or crash.
        frame = pd.DataFrame(
            [
                {"user_id": "demo-b", "food": "Rice", "rating": 1},
                {"user_id": "demo-a", "food": "Beans", "rating": 5},
                {"user_id": "demo-a", "food": "Rice", "rating": 4},
            ]
        )
        self.assertEqual(rating_for(frame, "demo-a", "rice"), 4)
        self.assertIsNone(rating_for(frame, "demo-a", "yam"))

    def test_recommendation_universe_is_the_rated_set(self):
        self.assertEqual(recommendation_universe(["Rice", "Rice", "Beans"]), {"Rice", "Beans"})


if __name__ == "__main__":
    unittest.main()
