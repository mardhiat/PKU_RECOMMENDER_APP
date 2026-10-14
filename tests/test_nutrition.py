import unittest
from datetime import date

from pku.nutrition import (
    LIKED_RATING_MIN,
    PHE_MG_PER_G_PROTEIN,
    calculate_age_months,
    get_child_adult_daily_needs,
    impute_phenylalanine,
    is_liked,
)


class NutritionTests(unittest.TestCase):
    def test_age_months_waits_for_the_birthday(self):
        born = date(2020, 5, 20)
        before_birthday = date(2024, 5, 19)
        on_birthday = date(2024, 5, 20)
        self.assertEqual(calculate_age_months(born.year, born.month, born.day, today=before_birthday), 47)
        self.assertEqual(calculate_age_months(born.year, born.month, born.day, today=on_birthday), 48)

    def test_impute_only_when_protein_exists_and_phe_is_missing(self):
        phe, imputed = impute_phenylalanine(0, 2)
        self.assertTrue(imputed)
        self.assertEqual(phe, 2 * PHE_MG_PER_G_PROTEIN)

        phe, imputed = impute_phenylalanine(0, 0)
        self.assertFalse(imputed)
        self.assertEqual(phe, 0)

        phe, imputed = impute_phenylalanine(12, 2)
        self.assertFalse(imputed)
        self.assertEqual(phe, 12)

    def test_liked_cutoff_is_four(self):
        self.assertEqual(LIKED_RATING_MIN, 4)
        self.assertFalse(is_liked(3))
        self.assertTrue(is_liked(4))

    def test_adult_age_group_matches_the_age_band(self):
        needs = get_child_adult_daily_needs(30 * 12, 70, "Female")
        self.assertEqual(needs["age_group"], "19+ years")
        self.assertLess(needs["phe_mg_min"], needs["phe_mg_max"])


if __name__ == "__main__":
    unittest.main()
