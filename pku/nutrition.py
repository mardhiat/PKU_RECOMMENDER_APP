"""Age and phenylalanine helpers shared by the meal planner and the evaluation pipeline.

Phenylalanine imputation: when a food has protein but no measured phenylalanine,
the pipeline estimates PHE as 50 mg per gram of protein. That is an assumption,
not a lab value. Callers must keep the imputed flag.
"""

from datetime import date

# A food counts as liked at this rating and above. The same cutoff is used in
# the train/test filter and in the preference metrics.
LIKED_RATING_MIN = 4

# Milligrams of phenylalanine assumed per gram of protein when PHE is missing.
PHE_MG_PER_G_PROTEIN = 50.0


def calculate_age_months(year, month, day, today=None):
    """Whole months of age. The day matters: a birthday later this month is not counted yet."""
    today = today or date.today()
    born = date(int(year), int(month), int(day))
    months = (today.year - born.year) * 12 + (today.month - born.month)
    if today.day < born.day:
        months -= 1
    return max(0, months)


def is_liked(rating):
    return float(rating) >= LIKED_RATING_MIN


def impute_phenylalanine(phe_mg, protein_g):
    """Return (phe_mg, imputed).

    Impute only when phenylalanine is missing and protein is present.
    A zero for both stays zero and is not treated as a safe food.
    """
    phe = float(phe_mg or 0.0)
    protein = float(protein_g or 0.0)
    if phe == 0.0 and protein > 0.0:
        return protein * PHE_MG_PER_G_PROTEIN, True
    return phe, False


def get_infant_daily_needs(age_months, weight_kg):
    """Daily targets for infants 0-12 months. Values are planning defaults, not a prescription."""
    if age_months < 3:
        band = {
            "protein_g_per_kg": 3.5,
            "phe_mg_per_kg_min": 25,
            "phe_mg_per_kg_max": 70,
            "energy_kcal_per_kg": 120,
            "fluid_ml_per_kg": 160,
            "age_group": "0-3 months",
        }
    elif age_months < 6:
        band = {
            "protein_g_per_kg": 3.5,
            "phe_mg_per_kg_min": 20,
            "phe_mg_per_kg_max": 45,
            "energy_kcal_per_kg": 120,
            "fluid_ml_per_kg": 160,
            "age_group": "3-6 months",
        }
    elif age_months < 9:
        band = {
            "protein_g_per_kg": 3.0,
            "phe_mg_per_kg_min": 15,
            "phe_mg_per_kg_max": 35,
            "energy_kcal_per_kg": 110,
            "fluid_ml_per_kg": 145,
            "age_group": "6-9 months",
        }
    else:
        band = {
            "protein_g_per_kg": 3.0,
            "phe_mg_per_kg_min": 10,
            "phe_mg_per_kg_max": 35,
            "energy_kcal_per_kg": 105,
            "fluid_ml_per_kg": 135,
            "age_group": "9-12 months",
        }

    needs = dict(band)
    needs["protein_g"] = band["protein_g_per_kg"] * weight_kg
    needs["phe_mg_min"] = band["phe_mg_per_kg_min"] * weight_kg
    needs["phe_mg_max"] = band["phe_mg_per_kg_max"] * weight_kg
    needs["phe_mg_target"] = (needs["phe_mg_min"] + needs["phe_mg_max"]) / 2
    needs["energy_kcal"] = band["energy_kcal_per_kg"] * weight_kg
    needs["fluid_ml"] = band["fluid_ml_per_kg"] * weight_kg
    return needs


def get_child_adult_daily_needs(age_months, weight_kg, sex):
    """Daily targets for ages 1 year and older. Planning defaults, not a prescription."""
    female = sex == "Female"
    if age_months < 48:
        needs = {"phe_mg_min": 200, "phe_mg_max": 400, "protein_g": 30, "energy_kcal": 1300, "age_group": "1-4 years"}
    elif age_months < 84:
        needs = {"phe_mg_min": 210, "phe_mg_max": 450, "protein_g": 35, "energy_kcal": 1700, "age_group": "4-7 years"}
    elif age_months < 132:
        needs = {"phe_mg_min": 220, "phe_mg_max": 500, "protein_g": 40, "energy_kcal": 2400, "age_group": "7-11 years"}
    elif age_months < 180:
        if female:
            needs = {"phe_mg_min": 250, "phe_mg_max": 750, "protein_g": 50, "energy_kcal": 2200, "age_group": "11-15 years"}
        else:
            needs = {"phe_mg_min": 225, "phe_mg_max": 900, "protein_g": 55, "energy_kcal": 2700, "age_group": "11-15 years"}
    elif age_months < 228:
        if female:
            needs = {"phe_mg_min": 230, "phe_mg_max": 700, "protein_g": 55, "energy_kcal": 2100, "age_group": "15-19 years"}
        else:
            needs = {"phe_mg_min": 295, "phe_mg_max": 1100, "protein_g": 65, "energy_kcal": 2800, "age_group": "15-19 years"}
    else:
        if female:
            needs = {"phe_mg_min": 220, "phe_mg_max": 700, "protein_g": 60, "energy_kcal": 2100, "age_group": "19+ years"}
        else:
            needs = {"phe_mg_min": 290, "phe_mg_max": 1200, "protein_g": 70, "energy_kcal": 2900, "age_group": "19+ years"}

    needs["phe_mg_target"] = (needs["phe_mg_min"] + needs["phe_mg_max"]) / 2
    needs["weight_kg"] = weight_kg
    return needs
