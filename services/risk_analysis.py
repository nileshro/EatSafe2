def analyze_risks(nutrition, profile):

    risks = []
    suitable_for = []
    not_suitable_for = []

    sugar = nutrition.get("sugar", 0)
    sodium = nutrition.get("sodium", 0)
    sat_fat = nutrition.get("saturated_fat", 0)
    energy = nutrition.get("energy", 0)
    fiber = nutrition.get("fiber", 0)

    # Risk Factors

    if sugar > 22.5:
        risks.append("Very High Sugar")

    if sodium > 600:
        risks.append("Very High Sodium")

    if sat_fat > 10:
        risks.append("Very High Saturated Fat")

    if energy > 500:
        risks.append("High Energy Density")

    # Generic suitability

    if sugar <= 5:
        suitable_for.append("Diabetes Friendly")

    if sodium <= 120:
        suitable_for.append("Hypertension Friendly")

    if energy <= 300:
        suitable_for.append("Weight Management")

    if fiber >= 6:
        suitable_for.append("High Fiber Diet")

    # Personalized checks

    if profile["diabetes"] and sugar > 5:
        not_suitable_for.append("Diabetes")

    if profile["hypertension"] and sodium > 120:
        not_suitable_for.append("Hypertension")

    if profile["obesity"] and energy > 300:
        not_suitable_for.append("Weight Management")

    return {
        "risks": risks,
        "suitable_for": suitable_for,
        "not_suitable_for": not_suitable_for
    }