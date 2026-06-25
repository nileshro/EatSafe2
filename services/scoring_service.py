def calculate_health_score(nutrition, profile):

    score = 10

    sugar = nutrition.get("sugar", 0)
    sodium = nutrition.get("sodium", 0)
    sat_fat = nutrition.get("saturated_fat", 0)
    energy = nutrition.get("energy", 0)
    protein = nutrition.get("protein", 0)
    fiber = nutrition.get("fiber", 0)

    # Sugar
    if sugar > 22.5:
        score -= 3
    elif sugar > 15:
        score -= 2
    elif sugar > 5:
        score -= 1

    # Sodium
    if sodium > 600:
        score -= 3
    elif sodium > 400:
        score -= 2
    elif sodium > 120:
        score -= 1

    # Saturated Fat
    if sat_fat > 10:
        score -= 3
    elif sat_fat > 5:
        score -= 2
    elif sat_fat > 1.5:
        score -= 1

    # Energy
    if energy > 500:
        score -= 3
    elif energy > 300:
        score -= 2
    elif energy > 150:
        score -= 1

    # Protein Bonus
    if protein > 20:
        score += 2
    elif protein > 10:
        score += 1

    # Fiber Bonus
    if fiber > 6:
        score += 2
    elif fiber > 3:
        score += 1

    # Personalization

    if profile["diabetes"]:
        if sugar > 15:
            score -= 2
        elif sugar > 5:
            score -= 1

    if profile["hypertension"]:
        if sodium > 400:
            score -= 2
        elif sodium > 120:
            score -= 1

    if profile["obesity"]:
        if energy > 300:
            score -= 2

    return max(1, min(10, score))