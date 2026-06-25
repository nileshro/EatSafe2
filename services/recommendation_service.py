def recommend_alternatives(product_name, nutrition):

    product = (product_name or "").lower()



    alternatives = []

    if "chocolate" in product:
        alternatives = [
            "Dark Chocolate (70% Cocoa)",
            "Protein Bar",
            "Unsweetened Cocoa Snacks"
        ]

    elif "chips" in product:
        alternatives = [
            "Roasted Makhana",
            "Air-Popped Popcorn",
            "Roasted Chana"
        ]

    elif "soft drink" in product:
        alternatives = [
            "Coconut Water",
            "Lemon Water",
            "Buttermilk"
        ]

    elif "noodle" in product:
        alternatives = [
            "Oats",
            "Poha",
            "Upma"
        ]

    elif nutrition.get("sugar", 0) > 20:

        alternatives = [
            "Fresh Fruits",
            "Greek Yogurt",
            "Dry Fruits"
        ]

    return alternatives