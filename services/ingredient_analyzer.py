def analyze_ingredients(ingredients):

    warnings = []

    ingredient_text = " ".join(ingredients).lower()

    risk_db = {

        "palm oil":
        "Contains Palm Oil (highly processed fat)",

        "msg":
        "Contains MSG (flavour enhancer)",

        "monosodium glutamate":
        "Contains MSG (flavour enhancer)",

        "artificial flavor":
        "Contains Artificial Flavours",

        "artificial flavour":
        "Contains Artificial Flavours",

        "artificial color":
        "Contains Artificial Colours",

        "artificial colour":
        "Contains Artificial Colours",

        "sodium benzoate":
        "Contains Preservative Sodium Benzoate",

        "potassium sorbate":
        "Contains Preservative Potassium Sorbate",

        "high fructose corn syrup":
        "Contains High Fructose Corn Syrup",

        "hfcs":
        "Contains High Fructose Corn Syrup"
    }

    for key, value in risk_db.items():

        if key in ingredient_text:
            warnings.append(value)

    return warnings