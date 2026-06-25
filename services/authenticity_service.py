def calculate_authenticity(
    gemini_brand,
    cnn_brand,
    cnn_confidence
):

    gemini_brand = (
        gemini_brand or ""
    ).lower().strip()

    cnn_brand = (
        cnn_brand or ""
    ).lower().strip()

    if not gemini_brand:
        return 0

    if gemini_brand == cnn_brand:

        return round(
            cnn_confidence * 100,
            2
        )

    return round(
        cnn_confidence * 25,
        2
    )