import os
import time
import json

from dotenv import load_dotenv

load_dotenv()

# ── Use new google.genai SDK (replaces deprecated google.generativeai) ─────────
try:
    from google import genai
    from google.genai import types
    _USE_NEW_SDK = True
except ImportError:
    # Fallback to old SDK if new one not installed
    import google.generativeai as genai_old
    _USE_NEW_SDK = False


def _get_model():
    """Initialise Gemini client — new SDK preferred."""
    if _USE_NEW_SDK:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return client
    else:
        genai_old.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return genai_old.GenerativeModel("gemini-2.5-flash")


_PROMPT = """
You are an expert food nutrition analyst.

I am giving you photos of a food product package.
These photos show the front of the pack, the nutrition label, and/or the ingredients list.

TASK: Extract ALL nutrition information and return structured JSON.

CRITICAL RULES:
1. Read the nutrition table carefully from whichever image shows it best.
2. Check whether values are given per 100g, per 100ml, per serving, or per pack.
3. ALWAYS convert and return values standardised to per 100g.
   - If per serving: multiply by (100 / serving_size_g)
   - If per pack: divide total by pack weight then multiply by 100
   - If per 100ml (liquid): treat as per 100g directly
4. Set confidence = 0.95 when label is clearly readable.
5. Set confidence = 0.80 when label is partially visible but readable.
6. Only set confidence < 0.5 when nutrition label is COMPLETELY absent.
7. Extract all visible ingredient names into a list.
8. Return ONLY raw JSON — no markdown, no code fences, no explanation.

JSON structure to return:
{
  "product_name": "full product name here",
  "brand": "brand name here",
  "nutrition_basis": "per 100g",
  "serving_size_g": 0,
  "nutrition_per_100g": {
    "energy": 0,
    "protein": 0,
    "fat": 0,
    "saturated_fat": 0,
    "sugar": 0,
    "sodium": 0,
    "fiber": 0
  },
  "ingredients": [],
  "confidence": 0.95
}

All numeric values must be numbers (not strings).
sodium in mg. All other nutrients in grams. Energy in kcal.
"""


def analyze_product(images, max_retries: int = 3):
    """
    Analyse food product images using Gemini Vision.
    Accepts a list of PIL Image objects.
    Handles quota errors with exponential backoff retry.
    """
    for attempt in range(max_retries):
        try:
            if _USE_NEW_SDK:
                result = _call_new_sdk(images)
            else:
                result = _call_old_sdk(images)
            return result

        except Exception as e:
            err_str = str(e)

            # ── Rate limit / quota exceeded → wait and retry ──────────────
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                # Parse retry delay from error if available
                wait_sec = 20
                if "retry_delay" in err_str:
                    try:
                        # Extract seconds value from error message
                        idx = err_str.find("seconds: ")
                        if idx != -1:
                            val = err_str[idx + 9:idx + 15].split()[0].strip()
                            wait_sec = int(float(val)) + 2
                    except Exception:
                        wait_sec = 20

                if attempt < max_retries - 1:
                    print(f"Gemini quota hit. Waiting {wait_sec}s before retry {attempt + 2}/{max_retries}…")
                    time.sleep(wait_sec)
                    continue
                else:
                    print("Gemini quota exhausted after all retries.")
                    return _quota_exceeded_result()

            # ── JSON parse error ──────────────────────────────────────────
            elif isinstance(e, json.JSONDecodeError):
                print(f"Gemini JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return _fallback_result()

            # ── Any other error ───────────────────────────────────────────
            else:
                print(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return _fallback_result()

    return _fallback_result()


def _call_new_sdk(images):
    """Call Gemini using the new google.genai SDK."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Convert PIL images to bytes for new SDK
    import io
    parts = [_PROMPT]
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        parts.append(
            types.Part.from_bytes(
                data=buf.getvalue(),
                mime_type="image/jpeg"
            )
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=parts
    )
    return _parse_response(response.text)


def _call_old_sdk(images):
    """Call Gemini using the old google.generativeai SDK (fallback)."""
    import google.generativeai as genai_old
    genai_old.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai_old.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([_PROMPT] + images)
    return _parse_response(response.text)


def _parse_response(text: str) -> dict:
    """Strip markdown fences and parse JSON from Gemini response."""
    text = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    if "```" in text:
        parts = text.split("```")
        # Take the part after the first fence
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    data = json.loads(text)

    # Ensure confidence key exists
    if "confidence" not in data:
        data["confidence"] = 0.85

    # Ensure all nutrition keys exist with defaults
    defaults = {
        "energy": 0, "protein": 0, "fat": 0,
        "saturated_fat": 0, "sugar": 0, "sodium": 0, "fiber": 0
    }
    if "nutrition_per_100g" not in data:
        data["nutrition_per_100g"] = defaults
    else:
        for k, v in defaults.items():
            if k not in data["nutrition_per_100g"]:
                data["nutrition_per_100g"][k] = v

    # Ensure ingredients is a list
    if "ingredients" not in data or not isinstance(data["ingredients"], list):
        data["ingredients"] = []

    return data


def _fallback_result():
    """Return a zero-confidence fallback when API fails."""
    return {
        "product_name": "Unknown",
        "brand": "Unknown",
        "nutrition_basis": "Unknown",
        "serving_size_g": 0,
        "nutrition_per_100g": {
            "energy": 0, "protein": 0, "fat": 0,
            "saturated_fat": 0, "sugar": 0, "sodium": 0, "fiber": 0
        },
        "ingredients": [],
        "confidence": 0.0
    }


def _quota_exceeded_result():
    """Return a special result when quota is exhausted — higher confidence so app doesn't block."""
    return {
        "product_name": "Quota Exceeded",
        "brand": "Unknown",
        "nutrition_basis": "unavailable",
        "serving_size_g": 0,
        "nutrition_per_100g": {
            "energy": 0, "protein": 0, "fat": 0,
            "saturated_fat": 0, "sugar": 0, "sodium": 0, "fiber": 0
        },
        "ingredients": [],
        "confidence": 0.0,
        "_quota_exceeded": True
    }