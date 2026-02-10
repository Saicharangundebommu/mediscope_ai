import os
import cv2
import numpy as np
import json
import re
from PIL import Image
from paddleocr import PaddleOCR
import torch
import spacy
from fuzzywuzzy import fuzz
from skimage.filters import unsharp_mask

# ✅ Disable Paddle model host checking (important for Render)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ===============================
# Load SpaCy Model (Safe)
# ===============================
try:
    nlp = spacy.load("en_core_sci_md")
except:
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None
        print("Warning: Spacy model not loaded.")

# ===============================
# Lazy PaddleOCR Initialization
# ===============================
ocr_engine = None

def get_ocr_engine():
    """Load PaddleOCR only when needed (Render-safe)."""
    global ocr_engine
    if ocr_engine is None:
        print("Loading PaddleOCR engine...")

        ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            rec_algorithm="SVTR_LCNet",
            rec_batch_num=6,
            det_limit_side_len=960,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            max_text_length=100,
            drop_score=0.5,
            use_gpu=False,   # Render does not support GPU
            show_log=False
        )

    return ocr_engine

# ===============================
# Medical Dictionary (Short Sample)
# ===============================
MEDICATION_DICT = {
    "paracetamol": ["crocin", "dolo", "panadol"],
    "amoxicillin": ["amox", "augmentin"],
    "ibuprofen": ["brufen", "advil"]
}

# ===============================
# Medical Dictionary Correction
# ===============================
def apply_medical_dictionary_correction(text):
    if not text:
        return text

    words = re.findall(r"\b\w+\b", text.lower())
    corrected_text = text

    for word in words:
        if len(word) < 3 or word.isdigit():
            continue

        best_match = None
        best_score = 0

        for key, aliases in MEDICATION_DICT.items():
            score = fuzz.ratio(word, key.lower())
            if score > best_score and score > 75:
                best_score = score
                best_match = key

            for alias in aliases:
                score = fuzz.ratio(word, alias.lower())
                if score > best_score and score > 75:
                    best_score = score
                    best_match = key

        if best_match:
            corrected_text = re.sub(rf"\b{word}\b", best_match, corrected_text, flags=re.IGNORECASE)

    return corrected_text

# ===============================
# Image Preprocessing
# ===============================
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    enhanced_path = os.path.join(
        os.path.dirname(image_path),
        "enhanced_" + os.path.basename(image_path)
    )

    cv2.imwrite(enhanced_path, thresh)

    return {
        "original": image_path,
        "enhanced": enhanced_path
    }

# ===============================
# Run OCR Pass
# ===============================
def run_ocr(image_path):
    engine = get_ocr_engine()
    result = engine.ocr(image_path, cls=True)

    extracted_text = ""
    confidence_scores = []

    if result and result[0]:
        for line in result[0]:
            extracted_text += line[1][0] + "\n"
            confidence_scores.append(float(line[1][1]))

    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    return extracted_text.strip(), avg_conf

# ===============================
# Entity Extraction
# ===============================
def extract_medical_entities(text):
    meds = []

    for key in MEDICATION_DICT.keys():
        if re.search(rf"\b{key}\b", text, re.IGNORECASE):
            meds.append(key)

    return list(set(meds))

# ===============================
# Main Prescription OCR Pipeline
# ===============================
def process_prescription_with_enhanced_ocr(image_path, output_dir=None):
    image_data = preprocess_image(image_path)

    if not image_data:
        return {"error": "Image preprocessing failed"}

    raw_text, confidence = run_ocr(image_data["enhanced"])

    corrected_text = apply_medical_dictionary_correction(raw_text)
    medications = extract_medical_entities(corrected_text)

    results = {
        "image_path": image_path,
        "raw_text": raw_text,
        "cleaned_text": corrected_text,
        "medications": medications,
        "confidence": float(confidence) * 100
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)

    return results

# ===============================
# Script Testing Mode
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True)
    parser.add_argument("--output", "-o", default="./output")

    args = parser.parse_args()

    results = process_prescription_with_enhanced_ocr(args.image, args.output)

    print("\n===== OCR Results =====")
    print(json.dumps(results, indent=4))
