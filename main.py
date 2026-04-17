import easyocr
import numpy as np
from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import re
from difflib import get_close_matches
pytesseract = None


app = FastAPI()

class DrugRequest(BaseModel):
    drugs: List[str]

from fastapi import FastAPI

reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return reader

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return reader

@app.get("/")
def home():
    return {"message": "PolyGuard Backend is Running 🚀"}


@app.post("/scan-prescription")
async def scan_prescription(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)

    reader_instance = get_reader()
    results = reader_instance.readtext(img_np)

    extracted_text = " ".join([res[1] for res in results if res[2] > 0.4])

    print("OCR TEXT:", extracted_text)

    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', extracted_text.lower())

    drugs = extract_drugs_smart(cleaned_text)
    drugs += map_brands(cleaned_text)

    return {
        "text": extracted_text,
        "drugs": list(set(drugs))
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------
# KNOWN DRUG LIST (CONTROL OUTPUT)
# -----------------------------
KNOWN_DRUGS = [
    # Common
    "paracetamol", "ibuprofen", "aspirin", "warfarin", "metformin",

    # From your strips
    "diclofenac", "nimesulide", "pantoprazole",
    "levocetirizine", "doxofylline",

    # Antibiotics
    "amoxicillin", "clavulanic", "clarithromycin", "ofloxacin", "ornidazole",

    # Syrup components
    "ambroxol", "chlorpheniramine", "dextromethorphan", "phenylephrine", "menthol",

    # Context drugs
    "alcohol", "vitamin k", "steroids", "clopidogrel", "antibiotics"
]

# -----------------------------
# BRAND → GENERIC MAPPING
# -----------------------------
BRAND_MAP = {
    "dolo": "paracetamol",
    "calpol": "paracetamol",
    "nomo": "paracetamol",
    "nomo p": "paracetamol",
    "livrin": "levocetirizine",
    "levocet": "levocetirizine",
    "a moxy": "amoxicillin",
    "amox": "amoxicillin",
    "olt": "pantoprazole",
    "fenak": "diclofenac",
    "gesic": "diclofenac",
    "alencuf": "dextromethorphan",
}

# -----------------------------
# SMART DRUG DETECTION
# -----------------------------
def extract_drugs_smart(text):
    text = text.lower()
    detected = []

    words = re.findall(r"[a-zA-Z]+", text)

    for drug in KNOWN_DRUGS:
        if drug in text:
            detected.append(drug)
            continue

        for word in words:
            match = get_close_matches(word, [drug], n=1, cutoff=0.6)
            if match:
                detected.append(drug)
                break

    return list(set(detected))


def map_brands(text):
    text = text.lower()
    mapped = []

    for brand, generic in BRAND_MAP.items():
        if brand in text:
            mapped.append(generic)

    return mapped


INTERACTIONS = [
    #  BLOOD THINNER + PAINKILLERS
    ("warfarin", "aspirin", "high", "Severe bleeding risk"),
    ("warfarin", "ibuprofen", "high", "High bleeding risk"),
    ("aspirin", "ibuprofen", "medium", "Stomach bleeding risk"),

    #  NSAID COMBINATIONS
    ("diclofenac", "ibuprofen", "high", "High risk of stomach ulcers and bleeding"),
    ("diclofenac", "nimesulide", "high", "Very high bleeding and liver risk"),
    ("nimesulide", "paracetamol", "high", "Severe liver damage risk"),

    #  ALCOHOL INTERACTIONS
    ("paracetamol", "alcohol", "high", "Severe liver damage"),
    ("diclofenac", "alcohol", "high", "Stomach bleeding risk"),
    ("metformin", "alcohol", "high", "Lactic acidosis (life-threatening)"),

    # ANTIBIOTICS
    ("amoxicillin", "oral contraceptives", "medium", "Reduces birth control effectiveness"),
    ("clarithromycin", "statins", "high", "Muscle breakdown risk"),
    ("ofloxacin", "antibiotics", "medium", "Resistance risk"),

    # BP + HEART
    ("lisinopril", "potassium", "high", "Dangerous potassium levels"),
    ("digoxin", "furosemide", "high", "Irregular heart rhythm risk"),

    #  DIABETES
    ("insulin", "beta blockers", "medium", "Masks low sugar symptoms"),

    #  PAIN + STEROIDS
    ("ibuprofen", "steroids", "high", "Severe stomach ulcers"),

    #  BLOOD THINNER CONTROL
    ("warfarin", "vitamin k", "medium", "Reduces effectiveness"),

    #  BRAIN / NERVOUS SYSTEM
    ("antidepressants", "tramadol", "high", "Serotonin syndrome"),
    ("dextromethorphan", "antidepressants", "high", "Serotonin syndrome"),

    #  COLD / COUGH COMBO RISKS
    ("phenylephrine", "hypertension drugs", "medium", "Increases blood pressure"),

    #  ANTIPLATELET
    ("aspirin", "clopidogrel", "high", "Severe bleeding risk"),

    # LIVER / COMBO DRUGS
    ("paracetamol", "diclofenac", "medium", "Liver stress risk"),

    # generally safe
    ("amoxicillin", "paracetamol", "low", "Generally safe but monitor usage"),
    ("levocetirizine", "alcohol", "medium", "Increased drowsiness"),
    ("pantoprazole", "antibiotics", "low", "May affect absorption"),
    ("levocetirizine", "paracetamol", "low", "Generally safe combination"),
    ("amoxicillin", "paracetamol", "low", "Commonly prescribed together"),
    ("pantoprazole", "paracetamol", "low", "Safe, often used together"),
    ("levocetirizine", "pantoprazole", "low", "No major interaction"),
    ("doxofylline", "paracetamol", "low", "Generally safe"),
    ("amoxicillin", "levocetirizine", "low", "Commonly used for infections and allergies"),
    ("diclofenac", "paracetamol", "medium", "Monitor liver usage"),
    ("paracetamol", "ofloxacin", "low", "Generally safe"),
    ("clarithromycin", "paracetamol", "low", "Safe combination"),
    ("ornidazole", "ofloxacin", "medium", "Combined antibiotic usage"),
]


# -----------------------------
# FINAL ANALYZE ENDPOINT
# -----------------------------

# -----------------------------
# /check → INTERACTION DETECTION
# -----------------------------
@app.post("/check")
def check(request: DrugRequest = Body(...)):
    drugs = request.drugs

    results = []

    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            d1 = drugs[i]
            d2 = drugs[j]

            for interaction in INTERACTIONS:
                if (
                    (d1 == interaction[0] and d2 == interaction[1]) or
                    (d2 == interaction[0] and d1 == interaction[1])
                ):
                    results.append({
                        "drug1": d1,
                        "drug2": d2,
                        "severity": interaction[2],
                        "effect": interaction[3],
                        "message": f"Taking {d1} and {d2} together can be dangerous. {interaction[3]}"
                    })

    if not results:
        results.append({
            "severity": "safe",
            "message": "No dangerous interaction detected. These medicines are generally safe together, but always consult a doctor."
    })

    return {"interactions": results}


# -----------------------------
# /report → SIMPLE PLACEHOLDER
# -----------------------------
@app.post("/report")
def report(data: dict):
    interactions = data.get("interactions", [])

# extract drugs from interactions
    drugs = []
    for item in interactions:
        if "drug1" in item:
            drugs.append(item["drug1"])
        if "drug2" in item:
            drugs.append(item["drug2"])


    drugs = list(set(drugs))
    interactions = data.get("interactions", [])

    summary = []
    risk_score = 0

    for item in interactions:
        if "drug1" in item:
            severity = item.get("severity", "low")

            # assign score
            if severity == "high":
                risk_score += 3
            elif severity == "medium":
                risk_score += 2
            else:
                risk_score += 1

            # timing suggestion (SAFE LOGIC)
            # 🔥 ADVANCED TIMING LOGIC
            if severity == "high":
                timing = {
                    "instruction": "Do NOT take these medicines together",
                    "gap": "Avoid completely or only under doctor supervision",
                    "food_note": "Take with food if required to reduce side effects",
                    "monitoring": "Watch for symptoms like bleeding, dizziness, or unusual pain"
                }

            elif severity == "medium":
                timing = {
                    "instruction": "Take medicines at different times",
                    "gap": "Maintain at least 4–6 hour gap",
                    "food_note": "Prefer taking after meals to reduce stomach irritation",
                    "monitoring": "Monitor for mild symptoms like nausea or discomfort"
                }

            else:
                timing = {
                    "instruction": "Medicines can be taken together if prescribed",
                    "gap": "No strict gap required",
                    "food_note": "Follow general prescription guidelines",
                    "monitoring": "No major concerns, but stay alert"
                }

            summary.append({
                "pair": f"{item['drug1']} + {item['drug2']}",
                "severity": severity,
                "effect": item.get("effect", ""),
                "timing_advice": timing["instruction"],
                "recommended_gap": timing["gap"],
                "food_instruction": timing["food_note"],
                "monitoring": timing["monitoring"],
                "simple_explanation": f"{item['drug1']} aur {item['drug2']} ek saath lene se dikkat ho sakti hai. {item.get('effect','')}"
            })

    if not summary:
        summary.append({
            "message": "No dangerous interaction detected. Medicines are generally safe together.",
            "timing_advice": "Follow doctor's prescription",
            "simple_explanation": "Yeh medicines generally safe hain saath lene ke liye."
        })

    # overall risk level
    if risk_score >= 5:
        overall = "HIGH RISK"
    elif risk_score >= 3:
        overall = "MODERATE RISK"
    else:
        overall = "LOW RISK"

    if len(drugs) >= 2:
        schedule = f"Space {drugs[0]} and {drugs[1]} by 4–6 hours if possible"
    elif len(drugs) == 1:
        schedule = f"Take {drugs[0]} as prescribed by doctor"
    else:
        schedule = "No medicines detected"

    return {
        "patient_report": {
            "total_drugs": len(drugs),
            "drugs_detected": drugs,
            "interaction_count": len(summary),
            
            "details": summary,
            "daily_schedule_hint": schedule,
            "final_advice": "Always consult a doctor before combining medicines."
        }
    }
