"""
PolyGuard Backend — main.py
Clean, production-ready FastAPI backend.

Endpoints:
  GET  /            health check
  POST /analyze     image → detected drugs ONLY
  POST /check       drug list → interactions + rich report
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import pytesseract
import io
import re
from difflib import get_close_matches

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="PolyGuard API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────

class DrugListRequest(BaseModel):
    drugs: List[str]

# ─────────────────────────────────────────────────────────────
# KNOWN DRUG DATABASE
# ─────────────────────────────────────────────────────────────

KNOWN_DRUGS = [
    # Painkillers / Antipyretics
    "paracetamol", "ibuprofen", "aspirin", "diclofenac", "nimesulide",
    "naproxen", "tramadol", "codeine", "morphine",

    # Antibiotics
    "amoxicillin", "clavulanic", "clarithromycin", "ofloxacin",
    "ornidazole", "metronidazole", "azithromycin", "doxycycline",
    "ciprofloxacin", "cetirizine",

    # Gastro
    "pantoprazole", "omeprazole", "ranitidine", "domperidone",

    # Antihistamines / Respiratory
    "levocetirizine", "cetirizine", "doxofylline", "salbutamol",
    "montelukast", "dextromethorphan", "ambroxol", "chlorpheniramine",
    "phenylephrine", "menthol",

    # Diabetes
    "metformin", "insulin", "glipizide", "gliclazide",

    # Cardiovascular
    "warfarin", "aspirin", "clopidogrel", "atorvastatin", "amlodipine",
    "losartan", "lisinopril", "metoprolol", "digoxin", "furosemide",
    "spironolactone",

    # Hormones / Thyroid
    "levothyroxine",

    # Others
    "alcohol", "vitamin k", "steroids", "antidepressants",
]

# ─────────────────────────────────────────────────────────────
# BRAND → GENERIC MAP
# ─────────────────────────────────────────────────────────────

BRAND_MAP = {
    # Paracetamol brands
    "dolo": "paracetamol", "calpol": "paracetamol",
    "nomo": "paracetamol", "crocin": "paracetamol",
    "pyrigesic": "paracetamol", "metacin": "paracetamol",

    # NSAIDs
    "fenak": "diclofenac", "voveran": "diclofenac",
    "brufen": "ibuprofen", "advil": "ibuprofen",

    # Antibiotics
    "amox": "amoxicillin", "novamox": "amoxicillin",
    "zithromax": "azithromycin", "azee": "azithromycin",

    # Antacids
    "pan": "pantoprazole", "pantop": "pantoprazole",
    "rantac": "ranitidine",

    # Antihistamines
    "levocet": "levocetirizine", "livrin": "levocetirizine",
    "alerid": "cetirizine", "zyrtec": "cetirizine",

    # Cholesterol
    "lipitor": "atorvastatin", "storvas": "atorvastatin",

    # BP
    "nifedipine": "amlodipine",

    # Cough/Cold
    "alencuf": "dextromethorphan",
}

# ─────────────────────────────────────────────────────────────
# INTERACTION DATABASE
# (drug1, drug2, severity, effect, recommendation)
# ─────────────────────────────────────────────────────────────

INTERACTIONS = [
    # ── HIGH RISK ──────────────────────────────────────────
    ("warfarin",      "aspirin",        "high",
     "Severe internal bleeding risk — both thin the blood via different mechanisms",
     "Avoid combination. Use only under strict medical supervision with regular INR monitoring."),

    ("warfarin",      "ibuprofen",      "high",
     "Ibuprofen raises warfarin blood levels, causing dangerous bleeding",
     "Use paracetamol for pain instead. Never self-medicate if on warfarin."),

    ("warfarin",      "diclofenac",     "high",
     "Diclofenac increases anticoagulant effect of warfarin — bleeding risk",
     "Avoid NSAIDs entirely while on warfarin. Consult doctor urgently."),

    ("aspirin",       "clopidogrel",    "high",
     "Double antiplatelet effect — major GI and cerebral bleeding risk",
     "Only combine if explicitly prescribed for dual antiplatelet therapy. Monitor closely."),

    ("diclofenac",    "ibuprofen",      "high",
     "Two NSAIDs together — extreme stomach ulcer and kidney damage risk",
     "Never take two NSAIDs simultaneously. Use only one at a time."),

    ("diclofenac",    "nimesulide",     "high",
     "Both NSAIDs — compounded liver toxicity and GI bleeding",
     "Do not combine. Serious hepatotoxicity reported in India. Consult doctor."),

    ("nimesulide",    "paracetamol",    "high",
     "Severe liver damage risk — both metabolized in liver",
     "Avoid combination. Use one analgesic at a time."),

    ("paracetamol",   "alcohol",        "high",
     "Even moderate alcohol with paracetamol causes acute liver failure",
     "Strictly avoid alcohol during paracetamol use. Fatal at high doses."),

    ("diclofenac",    "alcohol",        "high",
     "Alcohol amplifies diclofenac's stomach-lining damage — bleeding ulcers",
     "Avoid alcohol entirely when taking diclofenac."),

    ("metformin",     "alcohol",        "high",
     "Lactic acidosis — life-threatening buildup of lactic acid in blood",
     "Absolutely avoid alcohol with metformin. Can be fatal."),

    ("clarithromycin","atorvastatin",   "high",
     "Clarithromycin blocks statin metabolism → severe muscle breakdown (rhabdomyolysis)",
     "Stop atorvastatin during clarithromycin course. Resume after antibiotic is finished."),

    ("ibuprofen",     "steroids",       "high",
     "Both damage stomach lining — risk of perforated ulcer",
     "Use paracetamol instead of ibuprofen if on steroids. Add stomach protection."),

    ("lisinopril",    "spironolactone", "high",
     "Both raise potassium — can cause fatal hyperkalemia and cardiac arrest",
     "Requires close electrolyte monitoring. Only with doctor supervision."),

    ("digoxin",       "furosemide",     "high",
     "Furosemide depletes potassium → potassium loss increases digoxin toxicity",
     "Monitor potassium and digoxin levels closely. Report any palpitations."),

    ("dextromethorphan","antidepressants","high",
     "Serotonin syndrome — life-threatening overstimulation of serotonin receptors",
     "Do not combine. Use a non-serotonergic cough suppressant instead."),

    ("tramadol",      "antidepressants","high",
     "Serotonin syndrome risk — tremors, agitation, rapid heart rate",
     "Avoid tramadol if on SSRIs/SNRIs. Discuss alternative pain relief with doctor."),

    # ── MEDIUM RISK ────────────────────────────────────────
    ("aspirin",       "ibuprofen",      "medium",
     "Ibuprofen blocks aspirin's heart-protective effect — reduces cardioprotection",
     "Take aspirin 30 min before ibuprofen, or use paracetamol for pain."),

    ("warfarin",      "vitamin k",      "medium",
     "Vitamin K reduces warfarin's blood-thinning effect",
     "Keep vitamin K intake consistent. Sudden changes in leafy vegetables affect INR."),

    ("paracetamol",   "diclofenac",     "medium",
     "Additive liver load — use cautiously, especially with existing liver conditions",
     "Short-term use is generally acceptable. Avoid in liver disease. Monitor."),

    ("levocetirizine","alcohol",        "medium",
     "Increased drowsiness and impaired coordination",
     "Avoid alcohol while taking antihistamines. Do not drive."),

    ("ornidazole",    "ofloxacin",      "medium",
     "Both affect gut flora — risk of antibiotic-associated diarrhea",
     "Take probiotics after the course. Monitor for persistent diarrhea."),

    ("phenylephrine", "metoprolol",     "medium",
     "Phenylephrine (decongestant) raises BP, counteracting beta-blocker effect",
     "Use saline nasal drops instead. Avoid OTC decongestants if on beta-blockers."),

    ("insulin",       "metoprolol",     "medium",
     "Beta-blocker masks hypoglycemia symptoms (racing heart hidden)",
     "Monitor blood sugar more frequently. Recognize sweating as the main warning sign."),

    ("amoxicillin",   "metformin",      "medium",
     "Antibiotics can alter gut flora affecting metformin absorption slightly",
     "Monitor blood sugar during antibiotic course. Usually manageable."),

    ("ofloxacin",     "antidepressants","medium",
     "Quinolones may lower seizure threshold, risk increased with some antidepressants",
     "Use with caution. Report any unusual neurological symptoms immediately."),

    # ── LOW RISK ───────────────────────────────────────────
    ("amoxicillin",   "paracetamol",    "low",
     "Generally safe — commonly prescribed together for infections with pain/fever",
     "Safe to use together as prescribed. No special precautions needed."),

    ("levocetirizine","paracetamol",    "low",
     "No significant interaction — commonly combined for cold and allergy",
     "Safe combination. Take both as directed."),

    ("levocetirizine","pantoprazole",   "low",
     "No major interaction known",
     "Can be taken together. Follow prescription timing."),

    ("pantoprazole",  "paracetamol",    "low",
     "Pantoprazole protects stomach — actually beneficial with paracetamol",
     "Safe and often beneficial together."),

    ("doxofylline",   "paracetamol",    "low",
     "No significant interaction — both commonly prescribed for respiratory infections",
     "Safe to use together."),

    ("amoxicillin",   "levocetirizine", "low",
     "No significant interaction — commonly combined for respiratory infections",
     "Safe combination. Complete full antibiotic course."),

    ("pantoprazole",  "ofloxacin",      "low",
     "Pantoprazole may slightly reduce ofloxacin absorption if taken simultaneously",
     "Take ofloxacin 2 hours before pantoprazole for best absorption."),

    ("clarithromycin","paracetamol",    "low",
     "No significant interaction",
     "Safe to use together."),

    ("diclofenac",    "pantoprazole",   "low",
     "Pantoprazole protects the stomach from diclofenac's ulcer-causing effect",
     "Beneficial combination — pantoprazole reduces GI risk of diclofenac."),

    ("paracetamol",   "ofloxacin",      "low",
     "No significant interaction",
     "Safe to use together as prescribed."),
]

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def normalize_drug(name: str) -> str:
    """Lowercase and strip whitespace."""
    return name.strip().lower()


def extract_drugs_from_text(text: str) -> List[str]:
    """Multi-strategy drug detection from OCR text."""
    text = text.lower()
    detected = set()

    # Strategy 1: Direct substring match
    for drug in KNOWN_DRUGS:
        if drug in text:
            detected.add(drug)

    # Strategy 2: Brand name lookup
    for brand, generic in BRAND_MAP.items():
        if brand in text:
            detected.add(generic)

    # Strategy 3: Fuzzy word-level match (only if needed)
    words = re.findall(r"[a-zA-Z]{4,}", text)
    for word in words:
        match = get_close_matches(word, KNOWN_DRUGS, n=1, cutoff=0.75)
        if match:
            detected.add(match[0])

    return list(detected)


def run_interaction_check(drugs: List[str]):
    """
    Given a list of normalized drug names, return:
    - interactions list
    - risk level string
    - rich report dict
    """
    interactions = []
    highest_risk = "low"
    risk_score = 0

    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            d1, d2 = drugs[i], drugs[j]

            for entry in INTERACTIONS:
                a, b, severity, effect, recommendation = entry

                if (d1 == a and d2 == b) or (d1 == b and d2 == a):
                    interactions.append({
                        "drug1": d1,
                        "drug2": d2,
                        "severity": severity,
                        "effect": effect,
                        "recommendation": recommendation,
                        "message": f"{d1.title()} + {d2.title()}: {effect}"
                    })

                    if severity == "high":
                        highest_risk = "high"
                        risk_score += 3
                    elif severity == "medium" and highest_risk != "high":
                        highest_risk = "medium"
                        risk_score += 2
                    else:
                        risk_score += 1
                    break  # one match per pair is enough

    # Build overall risk label
    if risk_score >= 6:
        overall_risk = "HIGH RISK"
    elif risk_score >= 3:
        overall_risk = "MODERATE RISK"
    elif risk_score > 0:
        overall_risk = "LOW RISK"
    else:
        overall_risk = "NO RISK DETECTED"

    # Build schedule hint
    if len(drugs) >= 2:
        high_pairs = [i for i in interactions if i["severity"] == "high"]
        if high_pairs:
            p = high_pairs[0]
            schedule = (
                f"Do NOT take {p['drug1'].title()} and {p['drug2'].title()} together. "
                "Space other medicines by at least 2 hours."
            )
        else:
            schedule = (
                f"Space your medicines by 2–4 hours where possible. "
                f"Take {drugs[0].title()} in the morning and {drugs[1].title()} with meals if stomach upset occurs."
            )
    elif len(drugs) == 1:
        schedule = f"Take {drugs[0].title()} as prescribed by your doctor."
    else:
        schedule = "No medicines to schedule."

    # Build final advice
    if overall_risk == "HIGH RISK":
        final_advice = (
            "⛔ URGENT: One or more combinations in your list are HIGH RISK. "
            "Do NOT take these medicines together without consulting a doctor immediately. "
            "These interactions can cause severe internal bleeding, liver failure, or other life-threatening conditions."
        )
    elif overall_risk == "MODERATE RISK":
        final_advice = (
            "⚠️ CAUTION: Some combinations need care. "
            "Space medicines by 4–6 hours where advised. "
            "Monitor for symptoms like nausea, dizziness, or unusual bruising. "
            "Discuss with your pharmacist or doctor."
        )
    elif overall_risk == "LOW RISK":
        final_advice = (
            "✅ LOW RISK: Minor interactions found. "
            "These are generally manageable and commonly prescribed together. "
            "Follow dosing instructions carefully and consult a doctor if you notice any side effects."
        )
    else:
        final_advice = (
            "✅ SAFE: No known dangerous interactions found between these medicines. "
            "Always take medicines as prescribed. Consult your doctor before making any changes."
        )

    report = {
        "total_drugs": len(drugs),
        "drugs_detected": drugs,
        "interaction_count": len(interactions),
        "overall_risk": overall_risk,
        "risk_score": risk_score,
        "details": interactions,          # full detail list for report page
        "daily_schedule_hint": schedule,
        "final_advice": final_advice,
    }

    return interactions, overall_risk, report


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "PolyGuard API v3.0 is live 🚀",
        "endpoints": {
            "POST /analyze": "Upload image → get detected drugs",
            "POST /check":   "Send drug list → get interactions + report"
        }
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Accept an image of a medicine strip.
    Returns ONLY the list of detected drug names.
    Frontend accumulates multiple scans and calls /check when done.
    """
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # ── OCR ───────────────────────────────────────────
        extracted_text = ""
        try:
            image = Image.open(io.BytesIO(contents))

            # Convert to grayscale for better OCR
            gray = image.convert("L")

            # pytesseract with improved config
            custom_config = r"--oem 3 --psm 6"
            extracted_text = pytesseract.image_to_string(gray, config=custom_config).lower()

        except Exception as ocr_err:
            print(f"OCR error: {ocr_err}")
            # Don't crash — return empty result with clear message
            return {
                "drugs": [],
                "message": "OCR processing failed. Try a clearer, well-lit image.",
                "ocr_text": ""
            }

        print(f"OCR extracted: {extracted_text[:200]}")

        # ── DRUG DETECTION ────────────────────────────────
        detected = extract_drugs_from_text(extracted_text)

        print(f"Detected drugs: {detected}")

        return {
            "drugs": detected,
            "message": (
                f"Detected {len(detected)} medicine(s) from image."
                if detected
                else "No medicines detected. Try a clearer image with better lighting."
            ),
            "ocr_text_preview": extracted_text[:100] if extracted_text else ""
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Analyze endpoint error: {e}")
        return {
            "drugs": [],
            "message": "Analysis failed. Please try again.",
            "error": str(e)
        }


@app.post("/check")
def check_interactions(request: DrugListRequest):
    """
    Accept a list of drug names.
    Returns full interaction analysis + rich report.
    This is the ONLY endpoint that generates reports.
    """
    if not request.drugs:
        raise HTTPException(status_code=400, detail="Drug list cannot be empty")

    # Normalize all drug names to lowercase
    drugs = [normalize_drug(d) for d in request.drugs]
    drugs = list(dict.fromkeys(drugs))  # deduplicate preserving order

    if not drugs:
        raise HTTPException(status_code=400, detail="No valid drug names provided")

    interactions, overall_risk, report = run_interaction_check(drugs)

    return {
        "drugs": drugs,
        "interactions": interactions,
        "report": report
    }


# ─────────────────────────────────────────────────────────────
# LEGACY ALIASES (keep old endpoints working for any cached FE)
# ─────────────────────────────────────────────────────────────

@app.post("/check-interactions")
def check_interactions_alias(request: DrugListRequest):
    """Alias for /check — backwards compatibility."""
    return check_interactions(request)
