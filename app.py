import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Readmission Risk Stratification", page_icon="🏥", layout="wide")

AVG_RISK = 0.114   # average 30-day readmission rate in the training data
MAX_ROWS = 5000    # largest file accepted in batch mode

DISPOSITIONS = {
    "Home": "1",
    "Home with health service": "6",
    "Skilled nursing facility": "3",
    "Short-term hospital": "2",
    "Rehab facility": "22",
}
DX_GROUPS = ["Circulatory", "Diabetes", "Respiratory", "Digestive", "Injury",
             "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"]
A1C_VALUES = ["NotTested", "Norm", ">7", ">8"]
GLU_VALUES = ["NotTested", "Norm", ">200", ">300"]
INSULIN_VALUES = ["No", "Steady", "Up", "Down"]
GENDERS = ["Female", "Male"]

NUMERIC_RANGES = {
    "age": (0, 100),
    "weight_kg": (25, 250),
    "time_in_hospital": (1, 14),
    "num_medications": (0, 100),
    "num_lab_procedures": (0, 150),
    "num_procedures": (0, 10),
    "number_diagnoses": (1, 20),
    "number_inpatient": (0, 30),
    "number_emergency": (0, 80),
    "number_outpatient": (0, 60),
}
CATEGORICAL = {
    "gender": GENDERS,
    "primary_dx_group": DX_GROUPS,
    "A1Cresult": A1C_VALUES,
    "max_glu_serum": GLU_VALUES,
    "insulin": INSULIN_VALUES,
}
FLAGS = ["has_heart_failure", "has_renal_disease", "has_copd"]
REQUIRED = (list(NUMERIC_RANGES.keys()) + list(CATEGORICAL.keys())
            + ["discharge_disposition_id"] + FLAGS)


@st.cache_resource
def load_assets():
    art = joblib.load("readmission_model.joblib")
    with open("defaults.json") as f:
        defaults = json.load(f)
    return art, defaults


art, defaults = load_assets()
model, calibrator = art["model"], art["calibrator"]
threshold, feature_columns = art["threshold"], art["feature_columns"]


def build_model_frame(d):
    """Turn the simple input columns into the exact columns the model was trained on."""
    X = pd.DataFrame([defaults] * len(d), index=d.index)
    total_prior = d["number_inpatient"] + d["number_emergency"] + d["number_outpatient"]

    X["gender"] = d["gender"]
    X["weight_kg"] = d["weight_kg"].astype(float)
    X["weight_missing"] = 0
    X["age_num"] = np.minimum(d["age"] // 10, 9) * 10 + 5
    for c in ["time_in_hospital", "num_medications", "num_lab_procedures",
              "num_procedures", "number_diagnoses", "number_inpatient",
              "number_emergency", "number_outpatient"]:
        X[c] = d[c].astype(float)
    X["primary_dx_group"] = d["primary_dx_group"]
    X["discharge_disposition_id"] = d["discharge_disposition_id"].astype(str)
    X["A1Cresult"] = d["A1Cresult"]
    X["max_glu_serum"] = d["max_glu_serum"]
    X["insulin"] = d["insulin"]

    hf = d["has_heart_failure"].astype(int)
    renal = d["has_renal_disease"].astype(int)
    copd = d["has_copd"].astype(int)
    X["has_diabetes_dx"] = 1
    X["has_heart_failure"] = hf
    X["has_renal_disease"] = renal
    X["has_copd"] = copd
    X["comorbidity_score"] = 1 + hf + renal + copd
    X["total_prior_visits"] = total_prior
    X["emergency_share"] = d["number_emergency"] / (total_prior + 1)
    X["abnormal_a1c"] = d["A1Cresult"].isin([">7", ">8"]).astype(int)
    X["abnormal_glucose"] = d["max_glu_serum"].isin([">200", ">300"]).astype(int)
    return X[feature_columns]


def score(d):
    X = build_model_frame(d)
    raw = model.predict_proba(X)[:, 1]
    return calibrator.predict(raw)


def patient_form():
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", 0, 100, 65)
        gender = st.selectbox("Gender", GENDERS)
        weight_kg = st.number_input("Weight (kg)", 25, 250, 80)
        time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
        num_medications = st.slider("Number of medications", 1, 80, 15)
        num_lab_procedures = st.slider("Number of lab procedures", 1, 130, 43)
        num_procedures = st.slider("Number of procedures", 0, 6, 1)
        number_diagnoses = st.slider("Number of diagnoses", 1, 16, 8)
        dx_group = st.selectbox("Primary diagnosis group", DX_GROUPS)
    with c2:
        number_inpatient = st.number_input("Inpatient admissions, past 12 months", 0, 20, 0)
        number_emergency = st.number_input("Emergency visits, past 12 months", 0, 20, 0)
        number_outpatient = st.number_input("Outpatient visits, past 12 months", 0, 40, 0)
        disposition_label = st.selectbox("Discharged to", list(DISPOSITIONS.keys()))
        a1c = st.selectbox("HbA1c result", A1C_VALUES)
        glucose = st.selectbox("Serum glucose result", GLU_VALUES)
        insulin = st.selectbox("Insulin", INSULIN_VALUES)
        heart_failure = st.checkbox("Heart failure")
        renal = st.checkbox("Kidney disease")
        copd = st.checkbox("COPD")
    return pd.DataFrame([{
        "age": age, "weight_kg": weight_kg, "gender": gender, "time_in_hospital": time_in_hospital,
        "num_medications": num_medications, "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures, "number_diagnoses": number_diagnoses,
        "primary_dx_group": dx_group, "number_inpatient": number_inpatient,
        "number_emergency": number_emergency, "number_outpatient": number_outpatient,
        "discharge_disposition_id": DISPOSITIONS[disposition_label],
        "A1Cresult": a1c, "max_glu_serum": glucose, "insulin": insulin,
        "has_heart_failure": int(heart_failure), "has_renal_disease": int(renal),
        "has_copd": int(copd),
    }])


def reasons_for(r):
    reasons = []
    if r["number_inpatient"] >= 2:
        reasons.append("2 or more inpatient admissions in the past year")
    if r["number_emergency"] >= 1:
        reasons.append("Emergency visits in the past year")
    if r["number_diagnoses"] >= 9:
        reasons.append("High number of diagnoses")
    if r["num_medications"] >= 20:
        reasons.append("Polypharmacy (20 or more medications)")
    if r["time_in_hospital"] >= 7:
        reasons.append("Long hospital stay")
    if r["has_heart_failure"] or r["has_renal_disease"] or r["has_copd"]:
        reasons.append("Serious chronic condition recorded")
    return reasons


def show_result(risk, r, educational):
    st.subheader("Result")
    st.metric("Estimated 30-day readmission risk", f"{risk:.1%}")
    st.caption("Average risk across all patients in the training data is about 11%.")

    if not educational:
        if risk >= threshold:
            st.error(f"HIGH RISK (at or above the {threshold:.1%} action threshold). "
                     "Recommended: nurse phone call within 48 hours and a primary "
                     "care visit within 7 days.")
        else:
            st.success(f"Standard risk (below the {threshold:.1%} action threshold). "
                       "Routine discharge follow-up.")
    else:
        ratio = risk / AVG_RISK
        if ratio >= 1.2:
            st.warning(f"This estimate is about {ratio:.1f} times the average "
                       "for hospital patients in the training data. "
                       "Please discuss your follow-up care with your doctor.")
        else:
            st.info("This estimate is close to or below the average for hospital "
                    "patients in the training data. It does not mean you are "
                    "safe. Follow your doctor's discharge advice.")
        st.caption("Try changing one input at a time (for example, prior "
                   "admissions) to see which factors move the estimate most.")

    reasons = reasons_for(r)
    if reasons:
        st.write("**Factors present that are commonly linked to higher risk:**")
        for item in reasons:
            st.write("- " + item)
    st.caption("These are statistical associations, not causes. "
               "A clinician makes the final decision.")


def validate(raw):
    """Check an uploaded table. Returns (valid_rows, skipped_rows, missing_columns)."""
    d = raw.copy()
    d.columns = [str(c).strip() for c in d.columns]
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        return None, None, missing

    problems = pd.Series("", index=d.index)

    for c, (lo, hi) in NUMERIC_RANGES.items():
        d[c] = pd.to_numeric(d[c], errors="coerce")
        bad = d[c].isna() | (d[c] < lo) | (d[c] > hi)
        problems.loc[bad] = problems.loc[bad] + f"{c} missing or out of range; "

    d["gender"] = d["gender"].astype(str).str.strip().str.capitalize()
    for c in ["primary_dx_group", "A1Cresult", "max_glu_serum", "insulin"]:
        d[c] = d[c].astype(str).str.strip()
    for c, allowed in CATEGORICAL.items():
        bad = ~d[c].isin(allowed)
        problems.loc[bad] = problems.loc[bad] + f"{c} is not an allowed value; "

    truthy = {"1": 1, "1.0": 1, "yes": 1, "y": 1, "true": 1,
              "0": 0, "0.0": 0, "no": 0, "n": 0, "false": 0}
    for c in FLAGS:
        d[c] = d[c].astype(str).str.strip().str.lower().map(truthy)
        bad = d[c].isna()
        problems.loc[bad] = problems.loc[bad] + f"{c} must be 0/1 or yes/no; "

    disp = pd.to_numeric(d["discharge_disposition_id"], errors="coerce")
    bad = disp.isna() | (disp < 1) | (disp > 30) | (disp % 1 != 0)
    problems.loc[bad] = problems.loc[bad] + "discharge_disposition_id must be a whole number 1-30; "
    d["discharge_disposition_id"] = disp.fillna(1).astype(int).astype(str)

    ok = problems == ""
    valid = d[ok].copy()
    skipped = pd.DataFrame({"File row": d.index[~ok] + 2, "Problem": problems[~ok].values})
    return valid, skipped, []


def template_frame():
    return pd.DataFrame([
        {"patient_label": "A001", "weight_kg": 90, "age": 72, "gender": "Female", "time_in_hospital": 9,
         "num_medications": 25, "num_lab_procedures": 55, "num_procedures": 1,
         "number_diagnoses": 9, "primary_dx_group": "Circulatory", "number_inpatient": 3,
         "number_emergency": 1, "number_outpatient": 0, "discharge_disposition_id": 3,
         "A1Cresult": "NotTested", "max_glu_serum": "NotTested", "insulin": "Up",
         "has_heart_failure": 1, "has_renal_disease": 0, "has_copd": 0},
        {"patient_label": "A002", "weight_kg": 78, "age": 45, "gender": "Male", "time_in_hospital": 2,
         "num_medications": 5, "num_lab_procedures": 30, "num_procedures": 0,
         "number_diagnoses": 4, "primary_dx_group": "Diabetes", "number_inpatient": 0,
         "number_emergency": 0, "number_outpatient": 0, "discharge_disposition_id": 1,
         "A1Cresult": "Norm", "max_glu_serum": "NotTested", "insulin": "No",
         "has_heart_failure": 0, "has_renal_disease": 0, "has_copd": 0},
        {"patient_label": "A003", "weight_kg": 65, "age": 61, "gender": "Female", "time_in_hospital": 5,
         "num_medications": 14, "num_lab_procedures": 40, "num_procedures": 1,
         "number_diagnoses": 7, "primary_dx_group": "Respiratory", "number_inpatient": 1,
         "number_emergency": 0, "number_outpatient": 2, "discharge_disposition_id": 6,
         "A1Cresult": ">7", "max_glu_serum": "NotTested", "insulin": "Steady",
         "has_heart_failure": 0, "has_renal_disease": 0, "has_copd": 1},
    ])


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("🏥 30-Day Readmission Risk Stratification")
st.warning("Administrative risk stratification tool for post-discharge care "
           "coordination. It is NOT a diagnostic tool and does not replace "
           "physician judgment.")

mode = st.sidebar.radio(
    "Mode",
    ["Single patient (care team)",
     "Upload a file (batch scoring)",
     "What-if explorer (education)"],
)

if mode == "Single patient (care team)":
    st.subheader("Patient details at discharge")
    patient = patient_form()
    if st.button("Calculate risk", type="primary"):
        risk = float(score(patient)[0])
        show_result(risk, patient.iloc[0], educational=False)

elif mode == "What-if explorer (education)":
    st.info("Educational demo. It shows how hospital-record factors change a "
            "statistical estimate. It is not a symptom checker, not medical "
            "advice, and not a diagnosis.")
    recently_hospitalised = st.radio(
        "Were you recently discharged from a hospital stay (inpatient) with diabetes?",
        ["Yes", "No"],
        index=None,
    )
    if recently_hospitalised is None:
        st.stop()
    if recently_hospitalised == "No":
        st.error(
            "This model was trained only on hospital stays of patients with "
            "diabetes, so it cannot give a meaningful result for you. "
            "If you have symptoms or health concerns, please talk to a doctor "
            "or a healthcare professional. If you have severe symptoms such as "
            "chest pain, trouble breathing, confusion or fainting, contact "
            "your local emergency number straight away."
        )
        st.stop()
    st.subheader("Try different values")
    patient = patient_form()
    if st.button("Calculate estimate", type="primary"):
        risk = float(score(patient)[0])
        show_result(risk, patient.iloc[0], educational=True)

else:
    st.subheader("Score a file of discharges")
    st.info("Upload a CSV with one row per discharge. The app scores every row and "
            "returns a ranked follow-up queue. Files are processed in memory and "
            "this app does not save them.")

    st.download_button("Download CSV template",
                       template_frame().to_csv(index=False).encode("utf-8"),
                       "readmission_template.csv", "text/csv")
    with st.expander("Column guide"):
        st.markdown(
            "**Required columns:** `age` (0-100), `gender` (Female or Male), "
            "`time_in_hospital` (1-14 days), `num_medications`, `num_lab_procedures`, "
            "`num_procedures`, `number_diagnoses`, `primary_dx_group` "
            "(" + ", ".join(DX_GROUPS) + "), `number_inpatient`, `number_emergency`, "
            "`number_outpatient` (visits in the past 12 months), "
            "`discharge_disposition_id` (whole number 1-30, for example 1 = home, "
            "3 = skilled nursing facility, 6 = home with health service), "
            "`A1Cresult` (" + ", ".join(A1C_VALUES) + "), `max_glu_serum` ("
            + ", ".join(GLU_VALUES) + "), `insulin` (" + ", ".join(INSULIN_VALUES)
            + "), and `has_heart_failure`, `has_renal_disease`, `has_copd` "
            "(1/0 or yes/no).\n\n"
            "**Optional:** `patient_label`, an anonymous label of your choice "
            f"(never a name or real ID). Maximum {MAX_ROWS:,} rows."
        )

    confirmed = st.checkbox(
        "I confirm this file contains discharges from inpatient hospital stays of "
        "patients with diabetes, and contains no names or real patient identifiers."
    )
    uploaded = st.file_uploader("Browse for your CSV file", type=["csv"])

    if uploaded is None:
        st.stop()
    if not confirmed:
        st.warning("Please tick the confirmation box above to score the file.")
        st.stop()

    try:
        raw = pd.read_csv(uploaded)
    except Exception:
        st.error("The file could not be read as a CSV. Please check the format "
                 "against the template.")
        st.stop()

    if len(raw) == 0:
        st.error("The file has no rows.")
        st.stop()
    if len(raw) > MAX_ROWS:
        st.error(f"The file has {len(raw):,} rows. The limit is {MAX_ROWS:,}.")
        st.stop()

    valid, skipped, missing = validate(raw)
    if missing:
        st.error("These required columns are missing: " + ", ".join(missing)
                 + ". Please compare your file with the template.")
        st.stop()
    if len(skipped) > 0:
        with st.expander(f"{len(skipped)} row(s) skipped because of problems"):
            st.dataframe(skipped, hide_index=True)
    if len(valid) == 0:
        st.error("No valid rows to score.")
        st.stop()

    risk = score(valid)
    res = valid.copy()
    res["risk"] = risk
    res["Risk %"] = (risk * 100).round(1)
    res["Tier"] = np.where(risk >= threshold, "High",
                           np.where(risk >= AVG_RISK, "Medium", "Low"))
    res = res.sort_values("risk", ascending=False)

    n_high = int((res["Tier"] == "High").sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Discharges scored", f"{len(res):,}")
    m2.metric("High risk", f"{n_high:,} ({n_high / len(res):.0%})")
    m3.metric("Average risk", f"{res['risk'].mean():.1%}")

    st.caption(f"High risk means at or above the {threshold:.1%} action threshold: "
               "nurse phone call within 48 hours and a primary care visit within 7 days. "
               f"Medium is at or above the training-data average ({AVG_RISK:.1%}).")

    show_cols = (["patient_label"] if "patient_label" in res.columns else []) + [
        "Risk %", "Tier", "age", "number_inpatient", "number_emergency",
        "time_in_hospital", "num_medications", "primary_dx_group"]
    st.subheader("Follow-up queue (highest risk first)")
    st.dataframe(res[show_cols], hide_index=True)

    tier_counts = res["Tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    st.bar_chart(tier_counts)

    out_cols = [c for c in res.columns if c != "risk"]
    st.download_button("Download results (CSV)",
                       res[out_cols].to_csv(index=False).encode("utf-8"),
                       "readmission_risk_results.csv", "text/csv")
    st.caption("Scores are statistical estimates for prioritising follow-up. "
               "They are associations, not causes. A clinician makes the final decision.")
