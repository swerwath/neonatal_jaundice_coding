import csv
import os

NOTE_EVENT_CSV = "NOTEEVENTS.csv"
DIAGNOSIS_CSV = "DIAGNOSES_ICD.csv"
ADMISSIONS_CSV = "ADMISSIONS.csv"

CSV_DIR_PATH = "/Volumes/scottd/mimic_uncompressed"

NEONATAL_JAUNDICE_CODES = ["7730", "7731", "7732", "7741", "7742", "77430", "77431", "77439", "7746"]

# A HAMD is a unique identifier for a single hospital stay for a single patient
# This function returns all HAMDs that correpsond to a hospital stay during which
# a patient was diagnosed with neonatal jaundice
def get_relevant_hamds(dir_path=CSV_DIR_PATH, diagnosis_csv=DIAGNOSIS_CSV):
    hamds = []

    csv_path = os.path.join(dir_path, diagnosis_csv)
    with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["ICD9_CODE"] in NEONATAL_JAUNDICE_CODES:
                    hamds.append(int(row["HADM_ID"]))
    return set(hamds)

def get_relevant_notesets(hamds, dir_path=CSV_DIR_PATH, noteevent_csv=NOTE_EVENT_CSV):
    notesets = {h : "" for h in hamds}

    csv_path = os.path.join(dir_path, noteevent_csv)
    with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["HADM_ID"] != "" and int(row["HADM_ID"]) in notesets.keys() and row["ISERROR"] != '1' and "discharge" in row["TEXT"].lower():
                    notesets[int(row["HADM_ID"])] += "\n" + row["TEXT"]
    return notesets

def get_all_neonate_hamds(dir_path=CSV_DIR_PATH, admissions_csv=ADMISSIONS_CSV):
    hamds = []

    csv_path = os.path.join(dir_path, admissions_csv)
    with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["ADMISSION_TYPE"] == "NEWBORN":
                    hamds.append(int(row["HADM_ID"]))
    return set(hamds)
