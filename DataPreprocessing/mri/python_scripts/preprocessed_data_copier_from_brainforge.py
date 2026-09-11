import os
import re
import shutil
import pandas as pd
from pathlib import Path

def preprocessed_data_copier(base_dir, subject_ids):
    # Dictionary mapping task keys to their folder prefixes
    sym_project_dir="/data/users3/sbasodi1/symeditation_cabi_pilot_2023/"
    csv_files_dir=sym_project_dir+"metadata/"
    output_dir_path = sym_project_dir+"fmri/"

    task_prefixes = {
        "rest_1": "SYM_Rest1.csv",
        "rest_2": "SYMeditation_REST2.csv",
        "task_avs": "SYM_AVS.csv",
        "task_med": "SYM_taskMed.csv",
        "task_sym": "SYM_task_sym.csv", #video
    }

    for output_dir, csv_file_name in task_prefixes.items():
        copy_preproceesed_mri_files(base_dir, csv_files_dir+csv_file_name, output_dir_path+output_dir, subject_ids)


def copy_preproceesed_mri_files(base_dir, csv_file_path, output_dir_path, subject_ids):
    """Reads a CSV file containing file mapping columns, constructs the full path

    by combining (subject, session, series, datafile), and copies each file
    to the target destination folder.
    """
    csv_path = Path(csv_file_path)
    dest_base = Path(output_dir_path)

    # Ensure the target destination folder exists
    dest_base.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        print(f"Error: CSV file not found at {csv_file_path}")
        return

    print(f"Processing entries from: {csv_path.name}")
    # Read the CSV entirely into a DataFrame
    df = pd.read_csv(csv_path)
    df_cols = list(df.columns)

    # Basic validation to ensure the required columns are present
    required_columns = {"subject", "session", "series", "datafile"}
    if not required_columns.issubset(df_cols or []):
        missing = required_columns - set(df_cols or [])
        print(f"Error: Missing required columns in CSV: {missing}")
        return

    # Strip whitespace from string columns across the entire series vectors at once
    for col in ["subject", "session", "series", "datafile"]:
        df[col] = df[col].astype(str).str.strip()

    copied_count = 0
    failed_count = 0

    for index, row in df.iterrows():
        # Extract and clean spaces from column values
        subject = row["subject"].strip()
        session = row["session"].strip()
        series = row["series"].strip()
        datafile = row["datafile"].strip()

        # Skip incomplete rows dynamically
        if not (subject and session and series and datafile):
            print(f"  [Skipped] Incomplete row data: {row}")
            continue
        if not subject in subject_ids:
            print(f"  [Skipped] Subject not in subjectID : {subject}")
            failed_count += 1
            continue

        # Construct the full source file path: subject/session/series/datafile
        # Note: If these paths are relative, they will resolve relative to your script's working directory.
        # If they are absolute or need a base directory, prepend the base path here (e.g., base_dir / subject / ...)
        src_file_path = Path(base_dir) / subject / session / series / datafile

        # Build destination path keeping the original filename
        dest_file_name= f"{subject}_{datafile}"
        dest_file_path = dest_base / dest_file_name

        if not src_file_path.is_file():
            print(f"  [Not Found] Source file does not exist: {src_file_path}")
            failed_count += 1
            continue

        print(f"  --> Copying: {src_file_path}")
        print(f"      To:      {dest_file_path}")

        try:
            #shutil.copy2(src_file_path, dest_file_path) #TODO uncomment this line to perform actual copy preprocssed files
            copied_count += 1
        except Exception as e:
            print(
                f"  [Error] Failed to copy {src_file_path.name}. Reason: {e}"
            )
            failed_count += 1


    if not set(subject_ids).issubset(df["subject"].tolist()):
        missing = set(subject_ids) - set(df["subject"].tolist())
        print(f"Error: Missing file for subjects: {missing}")
        failed_count += len(missing)

    print("\n--- Process Complete ---")
    print(f"Successfully copied: {copied_count} files")
    print(f"Failed/Missing/Skipped:      {failed_count} files")



# --- Example Usage ---
if __name__ == "__main__":
    INPUT_DIRECTORY = "/data/brainforge/managed/cabi/vcalhoun/SYMeditation_H24183/prisma_fit/"
    SUBJECT_LIST = ["A00131600", "A00131606", "A00131898", "A00132008", "A00132075", "A00132127", "A00132202",
                    "A00132644", "A00132667", "A00132882", "A00133196", "A00133337", "A00134210", "A00132884",
                    "A00134766", "A00134785", "A00134786", "A00134805", "A00134855", "A00135232", "A00135233",
                    "A00135389"]
    preprocessed_data_copier(INPUT_DIRECTORY, SUBJECT_LIST)


