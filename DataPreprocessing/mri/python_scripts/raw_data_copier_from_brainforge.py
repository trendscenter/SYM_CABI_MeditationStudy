import os
import re
import shutil
import pandas as pd
from pathlib import Path


def copy_raw_task_fMRI_data(input_dir, subject_ids, output_dir):
    """Parses subject directories for specific task prefixes, finds the folder

    with the highest trailing numeric suffix, and copies the .nii files to
    task-specific folders inside the output directory.
    """
    input_base = Path(input_dir)
    output_base = Path(output_dir)

    # Dictionary mapping task keys to their folder prefixes
    task_prefixes = {
        "rest_1": "rest_open_epigre_20ch_tr2s_mb1_run01_",
        "rest_2": "rest_open_epigre_20ch_tr2s_mb1_run02_",
        "task_avs": "task_avs_epigre_20ch_tr2s_mb1_",
        "task_med": "task_med_epigre_20ch_tr2s_mb1_",
        "task_sym": "task_sym_epigre_20ch_tr2s_mb1_",
    }

    # Pre-create the 5 task directories in the output folder
    for task_key in task_prefixes.keys():
        task_output_dir = Path(os.path.join(output_base , task_key))
        task_output_dir.mkdir(parents=True, exist_ok=True)

    for subj_id in subject_ids:
        subj_dir = Path(os.path.join(input_base , subj_id))
        if not subj_dir.is_dir():
            print(f"Warning: Subject directory not found: {subj_dir}")
            continue

        print(f"Processing subject: {subj_id}")

        # Find the intermediate "*_v1" folder inside the subject directory
        v1_folders = list(subj_dir.glob("*_v1"))
        if not v1_folders:
               raise Exception(f"  [Warning] No *_v1 folder found inside {subj_id}. Skipping.")

        assert len(v1_folders)==1
        v1_dir = v1_folders[0]
        # List all contents of this intermediate directory
        try:
            all_contents = list(v1_dir.iterdir())
        except PermissionError:
            raise Exception(f"Error: Permission denied for directory {v1_dir}")


        # Process each task prefix
        for task_key, prefix in task_prefixes.items():
            matching_folders = []

            # Find all directories that match this task's prefix
            for item in all_contents:
                if item.is_dir() and item.name.startswith(prefix):
                    # Find trailing numbers at the very end of the folder name
                    match = re.search(r"(\d+)$", item.name)
                    if match:
                        suffix_num = int(match.group(1))
                        matching_folders.append((suffix_num, item))
                    else:
                        matching_folders.append((0, item))

            if not matching_folders:
                raise Exception (f"Missing task folder file for subject: {subj_id}, task: {task_key}\n {all_contents}")

            # Identify the directory with the maximum suffix value
            max_suffix, target_folder = max(matching_folders, key=lambda x: x[0])
            if len(matching_folders) >1:
                print(f"\n\nselecting folder with suffix: {max_suffix} of all {matching_folders}\n\n")
            # Grab any .nii or .nii.gz files inside that folder
            nii_files = list(target_folder.glob("*.nii*"))

            if not nii_files:
                print(
                    f"  [No NII Found] {target_folder.name} in {subj_id}"
                )
                break;

            # Copy files to the corresponding task folder
            task_dest_dir = Path(os.path.join(output_base , task_key))

            for nii_file in nii_files:
                # Format: {subjectID}_{actual_nifit_filename}
                dest_filename = f"{subj_id}_{nii_file.name}"
                dest_path = Path(os.path.join(task_dest_dir , dest_filename))

                try:
                    print(f"  --> Copying t0: {dest_path}")
                    #shutil.copy2(nii_file, dest_path) #TODO uncomment this line to perform actual copy
                except Exception as e:
                    print(
                        f"  [Error] Failed to copy {nii_file.name}. Reason: {e}"
                    )



def raw_data_copier():
    OUTPUT_DIRECTORY = "/data/users3/sbasodi1/symeditation_cabi_pilot_2023/fmri"


    copy_raw_task_fMRI_data(
        input_dir=INPUT_DIRECTORY,
        subject_ids=SUBJECT_LIST,
        output_dir=OUTPUT_DIRECTORY,
    )

# --- Example Usage ---
if __name__ == "__main__":
    INPUT_DIRECTORY = "/data/brainforge/managed/cabi/vcalhoun/SYMeditation_H24183/prisma_fit/"
    SUBJECT_LIST = ["A00131600", "A00131606", "A00131898", "A00132008", "A00132075", "A00132127", "A00132202",
                    "A00132644", "A00132667", "A00132882", "A00133196", "A00133337", "A00134210", "A00132884",
                    "A00134766", "A00134785", "A00134786", "A00134805", "A00134855", "A00135232", "A00135233",
                    "A00135389"]
    #raw_data_copier()


