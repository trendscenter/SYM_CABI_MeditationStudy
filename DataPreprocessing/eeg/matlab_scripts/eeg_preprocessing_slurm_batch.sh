#!/bin/bash
#SBATCH --job-name=SYM_EEG_preprocess
#SBATCH --output=../data/preprocessed_eeg/new_logs/SYM_eeg_out_%A_%a.log
#SBATCH --error=../data/preprocessed_eeg/new_logs/logs/SYM_eeg_err_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6          # MATLAB uses multithreading for FFTs
#SBATCH --mem=16G                  # 5000Hz data is large; 16GB is safe
#SBATCH --time=24:00:00            # Adjust based on file length
#SBATCH --partition=qTRDGPU        # Change to your cluster's partition name
#SBATCH --array=1-23              # If you have 20 files
#SBATCH --mail-type=ALL # types of emails to send out. See SLURM documentation for more possible values
#SBATCH --mail-user=sbasodi1@gsu.edu

# it is a good practice to add small delay at the beginning and end of the job- helps to preserve stability of SLURM controller when large number of jobs fail simultaneously 
sleep 10s

# for debugging purpose- in case the job fails, you know where to look for possible cause
echo $HOSTNAME >&2


# Load MATLAB module (name depends on your cluster)
module load matlab/R2023a

# Run MATLAB without GUI and pass the SLURM_ARRAY_TASK_ID as the argument
# matlab -nodisplay -nosplash -r "eeg_preprocess([${SLURM_ARRAY_TASK_ID}])"
matlab -batch "eeg_preprocessing_until_ica([${SLURM_ARRAY_TASK_ID}])"

# delay at the end (good practice)
sleep 10s
