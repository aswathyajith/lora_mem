#!/bin/sh

#SBATCH --partition=clab,general
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aswathy@uchicago.edu
#SBATCH --nodelist=j001-ds,j002-ds,j003-ds,j004-ds

./scripts/env_setup.sh
echo "Env setup done"
pwd
start_time=$(date)
start_timestamp=$(date -d "$start_time" +%s)


if [ -z "${SCRIPT_PATH}" ]; then
    echo "No script provided to run. Exiting."
    exit 1
fi

if [ -z "${DOMAIN}" ]; then
    echo "Domain not provided. Exiting."
    exit 1
fi

if [ -z "${STDOUT_PATH}" ]; then
    echo "STDOUT_PATH is not exported"
    exit 1
else
    # Replace %x and %j in STDOUT_PATH with the actual values
    STDOUT_PATH=`echo "$STDOUT_PATH" | sed -e "s/%j/$SLURM_JOB_ID/g" -e "s/%x/$SLURM_JOB_NAME/g"`

    SLURM_JOB_OUTPUT_DIR=`dirname "$STDOUT_PATH"`
    echo $SLURM_JOB_OUTPUT_DIR
    # check if the directory exists
    if [ ! -d "$SLURM_JOB_OUTPUT_DIR" ]; then
        echo "${SLURM_JOB_OUTPUT_DIR} does not exist"
        exit 1
    fi
fi

STDOUT_PATH="$SLURM_JOB_OUTPUT_DIR"
echo $SCRIPT_PATH $DOMAIN
./${SCRIPT_PATH}

end_time=$(date)
end_timestamp=$(date -d "$end_time" +%s)
elapsed_time=$((end_timestamp - start_timestamp))

# Convert seconds to hours, minutes, seconds
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

echo "JOB_ID: $SLURM_JOB_ID"
echo "START TIME: $(date)"
echo "END TIME: $(date)"