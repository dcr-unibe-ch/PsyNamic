#!/bin/bash

source ubelix.env

case ${BILLING} in
    gratis) SLURM_BILLING="#SBATCH --account=gratis
#SBATCH --partition=gpu" ;;
    preemptable) SLURM_BILLING="#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable" ;;
    paygo) [ -z "${PAYGO_WCKEY}" ] && { echo "Set PAYGO_WCKEY in ubelix.env (see swckeys)"; exit 1; }
        SLURM_BILLING="#SBATCH --account=paygo
#SBATCH --wckey=${PAYGO_WCKEY}
#SBATCH --partition=gpu" ;;
    *) echo "BILLING in ubelix.env must be gratis, preemptable or paygo"; exit 1 ;;
esac
mkdir -p ${OUTPUT_DIR}

# Create the environment (first time only) and download the models, in case the compute nodes have no internet
module load Anaconda3
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q psynamic_env; then
    conda create -y -n psynamic_env -c conda-forge python=${PYTHON_VERSION}
    conda activate psynamic_env
    pip install torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL}
    pip install -r requirements.txt
fi
conda activate psynamic_env
python -c "import evaluate; from transformers import AutoTokenizer, AutoModel; [(AutoTokenizer.from_pretrained(m), AutoModel.from_pretrained(m)) for m in ['microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', 'allenai/scibert_scivocab_uncased', 'dmis-lab/biobert-v1.1', 'emilyalsentzer/Bio_ClinicalBERT', 'michiyasunaga/BioLinkBERT-base']]; evaluate.load('seqeval')"

# List of models
models=(
    "biomedbert-abstract"
    "scibert"
    "biobert"
    "clinicalbert"
    "biolinkbert"
)

# List of tasks
tasks=(
    "Data Collection"
    "Data Type"
    "Number of Participants"
    "Age of Participants"
    "Application Form"
    "Clinical Trial Phase"
    "Condition"
    "Outcomes"
    "Regimen"
    "Setting"
    "Study Control"
    "Study Purpose"
    "Substance Naivety"
    "Substances"
    "Sex of Participants"
    "Study Conclusion"
    "Study Type"
    "NER Bio"
    "Relevant"
)

# Directory to store the generated scripts
mkdir -p training_scripts
mkdir -p log

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        # Convert the task name to a filename-friendly format
        task_filename=$(echo "$task" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')

        # Skip runs that already finished
        if ls ${OUTPUT_DIR}/${model}_${task_filename}_*/test_predictions.csv > /dev/null 2>&1; then
            continue
        fi

        # Create the training script for the task and model
        cat <<EOT > training_scripts/train_${model}_${task_filename}.sh
#!/bin/bash

#SBATCH --time=${TIME}   ## days-hours:minutes:seconds
${SLURM_BILLING}
#SBATCH --gres=gpu:${GPU}:1
#SBATCH --mem=${MEM}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=TrainPsynamic_${model}_${task_filename} ## job name
#SBATCH --output=log/psynamic_${model}_${task_filename}.out  ## standard out file

module load Anaconda3
eval "\$(conda shell.bash hook)"
conda activate psynamic_env
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda); torch.ones(1, device='cuda')" || exit 1
export EXPERIMENT_PATH=${OUTPUT_DIR}
python model/model.py --model ${model} --data ${DATA_DIR}/${task_filename} --task '${task}' --early_stopping_patience 5 --epoch 30
EOT

        # Make the generated script executable
        chmod +x training_scripts/train_${model}_${task_filename}.sh

        # Submit the job
        sbatch training_scripts/train_${model}_${task_filename}.sh

    done
done
