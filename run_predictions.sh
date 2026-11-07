#!/bin/bash
# Predict the two revision datasets with the selected models (UBELIX_instructions.md steps 6 and 7).
# Usage (in the PsyNamic folder):
#   bash run_predictions.sh          # update the Webapp, link + check the models, submit the prediction job
#   bash run_predictions.sh check    # once the job is gone from `squeue --me`: check that all 3 steps finished
#   bash run_predictions.sh pack     # pack the predictions and copy them to the research storage
set -eo pipefail
cd "$(dirname "$0")"

MODE=${1:-submit}
DEST=/storage/research/dcr_mds/psynamic_review
WEBAPP=PsyNamic-Webapp
MANUAL=data/analysis/as_review_studies_relevant_unannotated_20240101_00-00-00.csv
PUBMED=data/analysis/pubmed_results_19000101_20261031_00-03-23.csv
OUTPUTS="data/predictions/*_20240101_* data/predictions/*_20261031_* data/relevant_studies/studies_20261031_*"

source ubelix.env

case "$MODE" in
submit)
    # 6.1 Latest Webapp (model_paths.json with the selected models) and link the trained models
    git -C $WEBAPP fetch
    git -C $WEBAPP checkout origin/main
    ln -sfn "$(realpath ${OUTPUT_DIR})" $WEBAPP/pipeline/models
    cd $WEBAPP
    for f in $MANUAL $PUBMED; do [ -f "$f" ] || { echo "Input file missing: $f"; exit 1; }; done
    python3 - <<'EOF'
import json, os, sys
models = json.load(open('pipeline/model_paths.json'))
missing = [m['model_path'] for m in models if not os.path.isdir(os.path.join('pipeline', m['model_path']))]
print(f'{len(models)} models,', 'MISSING: ' + ', '.join(missing) if missing else 'all found')
sys.exit(1 if missing or len(models) != 19 else 0)
EOF
    if squeue --me -h -n PredictPsynamic | grep -q .; then
        echo "A PredictPsynamic job is already queued or running, not submitting a second one:"; squeue --me -n PredictPsynamic; exit 1
    fi

    # Submit (finished steps are skipped by the pipeline, so resubmitting after a failure or time limit is safe)
    mkdir -p log
    sbatch <<EOF
#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=PredictPsynamic
#SBATCH --output=log/predict_%j.out

module load Anaconda3
eval "\$(conda shell.bash hook)"
conda activate psynamic_env
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda); torch.ones(1, device='cuda')" || exit 1
export HF_HUB_OFFLINE=1   # everything is local

python -m pipeline.predict --log-to-stdout --batch-size 16 --keep-all-labels --prediction-mode classification -i $MANUAL
python -m pipeline.predict --log-to-stdout --batch-size 16 --prediction-mode ner -i $MANUAL
python -m pipeline.predict --log-to-stdout --batch-size 16 --keep-all-labels --prediction-mode all -i $PUBMED
EOF
    echo "Submitted. Check with: squeue --me   (expected: under 1 hour once running). When it is gone: bash run_predictions.sh check"
    ;;

check)
    # The pipeline logs errors instead of failing, so sacct can say COMPLETED although a step failed
    cd $WEBAPP
    LOG=$(ls -t log/predict_*.out | head -1)
    echo "Log: $LOG"
    DONE=$(grep -c "Prediction process completed successfully" "$LOG" || true)
    echo "Completed steps: $DONE (must be 3)"
    echo "Errors in log (must be none):"; grep -E "ERROR|Error|Traceback" "$LOG" || echo "  none"
    echo "Output files (must be 5):"; ls -lh $OUTPUTS || true
    if [ "$DONE" = 3 ] && ! grep -qE "ERROR|Error|Traceback" "$LOG" && [ "$(ls $OUTPUTS 2>/dev/null | wc -l)" = 5 ]; then
        echo "OK. Next: bash run_predictions.sh pack"
    else
        echo "NOT OK. If a step failed: delete only that step's output file(s), then run 'bash run_predictions.sh' again"
        echo "(finished steps are skipped). If it fails again, send Vera $LOG."
        exit 1
    fi
    ;;

pack)
    # Pack and copy to the research storage
    cd $WEBAPP
    mkdir -p "$DEST"
    [ -w "$DEST" ] || { echo "No write access to $DEST"; exit 1; }
    tar czf psynamic_predictions.tar.gz $OUTPUTS log/predict_*.out
    rsync -ah --partial --progress psynamic_predictions.tar.gz "$DEST/"
    echo "Done. Sizes must match:"
    ls -lh psynamic_predictions.tar.gz "$DEST/psynamic_predictions.tar.gz"
    ;;

*)
    echo "Usage: bash run_predictions.sh [check|pack]"; exit 1
    ;;
esac
