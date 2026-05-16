#!/bin/bash
#SBATCH --job-name=tent-r26
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=results/tent_replication/slurm_%j.log
#SBATCH --error=results/tent_replication/slurm_%j.log

# Train R-26 (pycls ResNet, depth=26, basic_transform) on CIFAR-10 and
# CIFAR-100 with the canonical recipe, then evaluate TENT/Norm/Source on
# CIFAR-10/100-C and aggregate against the paper's Table 2 numbers.
#
# Submit:
#   sbatch scripts/tent_replication_R26.sbatch
# Resubmit (after a partial / failed run): the eval step is idempotent
# (skips configs whose output dir already has 75 cells), so re-running is
# safe. Training is NOT skipped on a re-run; if you only want to re-eval,
# pass SKIP_TRAIN=1:
#   sbatch --export=ALL,SKIP_TRAIN=1 scripts/tent_replication_R26.sbatch

set -euo pipefail
cd /leonardo_work/IscrC_AIM/ProtoAbsorb

export PATH=/leonardo_work/IscrC_AIM/ProtoAbsorb/.venv/bin:$PATH
export SSL_CERT_FILE=/etc/pki/tls/cert.pem

echo "=== GPU info ==="
nvidia-smi
echo "=== Slurm env ==="
echo "JobID=$SLURM_JOB_ID Node=$SLURMD_NODENAME GPUs=$SLURM_GPUS_ON_NODE"

# Ensure CUDA 12.1 torch (Leonardo driver is 535 / CUDA 12.2; cu130 wheels run on CPU)
have_cu121=$(.venv/bin/python -c "import torch; print('+cu121' in torch.__version__)" 2>/dev/null || echo "False")
if [ "$have_cu121" != "True" ]; then
  uv pip install --index-url https://download.pytorch.org/whl/cu121 'torch<2.6' 'torchvision<0.21'
fi

echo "=== Python / torch sanity ==="
.venv/bin/python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'n_gpus=', torch.cuda.device_count(), 'dev0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA must be available on a GPU node'"

mkdir -p ckpt/local results/tent_replication

# -----------------------------------------------------------------------------
# Train R-26 source models (DDP across 4 GPUs)
# -----------------------------------------------------------------------------
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  for ds in cifar10 cifar100; do
    out=ckpt/local/R26_${ds}.pt
    if [ -f "$out" ]; then
      echo "=== SKIP train R-26 $ds (checkpoint exists: $out) ==="
      continue
    fi
    echo "=== Train R-26 $ds (4x DDP, 200 epochs, ~10 min) ==="
    .venv/bin/python -m torch.distributed.run \
        --standalone --nproc-per-node=4 \
        experiments/tent_replication/train_R26.py \
        --dataset "$ds" \
        --data-root data \
        --out "$out" \
        --epochs 200 \
        --batch-size 128 \
        --lr 0.1 \
        --weight-decay 5e-4 \
        --num-workers 4 \
        --precise-bn-samples 1024 \
        --seed 1
  done
else
  echo "=== SKIP_TRAIN=1 set, skipping training ==="
fi

# -----------------------------------------------------------------------------
# Evaluate TENT / Norm / Source on CIFAR-10-C and CIFAR-100-C with R-26
# -----------------------------------------------------------------------------
run_or_skip() {
  local cfg_path="$1"
  local out_dir="$2"
  local n=$(ls "$out_dir"/*.json 2>/dev/null | wc -l)
  if [ "$n" = "75" ]; then
    echo "SKIP  $cfg_path (75 cells already present in $out_dir)"
  else
    echo "RUN   $cfg_path → $out_dir (currently $n cells)"
    .venv/bin/python experiments/tent_replication/run_cifar_c.py \
      --cfg "$cfg_path" --out "$out_dir"
  fi
}

echo "=== Eval R-26 on CIFAR-10-C ==="
for cfg in source norm tent; do
  run_or_skip experiments/tent_replication/cfgs/${cfg}_R26_cifar10.yaml \
              results/tent_replication/R26_cifar10c/${cfg}
done

echo "=== Eval R-26 on CIFAR-100-C ==="
for cfg in source norm tent; do
  run_or_skip experiments/tent_replication/cfgs/${cfg}_R26_cifar100.yaml \
              results/tent_replication/R26_cifar100c/${cfg}
done

echo "=== Aggregate (R-26) + diff vs paper Table 2 ==="
.venv/bin/python experiments/tent_replication/aggregate.py \
    --cells \
        results/tent_replication/R26_cifar10c/source \
        results/tent_replication/R26_cifar10c/norm \
        results/tent_replication/R26_cifar10c/tent \
        results/tent_replication/R26_cifar100c/source \
        results/tent_replication/R26_cifar100c/norm \
        results/tent_replication/R26_cifar100c/tent \
    --expected experiments/tent_replication/expected.yaml \
    --severity 5 \
    --out docs/tent_replication/reproduction_report_R26.md \
  || echo "(aggregate exit code: $?)"

echo "=== Done — see docs/tent_replication/reproduction_report_R26.md ==="
