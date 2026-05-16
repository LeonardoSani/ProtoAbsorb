#!/bin/bash
#SBATCH --job-name=tent-replication-p1
#SBATCH --account=iscrc_aim
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=results/tent_replication/slurm_%j.log
#SBATCH --error=results/tent_replication/slurm_%j.log

set -euo pipefail

cd /leonardo_work/IscrC_AIM/ProtoAbsorb

# Use the project venv
export PATH=/leonardo_work/IscrC_AIM/ProtoAbsorb/.venv/bin:$PATH
export SSL_CERT_FILE=/etc/pki/tls/cert.pem

echo "=== GPU info ==="
nvidia-smi

echo "=== Ensure CUDA 12.1 torch (Leonardo driver is 535/CUDA 12.2; cu130 wheels run on CPU) ==="
have_cu121=$(.venv/bin/python -c "import torch; print('+cu121' in torch.__version__)" 2>/dev/null || echo "False")
if [ "$have_cu121" != "True" ]; then
  uv pip install --index-url https://download.pytorch.org/whl/cu121 'torch<2.6' 'torchvision<0.21'
fi

echo "=== Python / torch sanity ==="
.venv/bin/python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'dev=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA must be available on a GPU node'"

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

# --- T15: CIFAR-10-C ---
echo "=== T15: CIFAR-10-C ==="
for cfg in source norm tent; do
  run_or_skip experiments/tent_replication/cfgs/${cfg}.yaml \
              results/tent_replication/cifar10c/${cfg}
done

# --- T16: CIFAR-100-C — source first as the arch sanity check ---
echo "=== T16: CIFAR-100-C ==="
for cfg in source norm tent; do
  run_or_skip experiments/tent_replication/cfgs/${cfg}_cifar100.yaml \
              results/tent_replication/cifar100c/${cfg}
done

# --- T17: aggregate + diff vs paper ---
echo "=== T17: aggregate ==="
.venv/bin/python experiments/tent_replication/aggregate.py \
    --cells \
        results/tent_replication/cifar10c/source \
        results/tent_replication/cifar10c/norm \
        results/tent_replication/cifar10c/tent \
        results/tent_replication/cifar100c/source \
        results/tent_replication/cifar100c/norm \
        results/tent_replication/cifar100c/tent \
    --expected experiments/tent_replication/expected.yaml \
    --severity 5 \
    --out docs/tent_replication/reproduction_report.md

echo "=== Done — aggregate.py exit code was $? ==="
