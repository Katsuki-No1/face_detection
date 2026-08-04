#!/usr/bin/env bash
set -euo pipefail

project_root="/home/katsuki/projects/face_detection"
deim_config="$project_root/configs/dinov3_deimv2/dinov3_deimv2_l_vits16_face_pose_drive_v2_960_full.yaml"
deim_output="$project_root/outputs/experiments/dinov3_deimv2_l_vits16_face_pose_drive_v2_balanced_811_960_full"
yolo_config="$project_root/configs/yolo/yolo26_pose/face_pose_drive_v2_full.yaml"
queue_root="$project_root/outputs/training_queues/face_pose_drive_v2_balanced_811_deimv2_l_yolo26m"

mkdir -p "$queue_root"
cd "$project_root"

run_stage() {
  local stage_name="$1"
  shift
  printf '[%s] START %s\n' "$(date --iso-8601=seconds)" "$stage_name" | tee -a "$queue_root/queue_status.log"
  set +e
  "$@" 2>&1 | tee -a "$queue_root/${stage_name}.log"
  local stage_status=${PIPESTATUS[0]}
  set -e
  printf '[%s] END %s exit=%d\n' "$(date --iso-8601=seconds)" "$stage_name" "$stage_status" | tee -a "$queue_root/queue_status.log"
  return "$stage_status"
}

deim_command=(
  "$project_root/.venv/bin/python"
  "$project_root/tools/train_dinov3_deimv2.py"
  --config "$deim_config"
  --output-dir "$deim_output"
  --device cuda:0
  --wandb
  --wandb-entity sato-waseda-university
  --wandb-project face_detection
  --wandb-run-name dinov3_deimv2_l_vits16_face_pose_drive_v2_balanced_811_unfreeze_seed0
  --wandb-group face_pose_drive_v2_balanced_811_deimv2_l_yolo26m
  --wandb-mode online
  --dataset-version face_pose_drive_v2_balanced_811
)

if [[ -f "$deim_output/checkpoints/last.pt" && ! -f "$deim_output/summary.md" ]]; then
  deim_command+=(--resume-from "$deim_output/checkpoints/last.pt")
fi

if [[ -f "$deim_output/summary.md" ]]; then
  printf '[%s] SKIP deimv2_l: completed output exists\n' "$(date --iso-8601=seconds)" | tee -a "$queue_root/queue_status.log"
else
  run_stage deimv2_l "${deim_command[@]}"
fi

yolo_output="$project_root/runs/yolo/yolo26_pose/face_pose_drive_v2_balanced_811/yolo26m_pose_full"
yolo_command=(
  "$project_root/.venv/bin/python"
  "$project_root/scripts_YOLO/yolo26_pose/run_yolo26_pose.py"
  --config "$yolo_config"
  --stage train
  --models m
  --device 0
  --skip-weight-download
)

if [[ -f "$yolo_output/weights/last.pt" && ! -f "$yolo_output/yolo26_experiment_metadata.json" ]]; then
  yolo_command+=(--resume)
fi

if [[ -f "$yolo_output/yolo26_experiment_metadata.json" ]]; then
  printf '[%s] SKIP yolo26m: completed output exists\n' "$(date --iso-8601=seconds)" | tee -a "$queue_root/queue_status.log"
else
  run_stage yolo26m "${yolo_command[@]}"
fi

printf '[%s] QUEUE COMPLETE\n' "$(date --iso-8601=seconds)" | tee -a "$queue_root/queue_status.log"
