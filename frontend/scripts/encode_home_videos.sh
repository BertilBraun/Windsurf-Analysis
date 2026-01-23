#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  encode_home_videos.sh -i INPUT_PATH [-o OUTPUT_DIR] [--mp4-crf N] [--av1-crf N] [--mp4-preset NAME] [--av1-preset N]

Defaults:
  OUTPUT_DIR: script_dir/../public
  mp4-crf: 26
  av1-crf: 38
  mp4-preset: slow
  av1-preset: 8
EOF
}

# Defaults
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_PATH=""
OUTPUT_DIR="${SCRIPT_DIR}/../public"
MP4_CRF=26
AV1_CRF=38
MP4_PRESET="slow"
AV1_PRESET=8

# Args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      INPUT_PATH="${2:-}"; shift 2;;
    -o|--output-dir)
      OUTPUT_DIR="${2:-}"; shift 2;;
    --mp4-crf)
      MP4_CRF="${2:-}"; shift 2;;
    --av1-crf)
      AV1_CRF="${2:-}"; shift 2;;
    --mp4-preset)
      MP4_PRESET="${2:-}"; shift 2;;
    --av1-preset)
      AV1_PRESET="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2;;
  esac
done

# Validate
if [[ -z "$INPUT_PATH" ]]; then
  echo "Error: input is required (-i/--input)" >&2
  usage
  exit 2
fi

command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg not found in PATH." >&2; exit 1; }

[[ -d "$OUTPUT_DIR" ]] || { echo "Error: Output directory does not exist: $OUTPUT_DIR" >&2; exit 1; }
[[ -f "$INPUT_PATH" ]] || { echo "Error: Input file not found: $INPUT_PATH" >&2; exit 1; }

base_name="$(basename -- "$INPUT_PATH")"
base_name="${base_name%.*}"

mp4_output="${OUTPUT_DIR}/${base_name}.encoded.mp4"
av1_output="${OUTPUT_DIR}/${base_name}.av1.mp4"

echo "Encoding MP4 for ${base_name}..."
ffmpeg -y -i "$INPUT_PATH" -an -c:v libx264 -preset "$MP4_PRESET" -crf "$MP4_CRF" -movflags +faststart "$mp4_output"

echo "Encoding AV1 for ${base_name}..."
ffmpeg -y -i "$INPUT_PATH" -an -c:v libsvtav1 -preset "$AV1_PRESET" -crf "$AV1_CRF" -pix_fmt yuv420p -movflags +faststart "$av1_output"

echo "Wrote $mp4_output"
echo "Wrote $av1_output"
