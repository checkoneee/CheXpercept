#!/bin/bash
# Clone third-party dependencies that are not bundled in the repo.
#
# Stage 00 (ROSALIA) needs LISA.
# Stage 01 needs SAM3 (mask deformation) and CheXmask-U (anatomy masks).
#
# Run once after cloning this repository. Conda envs and api_info/api_keys.yaml
# must be set up separately (see README).
#
# Usage:
#   bash setup.sh                    # clones whatever is missing
#   bash setup.sh --skip-lisa        # skip if you only need sample_test.sh
#   bash setup.sh --skip-chexmask-u  # skip if you only need sample_test.sh

set -euo pipefail
cd "$(dirname "$0")"

SKIP_LISA=0
SKIP_SAM3=0
SKIP_CHEXMASK_U=0
for arg in "$@"; do
    case "$arg" in
        --skip-lisa)        SKIP_LISA=1 ;;
        --skip-sam3)        SKIP_SAM3=1 ;;
        --skip-chexmask-u)  SKIP_CHEXMASK_U=1 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

clone_if_missing() {
    local url="$1" dest="$2"
    if [ -d "$dest/.git" ] || [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "[ok] $dest already exists"
    else
        echo "[clone] $url -> $dest"
        git clone --depth 1 "$url" "$dest"
    fi
}

[ "$SKIP_LISA" -eq 0 ] && clone_if_missing \
    https://github.com/JIA-Lab-research/LISA.git \
    src/00_source_data_curation/LISA

[ "$SKIP_SAM3" -eq 0 ] && clone_if_missing \
    https://github.com/facebookresearch/sam3.git \
    src/01_mask_deformation/sam3

[ "$SKIP_CHEXMASK_U" -eq 0 ] && clone_if_missing \
    https://github.com/mcosarinsky/CheXmask-U.git \
    src/01_mask_deformation/CheXmask-U

echo
echo "setup.sh complete."
echo "Next: create conda envs (envs/*.yml), copy api_info/api_keys_example.yaml"
echo "      to api_info/api_keys.yaml, and edit cfg/config.yaml as needed."
