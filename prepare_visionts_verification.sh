#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-VisionTS-verification}"
BASE_COMMIT="${VISIONTS_BASE_COMMIT:-7f34731cea92f6d670f1f5835f2018267d1b6135}"
PATCH_FILE="${SCRIPT_DIR}/patches/0001-Add-VisionTS-verification-experiments.patch"

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "Patch file not found: ${PATCH_FILE}" >&2
    exit 2
fi

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
    git clone https://github.com/Keytoyze/VisionTS.git "${TARGET_DIR}"
fi

cd "${TARGET_DIR}"

if [[ -f "long_term_tsf/scripts/vision_verification/run_verification_dataset.sh" ]]; then
    echo "VisionTS verification patch already appears to be applied in ${TARGET_DIR}."
else
    git fetch origin "${BASE_COMMIT}" || true
    git checkout -B vision-verification "${BASE_COMMIT}"
    git am --3way "${PATCH_FILE}"
fi

cat <<'MSG'

Prepared VisionTS verification checkout.

Next steps:
  1. Install dependencies for VisionTS / long_term_tsf on the target server.
  2. Put MAE checkpoint under ckpt/ and datasets under long_term_tsf/dataset/.
  3. Run:

       cd long_term_tsf
       CUDA_VISIBLE_DEVICES=0 bash scripts/vision_verification/run_ettm1_quick.sh

  4. Full suite:

       CUDA_VISIBLE_DEVICES=0 bash scripts/vision_verification/run_official_ltsf_suite.sh

MSG
