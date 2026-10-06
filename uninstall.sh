#!/bin/sh
# Remove installation artifacts while preserving shared credentials and user data.
set -e
echo "=== ImageAnalyzer Uninstall ==="
echo "Keeping shared OCR_BROKER_SECRET for VoiceAnalyzer and other clients."
echo "To delete ONLY the shared key, run: $(dirname "$0")/delete-shared-secret.sh"
echo "Doing so requires reconfiguring both analyzers and their clients together."
node "$(dirname "$0")/scripts/uninstall.cjs"
