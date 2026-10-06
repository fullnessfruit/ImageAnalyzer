#!/bin/sh
# Explicitly delete only the shared authentication key, not services or data.
set -e
echo "This key is shared by ImageAnalyzer, VoiceAnalyzer, and the OCR broker."
node "$(dirname "$0")/scripts/shared-secret.cjs" delete
