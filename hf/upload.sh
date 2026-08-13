#!/usr/bin/env bash
# Publish the model card + direction vectors to the Hugging Face Hub,
# then cross-link the GitHub README back to it.
#
#   hf auth login                # once, needs a token with WRITE scope
#   ./hf/upload.sh               # repo name defaults to <your-user>/deepseek-v4-flash-0731-refusal-directions
#   ./hf/upload.sh myorg/my-name # or pass an explicit repo id
#
set -euo pipefail

cd "$(dirname "$0")/.."
GH_REPO="pocharlies/deepseek-v4-flash-rank1-refusal-projection"

command -v hf >/dev/null || { echo "hf CLI not found: pip install -U huggingface_hub"; exit 1; }

WHO=$(hf auth whoami 2>/dev/null | head -1 || true)
if [ -z "$WHO" ] || [ "$WHO" = "Not logged in" ]; then
  echo "Not logged in. Run:  hf auth login   (token needs WRITE scope)"
  exit 1
fi

REPO_ID="${1:-$WHO/deepseek-v4-flash-0731-refusal-directions}"
echo "==> publishing to https://huggingface.co/$REPO_ID"

hf repo create "$REPO_ID" --repo-type model -y 2>/dev/null || echo "    (repo already exists, updating)"
hf upload "$REPO_ID" ./hf . --repo-type model --exclude "upload.sh" \
  --commit-message "Runtime rank-1 refusal projection: directions + measured A/B results"

echo "==> published: https://huggingface.co/$REPO_ID"

# Cross-link GitHub -> Hugging Face, at the anchor left in README.md
if grep -q "<!-- HF_LINK_ANCHOR -->" README.md; then
  sed -i "s|<!-- HF_LINK_ANCHOR -->|Published at [\`$REPO_ID\`](https://huggingface.co/$REPO_ID).|" README.md
  git add README.md
  git commit -q -m "Link Hugging Face repo $REPO_ID"
  git push -q origin main
  echo "==> GitHub README now links to https://huggingface.co/$REPO_ID"
else
  echo "==> anchor already replaced; add the link to README.md by hand if needed"
fi

echo
echo "Cross-linked:"
echo "  https://huggingface.co/$REPO_ID"
echo "  https://github.com/$GH_REPO"
