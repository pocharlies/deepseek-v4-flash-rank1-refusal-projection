#!/usr/bin/env bash
# Publish the model card + direction vectors to the Hugging Face Hub,
# then cross-link the GitHub README back to it.
#
#   hf auth login                # once, needs a token with WRITE scope
#   ./hf/upload.sh               # repo name defaults to <your-user>/deepseek-v4-flash-0731-uncensored-abliterated-refusal-directions
#   ./hf/upload.sh myorg/my-name # or pass an explicit repo id
#
set -euo pipefail

cd "$(dirname "$0")/.."
GH_REPO="pocharlies/deepseek-v4-flash-rank1-refusal-projection"

command -v hf >/dev/null || { echo "hf CLI not found: pip install -U huggingface_hub"; exit 1; }

# `hf auth whoami` prints e.g. "user=alice orgs=acme" (and may emit a "Logged in"
# banner first). Pull the username out rather than trusting line order.
WHOAMI=$(hf auth whoami 2>/dev/null || true)
WHO=$(printf '%s\n' "$WHOAMI" | grep -oE 'user=[A-Za-z0-9._-]+' | head -1 | cut -d= -f2)
if [ -z "$WHO" ]; then
  # older CLIs print the bare username on its own line
  WHO=$(printf '%s\n' "$WHOAMI" | grep -vE '^(✓|Logged in|orgs:|$)' | head -1 | tr -d '[:space:]')
fi
if [ -z "$WHO" ]; then
  echo "Not logged in. Run:  hf auth login   (token needs WRITE scope)"
  exit 1
fi

REPO_ID="${1:-$WHO/deepseek-v4-flash-0731-uncensored-abliterated-refusal-directions}"
echo "==> publishing to https://huggingface.co/$REPO_ID"

# `hf repo create` is deprecated in favour of `hf repos create`; the old form also had
# a `-y` flag that no longer exists. Try the new spelling first, fall back to the old.
hf repos create "$REPO_ID" --type model --public 2>/dev/null \
  || hf repo create "$REPO_ID" --repo-type model 2>/dev/null \
  || echo "    (repo already exists, updating)"

# The xet uploader needs a writable cache. A root-owned ~/.cache/huggingface/{hub,xet}
# — common when models were first pulled by a root process — makes uploads die with
# "OSError: I/O error: Permission denied (os error 13)". Redirect rather than chown.
if [ ! -w "${HF_XET_CACHE:-$HOME/.cache/huggingface/xet}" ] 2>/dev/null; then
  export HF_XET_CACHE="${TMPDIR:-/tmp}/hf-xet-cache-$(id -u)"
  mkdir -p "$HF_XET_CACHE"
  echo "    (xet cache not writable; using $HF_XET_CACHE)"
fi

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
