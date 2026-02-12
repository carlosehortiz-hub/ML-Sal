#!/bin/bash

set -e

echo "🔍 Checking repository status..."
git status

echo ""
read -p "➡️ Commit message: " mensagem

if [ -z "$mensagem" ]; then
  echo "❌ Commit canceled: empty message."
  exit 1
fi

echo ""
read -p "➡️ Branch name: " branch

if [ -z "$branch" ]; then
  echo "❌ Branch name required."
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/$branch"; then
  echo "🔀 Switching to existing branch '$branch'..."
  git checkout "$branch"
else
  echo "🌿 Creating new branch '$branch'..."
  git checkout -b "$branch"
fi

echo ""
echo "➕ Adding files..."
git add .

echo ""
echo "📝 Creating commit..."
git commit -m "$mensagem"

echo ""
echo "🚀 Pushing to GitHub..."
git push -u origin "$branch"

echo ""
echo "✅ Push completed successfully on branch '$branch'!"
