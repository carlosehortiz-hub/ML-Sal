#!/bin/bash

echo "🔍 Checking repository status..."
git status

echo ""
read -p "➡️ Commit message: " mensagem

if [ -z "$mensagem" ]; then
  echo "❌ Commit canceled: empty message."
  exit 1
fi

echo ""
echo "➕ Adding files..."
git add .

echo ""
echo "📝 Creating commit..."
git commit -m "$mensagem"

echo ""
echo "🚀 Pushing to GitHub..."
git push

echo ""
echo "✅ Push completed successfully!"
