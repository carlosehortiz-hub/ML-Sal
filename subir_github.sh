#!/bin/bash

echo "🔍 Verificar estado do repositório..."
git status

echo ""
read -p "➡️ Mensagem do commit: " mensagem

if [ -z "$mensagem" ]; then
  echo "❌ Commit cancelado: mensagem vazia."
  exit 1
fi

echo ""
echo "➕ A adicionar ficheiros..."
git add .

echo ""
echo "📝 A criar commit..."
git commit -m "$mensagem"

echo ""
echo "🚀 A enviar para o GitHub..."
git push

echo ""
echo "✅ Submissão concluída com sucesso!"