# 🏠 ImmoLex — Plateforme RAG Franco-Finlandaise

> Système de questions-réponses intelligent sur la réglementation immobilière entre la France et la Finlande, basé sur une architecture RAG (Retrieval Augmented Generation).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange)

---

## 🎯 Problème résolu

Des milliers d'expatriés français en Finlande (et inversement) naviguent seuls des questions complexes : fiscalité croisée, droits des locataires, processus d'achat, aides disponibles. **Aucun outil n'existait** pour comparer et expliquer intelligemment ces deux systèmes en parallèle, avec sources officielles citées.

## ✨ Fonctionnalités

- **RAG sur textes officiels** : indexation de sources gouvernementales (service-public.fr, vero.fi, kela.fi...)
- **Réponses sourcées** : chaque réponse cite ses sources avec liens vers les textes officiels
- **Couverture cross-border** : convention fiscale France-Finlande, droits des expatriés UE
- **Interface bilingue** : réponses en français, sources en FR/EN/FI
- **Historique de conversation** : contexte maintenu sur la session

## 🏗️ Architecture

```
Question utilisateur
        ↓
ChromaDB (vector store)
  → Embedding de la question (OpenAI text-embedding-3-small)
  → Recherche par similarité cosinus
  → Top-3 documents pertinents récupérés
        ↓
GPT-4o-mini
  → Contexte = documents récupérés
  → Génération de réponse sourcée
        ↓
Réponse avec citations + liens officiels
```

## 📚 Base de connaissances

| Catégorie | Sources |
|-----------|---------|
| PTZ & Financement | service-public.fr |
| Fiscalité non-résidents | impots.gouv.fr |
| Achat immobilier Finlande | maanmittauslaitos.fi |
| Aides logement finlandaises | kela.fi |
| Convention fiscale FR-FI | vero.fi + impots.gouv.fr |
| DPE & réglementation | service-public.fr |
| Droits locataires Finlande | ymparisto.fi |
| Copropriété non-résidents | service-public.fr |

## 🚀 Installation & Lancement

```bash
# 1. Cloner le repo
git clone https://github.com/[votre-username]/immolex-rag
cd immolex-rag

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé API
export OPENAI_API_KEY="votre-clé-ici"

# 4. Lancer l'application
streamlit run app.py
```

## 💡 Exemples de questions

- *"En tant que résident finlandais, puis-je bénéficier du PTZ français ?"*
- *"Comment fonctionne la fiscalité si je loue mon appartement français depuis la Finlande ?"*
- *"Quelle est la taxe de transfert immobilier en Finlande ?"*
- *"Comment éviter la double imposition France-Finlande sur les revenus locatifs ?"*

## 🔮 Améliorations prévues

- [ ] Ajout de documents PDF uploadables par l'utilisateur
- [ ] Support multilingue (finnois, anglais)
- [ ] Intégration API Légifrance pour mise à jour automatique des textes
- [ ] Système de feedback utilisateur pour améliorer la pertinence
- [ ] Déploiement sur Hugging Face Spaces

## ⚠️ Disclaimer

ImmoLex est un outil d'information. Pour toute décision juridique ou fiscale importante, consultez un notaire, avocat ou conseiller fiscal agréé.

---

*Projet réalisé dans le cadre d'un portfolio Data/IA — Tampere, Finlande 2026*
