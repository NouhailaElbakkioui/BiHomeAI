import os
import streamlit as st
from google import genai
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import requests
from bs4 import BeautifulSoup
import json
import hashlib
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BiHomeAI — Réglementation Franco-Finlandaise",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── GEMINI EMBEDDING FUNCTION ────────────────────────────────────────────────
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        api_key = st.session_state.get("gemini_key", GEMINI_API_KEY)
        client = genai.Client(api_key=api_key)
        embeddings = []
        for text in input:
            result = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

# ─── CHROMA DB SETUP ──────────────────────────────────────────────────────────
@st.cache_resource
def get_chroma_client():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    ef = GeminiEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(
        name="immo_lex_gemini",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# ─── KNOWLEDGE BASE ───────────────────────────────────────────────────────────
# Sources officielles publiques — librement accessibles
KNOWLEDGE_BASE = [
    {
        "id": "fr_ptz_001",
        "country": "France",
        "category": "Financement",
        "title": "Prêt à Taux Zéro (PTZ) — Conditions d'éligibilité",
        "content": """Le Prêt à Taux Zéro (PTZ) est réservé aux primo-accédants en France. 
        Pour en bénéficier, il faut ne pas avoir été propriétaire de sa résidence principale 
        durant les 2 dernières années. Les non-résidents fiscaux français ne sont généralement 
        pas éligibles au PTZ. Un expatrié vivant en Finlande mais achetant en France doit 
        vérifier son statut de résident fiscal. Le PTZ peut financer jusqu'à 40% du prix 
        d'achat selon la zone géographique (A, B1, B2, C). En 2024-2025, le PTZ est étendu 
        à l'ancien sous conditions de travaux dans les zones B2 et C.""",
        "source": "service-public.fr",
        "url": "https://www.service-public.fr/particuliers/vosdroits/F10871",
        "lang": "fr"
    },
    {
        "id": "fr_taxe_nr_001",
        "country": "France",
        "category": "Fiscalité",
        "title": "Fiscalité immobilière pour non-résidents français",
        "content": """Un non-résident fiscal français qui vend un bien immobilier en France 
        est soumis à une taxe sur la plus-value immobilière de 19% (plus prélèvements sociaux 
        de 17,2% s'il est ressortissant UE/EEE). Pour les résidents en Finlande (UE), 
        la convention fiscale franco-finlandaise s'applique : la France a le droit d'imposer 
        les plus-values immobilières sur les biens situés en France. 
        L'abattement pour durée de détention s'applique : exonération totale après 22 ans 
        pour l'IR et 30 ans pour les prélèvements sociaux. 
        Les revenus locatifs d'un non-résident sont imposés en France au taux minimum de 20%.""",
        "source": "impots.gouv.fr",
        "url": "https://www.impots.gouv.fr/international-particulier/questions/je-suis-non-resident-et-je-possede-un-bien-immobilier-en-france",
        "lang": "fr"
    },
    {
        "id": "fi_purchase_001",
        "country": "Finland",
        "category": "Achat immobilier",
        "title": "Processus d'achat immobilier en Finlande pour étrangers",
        "content": """En Finlande, les ressortissants de l'UE/EEE peuvent acheter librement 
        des biens immobiliers sans restriction. Le processus comprend : 
        1) Offre d'achat (tarjous) signée par les deux parties, 
        2) Signature du contrat de vente (kauppakirja) devant un notaire ou directement, 
        3) Paiement de la taxe de transfert (varainsiirtovero) : 2% pour les logements 
        en copropriété (asunto-osakeyhtiö) et 4% pour les maisons individuelles, 
        4) Enregistrement auprès du cadastre (maanmittauslaitos). 
        Les expatriés français en Finlande bénéficient des mêmes droits que les citoyens finlandais 
        pour l'achat immobilier grâce aux règles UE.""",
        "source": "maanmittauslaitos.fi",
        "url": "https://www.maanmittauslaitos.fi/en/real-property/buying-and-owning",
        "lang": "en"
    },
    {
        "id": "fi_arava_001",
        "country": "Finland",
        "category": "Aides au logement",
        "title": "Aides au logement finlandaises (asumistuki)",
        "content": """En Finlande, l'aide au logement générale (yleinen asumistuki) est gérée 
        par Kela. Elle est accessible aux résidents en Finlande quelle que soit leur nationalité, 
        y compris les expatriés français. Le montant dépend des revenus, de la composition 
        du foyer et du loyer. Pour l'achat immobilier, l'ASP (asuntosäästöpalkkiojärjestelmä) 
        est un compte épargne logement finlandais similaire au PEL français, 
        réservé aux 15-44 ans. Les intérêts sont bonifiés par l'État finlandais.
        En 2024, la bonification d'intérêt ASP est de 3,8% supplémentaires.""",
        "source": "kela.fi",
        "url": "https://www.kela.fi/asumistuki",
        "lang": "fi"
    },
    {
        "id": "fr_fi_convention_001",
        "country": "Franco-Finlandais",
        "category": "Fiscalité croisée",
        "title": "Convention fiscale France-Finlande — Immobilier",
        "content": """La convention fiscale entre la France et la Finlande (signée en 1970, 
        révisée) prévoit que les revenus immobiliers sont imposables dans le pays où est 
        situé le bien. Ainsi, un résident finlandais possédant un appartement en France 
        paiera l'impôt sur les loyers en France. Pour éviter la double imposition, 
        la Finlande accorde un crédit d'impôt équivalent à l'impôt payé en France. 
        La résidence fiscale est déterminée par le lieu de séjour habituel (plus de 183 jours). 
        Un expatrié français travaillant en Finlande est considéré résident fiscal finlandais 
        s'il y passe plus de 6 mois par an.""",
        "source": "vero.fi / impots.gouv.fr",
        "url": "https://www.vero.fi/en/individuals/tax-card-and-tax-return/arriving-in-finland/",
        "lang": "fr"
    },
    {
        "id": "fr_dpe_001",
        "country": "France",
        "category": "Réglementation",
        "title": "DPE et obligations énergétiques — Impact sur l'achat",
        "content": """Le Diagnostic de Performance Énergétique (DPE) est obligatoire pour 
        toute vente en France depuis 2006. Depuis 2022, les logements classés G sont 
        considérés comme des passoires thermiques. À partir de 2025, les logements G 
        ne peuvent plus être mis en location. Les acheteurs non-résidents (ex: expatriés 
        en Finlande) doivent être particulièrement vigilants car un logement mal classé 
        représente un risque locatif. Le DPE doit être fourni avant la signature du 
        compromis de vente. Un DPE défavorable peut être un levier de négociation 
        du prix d'achat.""",
        "source": "service-public.fr",
        "url": "https://www.service-public.fr/particuliers/vosdroits/F16096",
        "lang": "fr"
    },
    {
        "id": "fi_tenant_rights_001",
        "country": "Finland",
        "category": "Droits des locataires",
        "title": "Droits des locataires en Finlande",
        "content": """En Finlande, la loi sur les locations résidentielles (laki asuinhuoneiston 
        vuokrauksesta, 481/1995) protège fortement les locataires. Le préavis minimum 
        pour un locataire est d'1 mois, pour un propriétaire de 3 à 6 mois selon la durée 
        du bail. Les loyers ne sont pas plafonnés en Finlande (contrairement à certaines 
        villes françaises avec l'encadrement des loyers). Le dépôt de garantie est limité 
        à 3 mois de loyer. Le chauffage est généralement inclus dans les charges en Finlande, 
        ce qui constitue une différence majeure avec la France où il est souvent séparé.""",
        "source": "ymparisto.fi",
        "url": "https://www.ymparisto.fi/en/housing/renting",
        "lang": "en"
    },
    {
        "id": "fr_syndic_001",
        "country": "France",
        "category": "Copropriété",
        "title": "Règles de copropriété française pour non-résidents",
        "content": """En France, un copropriétaire non-résident (vivant en Finlande par exemple) 
        a les mêmes droits et obligations qu'un résident. Il peut voter aux assemblées générales 
        par procuration ou en visioconférence depuis la loi ELAN 2018. 
        Les charges de copropriété sont dues même en cas d'absence. 
        Un non-résident doit désigner un représentant fiscal en France si la valeur 
        de ses biens immobiliers dépasse 150 000€. Le fonds de travaux (loi ALUR) 
        est obligatoire : 5% minimum des charges annuelles sont provisionnées pour 
        les futurs travaux.""",
        "source": "service-public.fr",
        "url": "https://www.service-public.fr/particuliers/vosdroits/F2589",
        "lang": "fr"
    }
]

# ─── INDEXING ─────────────────────────────────────────────────────────────────
def index_knowledge_base(collection):
    """Index documents if not already done."""
    existing = collection.count()
    if existing >= len(KNOWLEDGE_BASE):
        return
    
    for doc in KNOWLEDGE_BASE:
        doc_id = doc["id"]
        try:
            collection.get(ids=[doc_id])
        except:
            collection.add(
                ids=[doc_id],
                documents=[doc["content"]],
                metadatas=[{
                    "country": doc["country"],
                    "category": doc["category"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "url": doc["url"],
                    "lang": doc["lang"]
                }]
            )

# ─── RAG QUERY ────────────────────────────────────────────────────────────────
def query_rag(question: str, collection, n_results: int = 3):
    """Retrieve relevant documents and generate answer."""
    api_key = st.session_state.get("gemini_key", GEMINI_API_KEY)
    client = genai.Client(api_key=api_key)

    # Embed the question
    q_result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=question
    )
    q_embedding = q_result.embeddings[0].values

    # Retrieve top docs
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Build context
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        context_parts.append(
            f"[SOURCE {i+1}] {meta['title']} ({meta['country']}, {meta['category']})\n"
            f"URL: {meta['url']}\n"
            f"{doc}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Generate with Gemini
    prompt = f"""Tu es BiHomeAI, un assistant expert en réglementation immobilière franco-finlandaise.
Tu aides les expatriés français en Finlande et les Finlandais en France.

INSTRUCTIONS:
- Réponds toujours en français, clairement et de manière structurée
- Cite toujours tes sources en mentionnant [SOURCE X] dans ta réponse
- Si une question concerne les deux pays, compare-les explicitement
- Signale si une information nécessite une vérification auprès d'un professionnel
- Si tu n'as pas l'information, dis-le clairement

Question: {question}

Contexte documentaire:
{context}

Réponds à la question en t'appuyant sur les sources. Cite [SOURCE X]. Conclus par les points clés."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text, metas

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=DM+Sans:wght@300;400;500&display=swap');
    
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-family: 'DM Sans', sans-serif;
        font-weight: 300;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #f8f9ff;
        border-left: 3px solid #4A5CDB;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .country-badge-fr {
        background: #003189;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .country-badge-fi {
        background: #003580;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .country-badge-cross {
        background: #5a2d8c;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .answer-box {
        background: white;
        border: 1px solid #e8eaff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        line-height: 1.7;
    }
    .stTextInput > div > div > input {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
    }
    .example-question {
        cursor: pointer;
        padding: 0.5rem 0.8rem;
        background: #f0f2ff;
        border-radius: 8px;
        margin: 0.3rem 0;
        font-size: 0.88rem;
        border: 1px solid #dde0ff;
        color: #3a4db5;
        transition: background 0.2s;
    }
    .example-question:hover {
        background: #e0e4ff;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="main-header">🏠 BiHomeAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Réglementation immobilière franco-finlandaise · Powered by RAG + GPT-4</p>', unsafe_allow_html=True)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("🇫🇷 **France** ↔ 🇫🇮 **Finlande**")

st.markdown("---")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key_input = st.text_input("Gemini API Key", type="password",
                                   value=st.session_state.get("gemini_key", ""),
                                   help="Clé API Google Gemini — gratuite sur aistudio.google.com")
    if api_key_input:
        st.session_state["gemini_key"] = api_key_input
        pass
    
    st.markdown("---")
    st.markdown("### 📚 Base de connaissances")
    st.markdown(f"**{len(KNOWLEDGE_BASE)} documents** indexés")
    
    categories = list(set(doc["category"] for doc in KNOWLEDGE_BASE))
    for cat in categories:
        count = sum(1 for d in KNOWLEDGE_BASE if d["category"] == cat)
        st.markdown(f"• {cat} ({count})")
    
    st.markdown("---")
    st.markdown("### 🔍 Questions exemples")
    
    examples = [
        "En tant que résident finlandais, puis-je bénéficier du PTZ français ?",
        "Quelle est la différence entre acheter en France vs Finlande ?",
        "Comment fonctionne la fiscalité si je loue mon appart français depuis la Finlande ?",
        "Quels sont mes droits comme locataire en Finlande ?",
        "Comment éviter la double imposition France-Finlande ?",
    ]
    
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["question_input"] = ex

# ─── INIT SESSION STATE ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# ─── MAIN INTERFACE ───────────────────────────────────────────────────────────
col_main, col_sources = st.columns([2, 1])

with col_main:
    st.markdown("### 💬 Posez votre question")
    
    question = st.text_input(
        "Question",
        value=st.session_state.question_input,
        placeholder="Ex: Puis-je acheter en Finlande sans être citoyen finlandais ?",
        label_visibility="collapsed",
        key="main_question"
    )
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        ask_btn = st.button("🔍 Analyser", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Effacer l'historique", use_container_width=False):
            st.session_state.messages = []
            st.rerun()

    # Process question
    if ask_btn and question:
        if not st.session_state.get("gemini_key", GEMINI_API_KEY):
            st.error("⚠️ Veuillez entrer votre clé API Gemini dans la barre latérale.")
        else:
            with st.spinner("🔍 Recherche dans la base documentaire..."):
                try:
                    collection = get_chroma_client()
                    index_knowledge_base(collection)
                    answer, sources = query_rag(question, collection)
                    
                    st.session_state.messages.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    })
                    st.session_state.question_input = ""
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
    
    # Display conversation history
    for msg in reversed(st.session_state.messages):
        with st.container():
            st.markdown(f"**❓ {msg['question']}**")
            st.markdown(f'<div class="answer-box">{msg["answer"]}</div>', 
                       unsafe_allow_html=True)
            
            # Sources
            st.markdown("**📖 Sources consultées :**")
            for i, src in enumerate(msg["sources"]):
                country = src.get("country", "")
                badge_class = "country-badge-fr" if country == "France" else \
                             "country-badge-fi" if country == "Finland" else \
                             "country-badge-cross"
                
                st.markdown(
                    f'<div class="source-card">'
                    f'<span class="{badge_class}">{country}</span> '
                    f'<strong>{src["title"]}</strong><br>'
                    f'<small>Source: {src["source"]} · '
                    f'<a href="{src["url"]}" target="_blank">Consulter le texte officiel →</a></small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown("---")

with col_sources:
    st.markdown("### 📊 Couverture documentaire")
    
    countries = {}
    for doc in KNOWLEDGE_BASE:
        c = doc["country"]
        countries[c] = countries.get(c, 0) + 1
    
    for country, count in countries.items():
        flag = "🇫🇷" if country == "France" else "🇫🇮" if country == "Finland" else "🔀"
        st.metric(f"{flag} {country}", f"{count} docs")
    
    st.markdown("---")
    st.markdown("### 🏷️ Catégories")
    
    cats = {}
    for doc in KNOWLEDGE_BASE:
        c = doc["category"]
        cats[c] = cats.get(c, 0) + 1
    
    for cat, count in cats.items():
        st.markdown(f"**{cat}** — {count} source{'s' if count > 1 else ''}")
    
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    **BiHomeAI** est une plateforme RAG 
    (*Retrieval Augmented Generation*) 
    qui indexe des textes réglementaires 
    officiels et génère des réponses 
    sourcées.
    
    **Stack technique :**
    - LangChain-style RAG
    - ChromaDB (vector store)
    - OpenAI Embeddings
    - GPT-4o-mini
    - Streamlit
    
    ⚠️ *Pour toute décision importante, 
    consultez un notaire ou un avocat.*
    """)