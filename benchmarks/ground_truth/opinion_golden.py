"""
Adversarial golden benchmark — opinion-document questions that expose RAG weakness.

IMPORTANT: This file must be HAND-AUTHORED by a human reading the actual
source documents. Do NOT let an LLM fill in golden_answer or match_strings —
the entire point is human-verified ground truth.

Workflow:
  1. Use the Review tab in the UI (KG subgraph + source document viewer)
  2. Read the actual opinion text for each case
  3. Write the golden answer based on what the document says
  4. Add match_strings that would verify a correct answer
  5. Set is_negative=True for questions where the correct answer is "no" / "not found"

Question categories that expose RAG weakness:
  - Multi-hop citation chains (requires traversing the KG)
  - Negative-existence queries (requires knowing what the KG does NOT contain)
  - Cross-document citation accuracy (requires linking facts across cases)
  - Holding vs. dicta distinctions (requires structured extraction)
  - Citation verification (binary: is this citation real and accurately characterized?)
  - Jurisdictional applicability

Demo cluster: Gideon v. Wainwright, Loving v. Virginia,
Marbury v. Madison, Miranda v. Arizona, Brown v. Board of Education,
Roe v. Wade, Mapp v. Ohio, Powell v. Alabama, Betts v. Brady.

Format: (question, golden_answer, match_strings, is_negative)
"""

# Populate by running:
#   python -m crystal.ui  →  Review tab  →  author golden answers with source docs open
#
# Or add entries directly here in this format:
#   ("What court decided X v. Y?", "Supreme Court of ...", ["supreme court"], False),

OPINION_GOLDEN_CASES: list[tuple[str, str, list[str], bool]] = [
    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-HOP CITATION CHAINS
    # These require following citation edges through the KG.
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════
    # CROSS-DOCUMENT CITATION ACCURACY
    # Require linking structured facts across multiple cases.
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════
    # HOLDING vs. DICTA / REASONING QUESTIONS
    # Require distinguishing the actual holding from supporting reasoning.
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════
    # NEGATIVE EXISTENCE QUERIES
    # Crystal should correctly state it lacks data for these.
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════
    # CITATION VERIFICATION (binary correctness)
    # Is this citation real and accurately characterized?
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════
    # JURISDICTIONAL / PROCEDURAL QUESTIONS
    # Require understanding court structure and case metadata.
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-HOP REASONING — COMPARING ACROSS CASES
    # Require synthesizing facts from multiple KG nodes.
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════
    # DEEP FACT RETRIEVAL — SPECIFIC DETAILS
    # Require precise KG fact lookup, not vague LLM knowledge.
    # ═══════════════════════════════════════════════════════════════════════
]
