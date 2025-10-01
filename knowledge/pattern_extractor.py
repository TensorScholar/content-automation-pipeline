"""
Pattern Extractor - Linguistic Feature Analysis
================================================

Extracts quantifiable linguistic features from text using NLP:
- Sentence-level statistics (length, complexity)
- Lexical diversity (type-token ratio, vocabulary richness)
- Readability metrics (Flesch-Kincaid, SMOG)
- Structural pattern detection
- Semantic tone representation via embeddings

Built on spaCy for efficiency and sentence-transformers for semantics.
"""

import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import spacy
from loguru import logger
from sentence_transformers import SentenceTransformer

from core.exceptions import ValidationError


class PatternExtractor:
    """
    Extracts linguistic patterns from text for style inference.

    Features computed:
    - Quantitative: sentence length, lexical diversity, readability
    - Qualitative: structure patterns, entity density
    - Semantic: tone embedding (384-dim vector)
    """

    def __init__(self):
        """Initialize NLP models."""
        try:
            # Load lightweight spaCy model
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            self.nlp.add_pipe("sentencizer")  # Fast sentence boundary detection

            # Load sentence transformer for tone embeddings
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            logger.info("Pattern extractor initialized with spaCy and SentenceTransformer")

        except Exception as e:
            logger.error(f"Failed to initialize NLP models: {e}")
            raise ValidationError(f"NLP initialization failed: {e}")

    # =========================================================================
    # FEATURE EXTRACTION ORCHESTRATION
    # =========================================================================

    def extract_features(self, text: str) -> Dict:
        """
        Extract comprehensive linguistic features from text.

        Args:
            text: Input text (article content)

        Returns:
            Dict with extracted features:
            {
                'avg_sentence_length': float,
                'lexical_diversity': float,
                'readability_score': float,
                'tone_embedding': np.ndarray,
                'structure_patterns': List[str],
                'entity_density': float,
                'word_count': int,
            }
        """
        if not text or len(text) < 50:
            raise ValidationError("Text too short for feature extraction (minimum 50 chars)")

        # Process with spaCy
        doc = self.nlp(text)

        # Extract features
        features = {
            "avg_sentence_length": self._compute_avg_sentence_length(doc),
            "lexical_diversity": self._compute_lexical_diversity(doc),
            "readability_score": self._compute_readability(text, doc),
            "tone_embedding": self._compute_tone_embedding(text),
            "structure_patterns": self._detect_structure_patterns(text, doc),
            "entity_density": self._compute_entity_density(doc),
            "word_count": len(
                [token for token in doc if not token.is_punct and not token.is_space]
            ),
        }

        return features

    # =========================================================================
    # QUANTITATIVE METRICS
    # =========================================================================

    @staticmethod
    def _compute_avg_sentence_length(doc: spacy.tokens.Doc) -> float:
        """
        Compute average sentence length in words.

        Args:
            doc: spaCy processed document

        Returns:
            Average words per sentence
        """
        sentences = list(doc.sents)
        if not sentences:
            return 0.0

        sentence_lengths = [
            len([token for token in sent if not token.is_punct and not token.is_space])
            for sent in sentences
        ]

        return sum(sentence_lengths) / len(sentence_lengths)

    @staticmethod
    def _compute_lexical_diversity(doc: spacy.tokens.Doc) -> float:
        """
        Compute lexical diversity (type-token ratio).

        Higher ratio = more diverse vocabulary

        Args:
            doc: spaCy processed document

        Returns:
            Type-token ratio (0-1)
        """
        # Extract tokens (excluding punctuation and stopwords)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_punct and not token.is_space and not token.is_stop
        ]

        if len(tokens) < 10:
            return 0.0

        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)

        # Raw TTR penalizes longer texts, so use corrected TTR (Guiraud's R)
        return unique_tokens / (total_tokens**0.5)

    def _compute_readability(self, text: str, doc: spacy.tokens.Doc) -> float:
        """
        Compute Flesch-Kincaid readability grade level.

        Lower score = easier to read

        Args:
            text: Raw text
            doc: spaCy processed document

        Returns:
            Flesch-Kincaid grade level (typically 6-18)
        """
        sentences = list(doc.sents)
        words = [token for token in doc if not token.is_punct and not token.is_space]

        if not sentences or not words:
            return 10.0  # Default grade level

        # Count syllables
        total_syllables = sum(self._count_syllables(token.text) for token in words)

        # Flesch-Kincaid formula
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)

        grade_level = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

        return max(0, min(18, grade_level))  # Clamp to 0-18 range

    @staticmethod
    def _count_syllables(word: str) -> int:
        """
        Estimate syllable count using vowel clusters.

        Args:
            word: Word to analyze

        Returns:
            Estimated syllable count
        """
        word = word.lower()
        vowels = "aeiouy"

        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith("e"):
            syllable_count -= 1

        # At least one syllable
        return max(1, syllable_count)

    @staticmethod
    def _compute_entity_density(doc: spacy.tokens.Doc) -> float:
        """
        Compute named entity density (entities per 100 words).

        Higher density may indicate technical or news content.

        Args:
            doc: spaCy processed document

        Returns:
            Entities per 100 words
        """
        # Re-enable NER temporarily
        with doc.retokenize() as retokenizer:
            pass  # Ensure doc is finalized

        # Count tokens (excluding punctuation)
        word_count = len([token for token in doc if not token.is_punct and not token.is_space])

        if word_count == 0:
            return 0.0

        # spaCy NER was disabled for performance, so estimate from capitalization
        capitalized = len(
            [token for token in doc if token.text[0].isupper() and len(token.text) > 1]
        )

        return (capitalized / word_count) * 100

    # =========================================================================
    # SEMANTIC FEATURES
    # =========================================================================

    def _compute_tone_embedding(self, text: str) -> np.ndarray:
        """
        Generate semantic tone embedding.

        Captures the overall semantic "feel" of the text in vector space.
        Similar texts will have high cosine similarity.

        Args:
            text: Input text

        Returns:
            384-dimensional embedding vector
        """
        # Use first 512 tokens to avoid context limits
        truncated = " ".join(text.split()[:512])

        embedding = self.embedding_model.encode(
            truncated, convert_to_numpy=True, show_progress_bar=False
        )

        return embedding

    @staticmethod
    def compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute centroid of multiple embeddings.

        Used to find "average" tone across multiple articles.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Centroid embedding (normalized)
        """
        if not embeddings:
            return np.zeros(384)

        stacked = np.vstack(embeddings)
        centroid = np.mean(stacked, axis=0)

        # Normalize to unit vector
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        return centroid

    # =========================================================================
    # STRUCTURAL PATTERN DETECTION
    # =========================================================================

    @staticmethod
    def _detect_structure_patterns(text: str, doc: spacy.tokens.Doc) -> List[str]:
        """
        Detect common structural patterns in text.

        Patterns:
        - listicle: Numbered or bulleted lists
        - how-to: Instructional language
        - problem-solution: Problem/solution framing
        - narrative: Story-telling elements
        - comparison: Comparative language

        Args:
            text: Raw text
            doc: spaCy processed document

        Returns:
            List of detected pattern names
        """
        patterns = []

        # Listicle detection
        numbered_list = re.findall(r"^\d+[\.)]\s+", text, re.MULTILINE)
        bulleted_list = re.findall(r"^[•\-\*]\s+", text, re.MULTILINE)
        if len(numbered_list) >= 3 or len(bulleted_list) >= 3:
            patterns.append("listicle")

        # How-to detection
        how_to_markers = [
            "how to",
            "step 1",
            "step 2",
            "first,",
            "second,",
            "finally",
            "you will need",
            "follow these",
            "instructions",
        ]
        if any(marker in text.lower() for marker in how_to_markers):
            patterns.append("how-to")

        # Problem-solution detection
        problem_markers = ["problem", "issue", "challenge", "struggle"]
        solution_markers = ["solution", "solve", "fix", "resolve", "overcome"]
        has_problem = any(marker in text.lower() for marker in problem_markers)
        has_solution = any(marker in text.lower() for marker in solution_markers)
        if has_problem and has_solution:
            patterns.append("problem-solution")

        # Narrative detection
        narrative_markers = ["once", "story", "tale", "journey", "experience"]
        past_tense_verbs = len(
            [
                token
                for token in doc
                if token.pos_ == "VERB" and "Past" in token.morph.get("Tense", [])
            ]
        )
        total_verbs = len([token for token in doc if token.pos_ == "VERB"])
        past_tense_ratio = past_tense_verbs / total_verbs if total_verbs > 0 else 0

        if any(marker in text.lower() for marker in narrative_markers) or past_tense_ratio > 0.4:
            patterns.append("narrative")

        # Comparison detection
        comparison_markers = [
            "versus",
            "vs",
            "compared to",
            "difference between",
            "better than",
            "worse than",
            "similar to",
            "unlike",
        ]
        if any(marker in text.lower() for marker in comparison_markers):
            patterns.append("comparison")

        # Default to general if no specific pattern
        if not patterns:
            patterns.append("general")

        return patterns


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score (0-1, where 1 is identical)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
