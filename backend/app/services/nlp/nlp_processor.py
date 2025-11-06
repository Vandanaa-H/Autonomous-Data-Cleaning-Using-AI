import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import re

# NLP imports
try:
    from sentence_transformers import SentenceTransformer
    import spacy
    from symspellpy import SymSpell, Verbosity
    from sklearn.metrics.pairwise import cosine_similarity
    ADVANCED_NLP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced NLP libraries not available: {e}")
    ADVANCED_NLP_AVAILABLE = False

from app.models.schemas import DetectedIssue

logger = logging.getLogger(__name__)


class NLPProcessor:
    """Advanced NLP processor for text data cleaning and analysis"""

    def __init__(self):
        self.embedding_model = None
        self.spacy_model = None
        self.spell_checker = None
        self.initialized = False

        if ADVANCED_NLP_AVAILABLE:
            try:
                self._initialize_models()
                self.initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize NLP models: {e}")
                self.initialized = False

    def _initialize_models(self):
        """Initialize NLP models"""

        # Initialize sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")

        # Initialize spaCy model
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            # Try to download it
            try:
                import subprocess
                subprocess.run(
                    ["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self.spacy_model = spacy.load("en_core_web_sm")
                logger.info("Downloaded and loaded spaCy model")
            except Exception as e2:
                logger.error(f"Failed to download spaCy model: {e2}")

        # Initialize spell checker
        try:
            self.spell_checker = SymSpell(
                max_dictionary_edit_distance=2, prefix_length=7)
            # Load dictionary (using built-in frequency dictionary)
            dictionary_path = Path(__file__).parent / \
                "data" / "frequency_dictionary_en_82_765.txt"
            if not dictionary_path.exists():
                # Create basic dictionary if file doesn't exist
                self._create_basic_dictionary()
            else:
                self.spell_checker.load_dictionary(
                    str(dictionary_path), term_index=0, count_index=1)
            logger.info("Spell checker initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize spell checker: {e}")

    def _create_basic_dictionary(self):
        """Create a basic spell check dictionary from common words"""

        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
            "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "she", "or", "an",
            "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about",
            "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him",
            "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after",
            "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
            "any", "these", "give", "day", "most", "us", "data", "value", "number", "text", "column", "row"
        ]

        for word in common_words:
            self.spell_checker.create_dictionary_entry(word, 1)

    def detect_semantic_duplicates(self, df: pd.DataFrame, text_columns: List[str] = None, threshold: float = 0.85) -> List[DetectedIssue]:
        """Detect semantic duplicates using sentence embeddings"""

        issues = []

        if not self.initialized or not self.embedding_model:
            logger.warning(
                "Embedding model not available, skipping semantic duplicate detection")
            return issues

        if text_columns is None:
            text_columns = df.select_dtypes(
                include=['object']).columns.tolist()

        for column in text_columns:
            if column not in df.columns:
                continue

            try:
                # Get non-null text values
                text_values = df[column].dropna().astype(str).tolist()

                if len(text_values) < 2:
                    continue

                # Generate embeddings
                embeddings = self.embedding_model.encode(text_values)

                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(embeddings)

                # Find semantic duplicates
                semantic_duplicates = 0
                duplicate_pairs = []

                for i in range(len(similarity_matrix)):
                    for j in range(i + 1, len(similarity_matrix)):
                        if similarity_matrix[i][j] > threshold:
                            semantic_duplicates += 1
                            duplicate_pairs.append(
                                (text_values[i], text_values[j], similarity_matrix[i][j]))

                if semantic_duplicates > 0:
                    issue = DetectedIssue(
                        issue_type="semantic_duplicates",
                        column=column,
                        description=f"Column '{column}' has {semantic_duplicates} semantic duplicate pairs (similarity > {threshold})",
                        severity="medium" if semantic_duplicates > 5 else "low",
                        affected_rows=semantic_duplicates
                    )
                    issues.append(issue)

                    logger.info(
                        f"Found {semantic_duplicates} semantic duplicates in column '{column}'")

            except Exception as e:
                logger.error(
                    f"Error detecting semantic duplicates in column '{column}': {e}")

        return issues

    def detect_text_anomalies(self, df: pd.DataFrame, text_columns: List[str] = None) -> List[DetectedIssue]:
        """Detect text anomalies using NLP"""

        issues = []

        if text_columns is None:
            text_columns = df.select_dtypes(
                include=['object']).columns.tolist()

        for column in text_columns:
            if column not in df.columns:
                continue

            # Detect spelling errors
            spelling_issues = self._detect_spelling_errors(df[column], column)
            issues.extend(spelling_issues)

            # Detect entity inconsistencies using spaCy
            if self.spacy_model:
                entity_issues = self._detect_entity_inconsistencies(
                    df[column], column)
                issues.extend(entity_issues)

            # Detect encoding issues
            encoding_issues = self._detect_encoding_issues(df[column], column)
            issues.extend(encoding_issues)

        return issues

    def _detect_spelling_errors(self, series: pd.Series, column_name: str) -> List[DetectedIssue]:
        """Detect spelling errors in text column"""

        issues = []

        if not self.spell_checker:
            return issues

        try:
            spelling_errors = 0
            total_words = 0

            for text in series.dropna().astype(str):
                words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
                total_words += len(words)

                for word in words:
                    if len(word) > 2:  # Skip very short words
                        suggestions = self.spell_checker.lookup(
                            word, Verbosity.CLOSEST, max_edit_distance=2)
                        if not suggestions or suggestions[0].term != word:
                            spelling_errors += 1

            if spelling_errors > 0 and total_words > 0:
                error_rate = (spelling_errors / total_words) * 100

                if error_rate > 5:  # More than 5% spelling errors
                    issue = DetectedIssue(
                        issue_type="spelling_errors",
                        column=column_name,
                        description=f"Column '{column_name}' has {spelling_errors} spelling errors ({error_rate:.1f}% error rate)",
                        severity="medium" if error_rate > 10 else "low",
                        affected_rows=spelling_errors
                    )
                    issues.append(issue)

        except Exception as e:
            logger.error(
                f"Error detecting spelling errors in column '{column_name}': {e}")

        return issues

    def _detect_entity_inconsistencies(self, series: pd.Series, column_name: str) -> List[DetectedIssue]:
        """Detect entity inconsistencies using spaCy NER"""

        issues = []

        if not self.spacy_model:
            return issues

        try:
            entity_types = {}

            # Analyze a sample of texts for entity types
            sample_texts = series.dropna().astype(str).head(100).tolist()

            for text in sample_texts:
                doc = self.spacy_model(text)
                for ent in doc.ents:
                    if ent.label_ not in entity_types:
                        entity_types[ent.label_] = []
                    entity_types[ent.label_].append(ent.text)

            # Check for inconsistent entity formats
            for entity_type, entities in entity_types.items():
                if len(set(entities)) != len(entities):  # Has duplicates
                    unique_entities = len(set(entities))
                    total_entities = len(entities)
                    inconsistencies = total_entities - unique_entities

                    if inconsistencies > 0:
                        issue = DetectedIssue(
                            issue_type="entity_inconsistencies",
                            column=column_name,
                            description=f"Column '{column_name}' has inconsistent {entity_type} entities ({inconsistencies} variations)",
                            severity="low",
                            affected_rows=inconsistencies
                        )
                        issues.append(issue)

        except Exception as e:
            logger.error(
                f"Error detecting entity inconsistencies in column '{column_name}': {e}")

        return issues

    def _detect_encoding_issues(self, series: pd.Series, column_name: str) -> List[DetectedIssue]:
        """Detect text encoding issues"""

        issues = []

        try:
            encoding_problems = 0

            for text in series.dropna().astype(str):
                # Check for common encoding issues
                if any(char in text for char in ['�', 'Ã¢', 'Ã¡', 'Ã©', 'â€™', 'â€œ', 'â€�']):
                    encoding_problems += 1

                # Check for mixed encodings
                try:
                    text.encode('ascii')
                except UnicodeEncodeError:
                    # Has non-ASCII characters, which might be encoding issues if unexpected
                    # These shouldn't have special chars
                    if column_name.lower() in ['id', 'code', 'number']:
                        encoding_problems += 1

            if encoding_problems > 0:
                issue = DetectedIssue(
                    issue_type="encoding_issues",
                    column=column_name,
                    description=f"Column '{column_name}' has {encoding_problems} potential encoding issues",
                    severity="medium" if encoding_problems > 10 else "low",
                    affected_rows=encoding_problems
                )
                issues.append(issue)

        except Exception as e:
            logger.error(
                f"Error detecting encoding issues in column '{column_name}': {e}")

        return issues

    def correct_text(self, text: str, correct_spelling: bool = True, normalize_entities: bool = True) -> str:
        """Correct text using available NLP tools"""

        if not isinstance(text, str) or not text.strip():
            return text

        corrected_text = text

        # Spell correction
        if correct_spelling and self.spell_checker:
            try:
                words = re.findall(r'\b[a-zA-Z]+\b', corrected_text)
                for word in words:
                    if len(word) > 2:
                        suggestions = self.spell_checker.lookup(
                            word, Verbosity.CLOSEST, max_edit_distance=2)
                        if suggestions and suggestions[0].term != word.lower():
                            # Replace with suggestion if confidence is high
                            if suggestions[0].distance <= 1:
                                corrected_text = corrected_text.replace(
                                    word, suggestions[0].term)
            except Exception as e:
                logger.error(f"Error correcting spelling: {e}")

        # Entity normalization using spaCy
        if normalize_entities and self.spacy_model:
            try:
                doc = self.spacy_model(corrected_text)
                # This is a placeholder for more sophisticated entity normalization
                # In practice, you'd have lookup tables or rules for standardizing entities
                pass
            except Exception as e:
                logger.error(f"Error normalizing entities: {e}")

        return corrected_text

    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get sentence embeddings for texts"""

        if not self.embedding_model:
            logger.warning("Embedding model not available")
            return np.array([])

        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])

    def cluster_similar_texts(self, texts: List[str], threshold: float = 0.8) -> Dict[int, List[str]]:
        """Cluster similar texts using embeddings"""

        if not self.embedding_model:
            return {}

        try:
            embeddings = self.embedding_model.encode(texts)
            similarity_matrix = cosine_similarity(embeddings)

            clusters = {}
            used = set()
            cluster_id = 0

            for i in range(len(texts)):
                if i in used:
                    continue

                cluster = [texts[i]]
                used.add(i)

                for j in range(i + 1, len(texts)):
                    if j not in used and similarity_matrix[i][j] > threshold:
                        cluster.append(texts[j])
                        used.add(j)

                if len(cluster) > 1:
                    clusters[cluster_id] = cluster
                    cluster_id += 1

            return clusters

        except Exception as e:
            logger.error(f"Error clustering texts: {e}")
            return {}


# Global instance
_nlp_processor_instance = None


def get_nlp_processor() -> NLPProcessor:
    """Get or create NLP processor instance"""
    global _nlp_processor_instance
    if _nlp_processor_instance is None:
        _nlp_processor_instance = NLPProcessor()
    return _nlp_processor_instance
