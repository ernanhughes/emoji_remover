import re
import os
import shutil
import sqlite3
import logging
import nltk
import torch
import markdown
from datetime import datetime
from dataclasses import dataclass
from collections import Counter
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure required NLTK resources are available
nltk.download("punkt")

# 📌 **Configuration Class**
class AIWatermarkConfig:
    """Configuration settings for AI watermark detection."""
    APP_NAME = "ai_watermark_detector"
    BACKUP_DIR = r"C:\backup"
    DB_PATH = "ai_detection.db"
    REPORT_DIR = r"C:\backup\reports"
    LOG_FILE = f"{APP_NAME}.log"
    AI_TOKENS = {"thus", "moreover", "indeed", "consequently", "notably"}  # Probabilistic watermark words
    FILE_EXTENSIONS = (".txt", ".md", ".json", ".py", ".log")  # File types to scan
    AI_MODEL = "roberta-base-openai-detector"  # AI classifier model


# 📌 **Setup Logging**
logging.basicConfig(
    filename=AIWatermarkConfig.LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(AIWatermarkConfig.APP_NAME)


# 📌 **Dataclass for Storing Detection Results**
@dataclass
class DetectionResult:
    """Structure for storing AI watermark detection results."""
    date: str
    filename: str
    filepath: str
    token_watermark: bool
    zero_width_watermark: bool
    style_fingerprint: bool
    token_watermark_score: str
    zero_width_watermark_score: str
    style_fingerprint_score: str
    verdict: str


# 📌 **Database Class**
class AIWatermarkDatabase:
    """Handles database initialization and updates for AI watermark detection."""

    def __init__(self, db_path=AIWatermarkConfig.DB_PATH):
        """Initialize the database connection."""
        self.db_path = db_path
        self._setup_database()

    def _setup_database(self):
        """Create the detection_results table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        filename TEXT,
                        filepath TEXT,
                        token_watermark TEXT,
                        zero_width_watermark TEXT,
                        style_fingerprint TEXT,
                        token_watermark_score TEXT,
                        zero_width_watermark_score TEXT,
                        style_fingerprint_score TEXT,
                        verdict TEXT
                    )
                """)
                conn.commit()
            logger.info("Database setup completed.")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")

    def save_detection_result(self, result: DetectionResult):
        """Save detection results to SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO detection_results (date, filename, filepath, 
                        token_watermark, zero_width_watermark, style_fingerprint, 
                        token_watermark_score, zero_width_watermark_score, style_fingerprint_score, 
                        verdict)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.date,
                    result.filename,
                    result.filepath,
                    str(result.token_watermark),
                    str(result.zero_width_watermark),
                    str(result.style_fingerprint),
                    result.token_watermark_score,
                    result.zero_width_watermark_score,
                    result.style_fingerprint_score,
                    result.verdict
                ))
                conn.commit()
            logger.info(f"Saved detection results for {result.filename} to database.")
        except Exception as e:
            logger.error(f"Failed to save results for {result.filename}: {e}")


# 📌 **AI Watermark Detector Class**
class AIWatermarkDetector:
    """
    Detects AI-generated watermarks in text using:
    - Token frequency analysis
    - Zero-width character detection
    - Style-based AI classification
    """

    def __init__(self, config=AIWatermarkConfig()):
        """Initialize AI watermark detector and database connection."""
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.AI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.AI_MODEL)
        self.db = AIWatermarkDatabase(config.DB_PATH)

        logger.info("AIWatermarkDetector initialized successfully.")

    def detect_token_watermark(self, text):
        """Check if AI-favored words appear more frequently than expected."""
        tokens = word_tokenize(text.lower())
        token_counts = Counter(tokens)

        ai_bias_score = sum(token_counts[token] for token in self.config.AI_TOKENS if token in token_counts)
        normalized_score = ai_bias_score / max(len(tokens), 1)  # Avoid division by zero

        return normalized_score > 0.02, f"AI Token Watermark Score: {normalized_score:.4f}"

    def detect_zero_width_chars(self, text):
        """Detect hidden Unicode zero-width characters used for AI watermarking."""
        hidden_chars = re.findall(r'[\u200B\u200C\u200D\uFEFF]', text)
        return len(hidden_chars) > 0, f"Zero-Width Characters Found: {len(hidden_chars)}"

    def detect_style_patterns(self, text):
        """Use an AI classifier to detect stylistic patterns typical of AI-generated text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        ai_confidence = outputs.logits.softmax(dim=-1).tolist()[0][1]  # AI-generated confidence

        return ai_confidence > 0.7, f"Style AI Confidence Score: {ai_confidence:.4f}"

    def analyze_text(self, file_path):
        """Analyze a text file for AI watermarks and save the results."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"❌ Error reading file {file_path}: {e}")
            return None

        token_watermark, token_score = self.detect_token_watermark(text)
        zero_width_watermark, zero_width_score = self.detect_zero_width_chars(text)
        style_fingerprint, style_score = self.detect_style_patterns(text)

        ai_detected = any([token_watermark, zero_width_watermark, style_fingerprint])
        final_verdict = "AI-Generated Text Detected" if ai_detected else "Likely Human-Written"

        # Create dataclass instance
        result = DetectionResult(
            date=get_current_date(),
            filename=os.path.basename(file_path),
            filepath=file_path,
            token_watermark=token_watermark,
            zero_width_watermark=zero_width_watermark,
            style_fingerprint=style_fingerprint,
            token_watermark_score=token_score,
            zero_width_watermark_score=zero_width_score,
            style_fingerprint_score=style_score,
            verdict=final_verdict
        )

        self.db.save_detection_result(result)
        logger.info(f"Analysis complete for {file_path}: {final_verdict}")
        return result


def get_current_date():
    """Returns today's date formatted as YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")


def generate_markdown_report(results):
    """Generate a Markdown report of the AI detection results."""
    report_filename = f"{AIWatermarkConfig.REPORT_DIR}/ai_detection_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    os.makedirs(AIWatermarkConfig.REPORT_DIR, exist_ok=True)

    with open(report_filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"### 📂 File: {result.filename}\n")
            f.write(f"- **Verdict:** {result.verdict}\n")
            f.write(f"- **Token Watermark Score:** {result.token_watermark_score}\n\n")

    logger.info(f"Markdown report generated: {report_filename}")


# Run the analysis and generate the report
if __name__ == "__main__":
    folder_path = "C:/Users/ernan/Project/programmer.ie/content/post"

    results = []
    for root, dirs, files in os.walk(os.path.abspath(folder_path)):
        for file in files:
            results.append(AIWatermarkDetector().analyze_text(os.path.join(root, file)))
    generate_markdown_report(results)
