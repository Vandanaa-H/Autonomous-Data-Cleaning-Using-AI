# Test configuration
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Test settings
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_UPLOAD_DIR = Path(__file__).parent / "test_uploads"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs"

# Create test directories
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_UPLOAD_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)
