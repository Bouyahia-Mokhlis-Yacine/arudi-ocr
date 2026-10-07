"""ArudiOCR: Arabic poetry images/text -> prosodic (arudi) writing."""
import warnings

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", message=".*mismatched key_padding_mask.*")

from .models import load_ocr, load_converter
from .pipeline import ArudiPipeline, E2EReader

__all__ = ["load_ocr", "load_converter", "ArudiPipeline", "E2EReader"]
