import base64
import io
import json
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter

from config import LLM_CONFIG, IMAGE_CONFIG, DECISION_METRICS
from logger import Logger

logger = Logger.get_logger()


class NIDSide(Enum):
    """Enumeration for NID card sides"""
    FRONT = "front"
    BACK = "back"


@dataclass
class NIDData:
    """Data structure for NID card information"""
    english_name: Optional[str] = None
    bengali_name: Optional[str] = None
    father_name: Optional[str] = None
    mother_name: Optional[str] = None
    nid_no: Optional[str] = None
    date_of_birth: Optional[str] = None
    address: Optional[str] = None
    blood_group: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def is_valid(self) -> bool:
        """Check if essential fields are present"""
        essential_fields = [
            getattr(self, field) for field in DECISION_METRICS["nid_validation_fields"]]
        return all(field is not None and str(field).strip() for field in essential_fields)


class ImagePreprocessor:
    """Image preprocessing utilities for better OCR accuracy"""

    @staticmethod
    def enhance_image(image: Union[str, np.ndarray, Image.Image], processed_dir: Path) -> Image.Image:
        """
        Enhance image quality for better OCR results
        """
        try:
            if isinstance(image, str):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image

            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            width, height = pil_image.size
            min_width, min_height = 800, 500
            if width < min_width or height < min_height:
                scale_factor = max(min_width / width, min_height / height)
                new_size = (int(width * scale_factor),
                            int(height * scale_factor))
                pil_image = pil_image.resize(
                    new_size, Image.Resampling.LANCZOS)

            for enhancer_class, factor in [
                (ImageEnhance.Contrast, IMAGE_CONFIG["contrast_factor"]),
                (ImageEnhance.Sharpness, IMAGE_CONFIG["sharpness_factor"])
            ]:
                enhancer = enhancer_class(pil_image)
                pil_image = enhancer.enhance(factor)

            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))

            processed_filename = f"nid_processed_{uuid.uuid4().hex}.jpg"
            processed_path = processed_dir / processed_filename
            pil_image.save(
                processed_path, quality=IMAGE_CONFIG["jpeg_quality"])
            logger.info(f"Processed image saved to: {processed_path}")

            return pil_image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise


class VLMClient:
    """Client for interacting with Vision Language Models"""

    def __init__(self):
        self.base_url = LLM_CONFIG["base_url"]
        self.model = LLM_CONFIG["model"]
        self._check_service_availability()

    def _check_service_availability(self):
        """Check if Ollama service is available"""
        try:
            health_url = self.base_url.replace('/api/generate', '/api/tags')
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            logger.info("Ollama service is available")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama service unavailable: {str(e)}")
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}")

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=IMAGE_CONFIG["jpeg_quality"])
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def extract_text(self, image: Image.Image, prompt: str) -> str:
        """Extract text using Vision Language Model"""
        try:
            image_b64 = self._encode_image(image)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": LLM_CONFIG["options"],
            }
            response = requests.post(
                self.base_url, json=payload, timeout=LLM_CONFIG["timeout"])
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.RequestException as e:
            logger.error(f"VLM request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"VLM extraction failed: {str(e)}")
            raise


class NIDPromptGenerator:
    """Generates optimized prompts for NID card information extraction"""

    @staticmethod
    def get_front_side_prompt() -> str:
        """Generate prompt for front side of NID card"""
        return """
You are an expert OCR system specialized in extracting information from Bangladesh National ID cards.

Analyze this image of a Bangladesh National ID card (front side) and extract the following information with high precision:

REQUIRED FIELDS TO EXTRACT:
1. English Name (Name field in English)
2. Bengali Name (নাম field in Bengali)
3. Father's Name (পিতা field in Bengali)
4. Mother's Name (মাতা field in Bengali)
5. NID Number (NID NO field - exactly as shown)
6. Date of Birth (Date of Birth field - format as DD/MM/YYYY)

EXTRACTION RULES:
- Extract text EXACTLY as it appears on the card
- For Bengali text, preserve the original Bengali characters
- For NID number, include all digits without spaces or formatting
- For date of birth, convert to DD/MM/YYYY format if needed
- If any field is unclear or missing, mark it as "NOT_FOUND"
- Be extremely careful with number recognition (0 vs O, 1 vs I, etc.)

RESPONSE FORMAT (JSON):
{
    "english_name": "exact English name from card",
    "bengali_name": "exact Bengali name",
    "father_name": "exact father's name in Bengali",
    "mother_name": "exact mother's name in Bengali",
    "nid_no": "exact NID number digits only",
    "date_of_birth": "DD/MM/YYYY format",
    "confidence": "high/medium/low based on text clarity"
}

Analyze the image carefully and provide the JSON response:
"""

    @staticmethod
    def get_back_side_prompt() -> str:
        """Generate prompt for back side of NID card"""
        return """
You are an expert OCR system specialized in extracting information from Bangladesh National ID cards.

Analyze this image of a Bangladesh National ID card (back side) and extract the following information:

REQUIRED FIELDS TO EXTRACT:
1. Address (ঠিকানা field in Bengali - full address)
2. Blood Group (রক্তের গ্রুপ field)

EXTRACTION RULES:
- Extract text EXACTLY as it appears on the card
- For Bengali text, preserve the original Bengali characters
- For address, include the complete address as written
- For blood group, extract the exact notation (A+, B+, O+, AB+, etc.)
- If any field is unclear or missing, mark it as "NOT_FOUND"

SPECIAL ADDRESS EXTRACTION RULES:
- Look for the word "ঠিকানা" on the card
- Extract ONLY the text that comes AFTER the "ঠিকানা" word as the address
- The address section should contain these specific Bengali words: "বাসা/হোল্ডিং:", "গ্রাম/রাস্তা:"
- If OCR makes mistakes and you find similar words that are close to these exact words, correct them:
  * Words similar to "ঠিকানা" → "ঠিকানা"
  * Words similar to "বাসা/হোল্ডিং:" → "বাসা/হোল্ডিং:"
  * Words similar to "গ্রাম/রাস্তা:" → "গ্রাম/রাস্তা:"
- Common OCR errors to watch for and correct:
  * "ঠিকানা" might be misread as "ঠিকানো", "ঠিকানন", "ঠিকানে"
  * "বাসা/হোল্ডিং:" might be misread as "বাসা/হোল্ডিং", "বাসা/হোল্ডিংঃ", "বাসা/হোল্ডিন:"
  * "গ্রাম/রাস্তা:" might be misread as "গ্রাম/রাস্তাঃ", "গ্রাম/রাস্তো:", "গ্রাম/রাস্তে:"

RESPONSE FORMAT (JSON):
{
    "address": "complete address in Bengali",
    "blood_group": "blood group notation",
    "confidence": "high/medium/low based on text clarity"
}

Analyze the image carefully and provide the JSON response:
"""


class NIDDataExtractor:
    """Main class for extracting data from NID cards"""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.vlm_client = VLMClient()
        self.prompt_generator = NIDPromptGenerator()

    @staticmethod
    def _parse_vlm_response(response: str) -> Dict:
        """Parse VLM response and extract JSON data"""
        try:
            response = response.strip()
            json_match = re.search(r'\{.*}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                return json.loads(json_str)
            logger.warning("No JSON found in VLM response")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {}

    def _validate_nid_number(self, nid: str) -> bool:
        """Validate NID number format"""
        clean_nid = re.sub(r'[^\d]', '', str(nid))
        return len(clean_nid) in [10, 13, 17] and clean_nid.isdigit()

    def _validate_date_of_birth(self, dob: str) -> bool:
        """Validate date of birth format"""
        try:
            formats = ['%d/%m/%Y', '%d-%m-%Y',
                       '%Y-%m-%d', '%d %b %Y', '%d.%m.%Y']
            for fmt in formats:
                parsed_date = datetime.strptime(str(dob), fmt)
                if 1900 <= parsed_date.year <= datetime.now().year:
                    return True
            return False
        except ValueError:
            return False

    def _normalize_date_format(self, date_str: str) -> str:
        """Normalize date to DD/MM/YYYY format"""
        try:
            formats = ['%d/%m/%Y', '%d-%m-%Y',
                       '%Y-%m-%d', '%d %b %Y', '%d.%m.%Y']
            for fmt in formats:
                date_obj = datetime.strptime(str(date_str), fmt)
                return date_obj.strftime('%d/%m/%Y')
            return str(date_str)
        except Exception:
            return str(date_str)

    def extract_from_single_side(self, image_path: str, side: NIDSide, processed_dir: Path) -> Dict:
        """
        Extract information from a single side of NID card
        """
        try:
            logger.info(f"Processing {side.value} side: {image_path}")
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            enhanced_image = self.preprocessor.enhance_image(
                image,
                processed_dir,
            )

            prompt = self.prompt_generator.get_front_side_prompt(
            ) if side == NIDSide.FRONT else self.prompt_generator.get_back_side_prompt()
            vlm_response = self.vlm_client.extract_text(enhanced_image, prompt)
            extracted_data = self._parse_vlm_response(vlm_response)

            return extracted_data
        except Exception as e:
            logger.error(f"Error extracting from {side.value} side: {str(e)}")
            return {}

    def extract_complete_nid_data(self, front_image_path: str, back_image_path: str, processed_dir: Path) -> NIDData:
        """
        Extract complete NID data from both sides of the card
        """
        logger.info("Starting complete NID data extraction")
        nid_data = NIDData()

        try:
            front_data = self.extract_from_single_side(
                front_image_path,
                NIDSide.FRONT,
                processed_dir,
            )
            back_data = self.extract_from_single_side(
                back_image_path,
                NIDSide.BACK,
                processed_dir,
            )

            if front_data:
                nid_data.english_name = front_data.get('english_name')
                nid_data.bengali_name = front_data.get('bengali_name')
                nid_data.father_name = front_data.get('father_name')
                nid_data.mother_name = front_data.get('mother_name')
                nid_data.nid_no = front_data.get('nid_no')
                nid_data.date_of_birth = self._normalize_date_format(
                    front_data.get('date_of_birth'))

            if back_data:
                nid_data.address = back_data.get('address')
                nid_data.blood_group = back_data.get('blood_group')

            logger.info("NID data extraction completed")
            return nid_data
        except Exception as e:
            logger.error(f"Complete extraction failed: {str(e)}")
            return nid_data
