import base64
import io
import json
import re
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter

from config import OCRConfig
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
    model_name: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def is_valid(self) -> bool:
        """Check if essential fields are present"""
        essential_fields = [
            getattr(self, field) for field in OCRConfig.DECISION_METRICS["nid_validation_fields"]]
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

            # Enhanced image processing for better OCR
            width, height = pil_image.size
            min_width, min_height = 1200, 800  # Increased minimum size for better quality
            
            if width < min_width or height < min_height:
                scale_factor = max(min_width / width, min_height / height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            # Apply multiple enhancement passes for optimal OCR results
            # 1. Contrast enhancement
            contrast_enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = contrast_enhancer.enhance(1.3)  # Increased contrast
            
            # 2. Sharpness enhancement
            sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = sharpness_enhancer.enhance(1.4)  # Increased sharpness
            
            # 3. Brightness adjustment
            brightness_enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = brightness_enhancer.enhance(1.1)  # Slight brightness increase
            
            # 4. Color enhancement for better text visibility
            color_enhancer = ImageEnhance.Color(pil_image)
            pil_image = color_enhancer.enhance(0.9)  # Reduce color saturation for better text focus

            # 5. Apply noise reduction filter
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
            
            # 6. Apply unsharp mask for edge enhancement
            pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            # Save enhanced image with high quality
            processed_filename = f"nid_processed_{uuid.uuid4().hex}.jpg"
            processed_path = processed_dir / processed_filename
            pil_image.save(
                processed_path, 
                quality=95,  # Increased JPEG quality
                optimize=True,
                progressive=True
            )
            logger.info(f"Enhanced image saved to: {processed_path}")

            return pil_image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise


class VLMClient:
    """Client for interacting with Vision Language Models"""

    def __init__(self):
        self.base_url = OCRConfig.LLM_CONFIG["base_url"]
        self.model = OCRConfig.LLM_CONFIG["model"]
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

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=OCRConfig.IMAGE_CONFIG["jpeg_quality"])
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def extract_text(self, image: Image.Image, prompt: str) -> str:
        """Extract text using Vision Language Model"""
        try:
            image_b64 = self._encode_image(image)
            
            # Enhanced options for better accuracy
            OCRConfig.LLM_CONFIG["options"] = {
                "temperature": 0.01,  # Very low temperature for consistent output
                "top_p": 0.95,       # High top_p for focused responses
                "num_predict": 2000,  # Sufficient tokens for complete response
                "repeat_penalty": 1.05,  # Low repeat penalty to avoid truncation
                "top_k": 10,         # Limit vocabulary choices for consistency
                "num_ctx": 4096,     # Large context window
                "stop": ["```", "```json", "```\n"],  # Stop at code blocks
            }
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": OCRConfig.LLM_CONFIG,
            }
            
            response = requests.post(
                self.base_url, json=payload, timeout=OCRConfig.LLM_CONFIG["timeout"])
            response.raise_for_status()
            
            result = response.json().get('response', '')
            logger.info(f"VLM response received, length: {len(result)}")
            
            return result
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
        return """You are an expert OCR system specialized in extracting information from Bangladesh National ID cards with 99% accuracy.

IMPORTANT: This is a Bangladesh National ID card (front side). Analyze the image carefully and extract information with extreme precision.

REQUIRED FIELDS TO EXTRACT:
1. English Name - Look for the "Name" field in English text
2. Bengali Name - Look for the "নাম" field in Bengali text  
3. Father's Name - Look for the "পিতা" field in Bengali text
4. Mother's Name - Look for the "মাতা" field in Bengali text
5. NID Number - Look for the "NID NO" or "NID Number" field
6. Date of Birth - Look for the "Date of Birth" or "জন্ম তারিখ" field

EXTRACTION RULES:
- Extract text EXACTLY as it appears on the card, character by character
- For Bengali text, preserve ALL original Bengali characters and diacritics
- For NID number: 
  * Include ONLY the digits (0-9)
  * Remove any spaces, dashes, or other characters
  * Common formats: 10 digits, 13 digits, or 17 digits
  * Be extremely careful with similar characters: 0 vs O, 1 vs I, 6 vs G, 8 vs B
- For date of birth:
  * Convert to DD/MM/YYYY format
  * If month is in text (Jan, Feb, etc.), convert to number (01, 02, etc.)
  * Ensure year is 4 digits
- For names:
  * Remove any field labels (e.g., "পিতা:", "মাতা:", "নাম:")
  * Keep only the actual name text
  * Preserve exact spelling and capitalization
- If any field is unclear, illegible, or missing, mark it as "NOT_FOUND"
- Do NOT guess or approximate any values

VALIDATION CHECKS:
- NID number should contain only digits (0-9)
- Date should be in valid format (DD/MM/YYYY)
- Names should not be empty or contain only special characters
- English name should contain only English letters and spaces
- Bengali names should contain Bengali characters

RESPONSE FORMAT (JSON only):
{
    "english_name": "exact English name from card",
    "bengali_name": "exact Bengali name from card", 
    "father_name": "exact father's name in Bengali without 'পিতা'",
    "mother_name": "exact mother's name in Bengali without 'মাতা'",
    "nid_no": "exact NID number digits only",
    "date_of_birth": "DD/MM/YYYY format",
    "confidence": "high/medium/low based on text clarity",
    "extraction_notes": "any important observations about the extraction process"
}

CRITICAL: Provide ONLY the JSON response. No additional text, explanations, or markdown formatting."""

    @staticmethod
    def get_front_side_fallback_prompt() -> str:
        """Generate fallback prompt for front side when primary extraction fails"""
        return """
You are an expert at reading Bangladesh National ID cards. 

Look at this image and extract the following information in JSON format:

{
    "english_name": "English name from the card",
    "bengali_name": "Bengali name from the card",
    "father_name": "Father's name in Bengali",
    "mother_name": "Mother's name in Bengali", 
    "nid_no": "NID number (digits only)",
    "date_of_birth": "Date of birth in DD/MM/YYYY format"
}

Focus on accuracy. If you can't read something clearly, use "NOT_FOUND". Return only the JSON.
"""

    @staticmethod
    def get_back_side_prompt() -> str:
        """Generate prompt for back side of NID card"""
        return """You are an expert OCR system specialized in extracting information from Bangladesh National ID cards with 99% accuracy.

IMPORTANT: This is a Bangladesh National ID card (back side). Analyze the image carefully and extract information with extreme precision.

REQUIRED FIELDS TO EXTRACT:
1. Address - Look for the "ঠিকানা" (address) section
2. Blood Group - Look for the "রক্তের গ্রুপ" (blood group) field

EXTRACTION RULES:
- Extract text EXACTLY as it appears on the card, character by character
- For Bengali text, preserve ALL original Bengali characters and diacritics
- For address:
  * Look for the complete address section starting with "ঠিকানা:"
  * Include the full address including:
    - বাসা/হোল্ডিং: (house/holding number)
    - গ্রাম/রাস্তা: (village/street name)
    - ডাকঘর: (post office)
  * Extract ONLY the text that comes AFTER the field labels
  * Preserve exact formatting, line breaks, and punctuation
- For blood group:
  * Look for blood group notation (A+, B+, O+, AB+, A-, B-, O-, AB-)
  * Extract the exact notation as shown
  * If multiple blood groups are shown, extract the primary one
- If any field is unclear, illegible, or missing, mark it as "NOT_FOUND"
- Do NOT guess or approximate any values

VALIDATION CHECKS:
- Address should contain Bengali text and be reasonably long
- Blood group should match standard notation patterns
- Address should include key Bengali words like "ঠিকানা", "বাসা", "গ্রাম", "ডাকঘর"

RESPONSE FORMAT (JSON only):
{
    "address": "complete address in Bengali starting with 'ঠিকানা:' followed by full address details",
    "blood_group": "exact blood group notation (e.g., A+, B+, O+, AB+)",
    "confidence": "high/medium/low based on text clarity",
    "extraction_notes": "any important observations about the extraction process"
}

CRITICAL: Provide ONLY the JSON response. No additional text, explanations, or markdown formatting.
"""

    @staticmethod
    def get_back_side_fallback_prompt() -> str:
        """Generate fallback prompt for back side when primary extraction fails"""
        return """You are an expert at reading Bangladesh National ID cards.

Look at this image and extract the following information in JSON format:

{
    "address": "Full address in Bengali",
    "blood_group": "Blood group (A+, B+, O+, AB+, etc.)"
}

Focus on accuracy. If you can't read something clearly, use "NOT_FOUND". Return only the JSON."""


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
            
            # Try multiple JSON extraction strategies
            json_str = None
            
            # Strategy 1: Look for JSON between triple backticks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            
            # Strategy 2: Look for JSON between curly braces
            if not json_str:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
            
            # Strategy 3: Look for any JSON-like structure
            if not json_str:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
            
            if json_str:
                # Clean up the JSON string
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                # Parse JSON
                parsed_data = json.loads(json_str)
                
                # Validate and clean the parsed data
                cleaned_data = {}
                for key, value in parsed_data.items():
                    if value is not None and str(value).strip():
                        # Clean up common OCR artifacts
                        if isinstance(value, str):
                            cleaned_value = value.strip()
                            # Remove common OCR artifacts
                            cleaned_value = re.sub(r'[^\w\s\-\.\/\+,]', '', cleaned_value)
                            cleaned_data[key] = cleaned_value
                        else:
                            cleaned_data[key] = value
                
                logger.info(f"Successfully parsed VLM response with {len(cleaned_data)} fields")
                return cleaned_data
            
            logger.warning("No valid JSON found in VLM response")
            logger.debug(f"Raw response: {response[:200]}...")
            return {}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Problematic JSON string: {json_str}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error parsing VLM response: {e}")
            return {}

    @staticmethod
    def _validate_nid_number(nid: str) -> bool:
        """Validate NID number format"""
        clean_nid = re.sub(r'[^\d]', '', str(nid))
        return len(clean_nid) in [10, 13, 17] and clean_nid.isdigit()

    @staticmethod
    def _validate_date_of_birth(dob: str) -> bool:
        """Validate date of birth format"""
        try:
            formats = ['%d/%m/%Y', '%d-%m-%Y',
                       '%Y-%m-%d', '%d %b %Y', '%d.%m.%Y']
            for fmt in formats:
                parsed_date = datetime.strptime(str(dob), fmt)
                if 1900 <= parsed_date.year <= datetime.now(timezone.utc).year:
                    return True
            return False
        except ValueError:
            return False

    @staticmethod
    def _normalize_date_format(date_str: str) -> str:
        """Normalize date to DD/MM/YYYY format"""
        try:
            formats = ['%d/%m/%Y', '%d-%m-%Y',
                       '%Y-%m-%d', '%d %b %Y', '%d.%m.%Y']
            for fmt in formats:
                date_obj = datetime.strptime(str(date_str), fmt)
                return date_obj.strftime('%d/%m/%Y')
            return str(date_str)
        except Exception as e:
            logger.error(f"Failed to normalize date format: {str(e)}")
            return str(date_str)

    @staticmethod
    def _clean_and_validate_extracted_data(data: Dict, side: NIDSide) -> Dict:
        """Clean and validate extracted data for consistency and accuracy"""
        cleaned_data = {}
        
        try:
            if side == NIDSide.FRONT:
                # Clean and validate front side data
                if 'english_name' in data:
                    name = str(data['english_name']).strip()
                    # Remove common OCR artifacts and normalize
                    name = re.sub(r'[^\w\s\-\.]', '', name)
                    name = ' '.join(name.split())  # Normalize whitespace
                    cleaned_data['english_name'] = name if name else None
                
                if 'bengali_name' in data:
                    name = str(data['bengali_name']).strip()
                    # Preserve Bengali characters but clean artifacts
                    name = re.sub(r'[^\u0980-\u09FF\s\-\.]', '', name)
                    name = ' '.join(name.split())
                    cleaned_data['bengali_name'] = name if name else None
                
                if 'father_name' in data:
                    name = str(data['father_name']).strip()
                    # Remove 'পিতা' prefix if present
                    name = re.sub(r'^পিতা\s*[:\.]?\s*', '', name)
                    name = re.sub(r'[^\u0980-\u09FF\s\-\.]', '', name)
                    name = ' '.join(name.split())
                    cleaned_data['father_name'] = name if name else None
                
                if 'mother_name' in data:
                    name = str(data['mother_name']).strip()
                    # Remove 'মাতা' prefix if present
                    name = re.sub(r'^মাতা\s*[:\.]?\s*', '', name)
                    name = re.sub(r'[^\u0980-\u09FF\s\-\.]', '', name)
                    name = ' '.join(name.split())
                    cleaned_data['mother_name'] = name if name else None
                
                if 'nid_no' in data:
                    nid = str(data['nid_no']).strip()
                    # Extract only digits
                    nid = re.sub(r'[^\d]', '', nid)
                    cleaned_data['nid_no'] = nid if nid else None
                
                if 'date_of_birth' in data:
                    dob = str(data['date_of_birth']).strip()
                    # Normalize date format
                    cleaned_data['date_of_birth'] = NIDDataExtractor._normalize_date_format(dob)
                
                if 'confidence' in data:
                    cleaned_data['confidence'] = data['confidence']
                
                if 'extraction_notes' in data:
                    cleaned_data['extraction_notes'] = data['extraction_notes']
                    
            elif side == NIDSide.BACK:
                # Clean and validate back side data
                if 'address' in data:
                    address = str(data['address']).strip()
                    # Clean address but preserve Bengali structure
                    address = re.sub(r'[^\u0980-\u09FF\w\s\-\.\/\:,]', '', address)
                    address = ' '.join(address.split())
                    cleaned_data['address'] = address if address else None
                
                if 'blood_group' in data:
                    bg = str(data['blood_group']).strip().upper()
                    # Validate blood group format
                    if re.match(r'^(A|B|AB|O)[+-]$', bg):
                        cleaned_data['blood_group'] = bg
                    else:
                        # Try to extract valid blood group from text
                        bg_match = re.search(r'(A|B|AB|O)[+-]', bg)
                        if bg_match:
                            cleaned_data['blood_group'] = bg_match.group()
                        else:
                            cleaned_data['blood_group'] = None
                
                if 'confidence' in data:
                    cleaned_data['confidence'] = data['confidence']
                
                if 'extraction_notes' in data:
                    cleaned_data['extraction_notes'] = data['extraction_notes']
            
            logger.info(f"Cleaned {side.value} side data: {len(cleaned_data)} fields")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning {side.value} side data: {str(e)}")
            return data  # Return original data if cleaning fails

    def extract_from_single_side(self, image_path: str, side: NIDSide, processed_dir: Path) -> Dict:
        """
        Extract information from a single side of NID card with retry mechanism
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

            # Try primary prompt first
            prompt = self.prompt_generator.get_front_side_prompt(
            ) if side == NIDSide.FRONT else self.prompt_generator.get_back_side_prompt()
            
            logger.info(f"Sending {side.value} side to VLM with primary prompt")
            vlm_response = self.vlm_client.extract_text(enhanced_image, prompt)
            
            logger.info(f"Parsing VLM response for {side.value} side")
            extracted_data = self._parse_vlm_response(vlm_response)
            
            # If primary extraction failed or returned insufficient data, try fallback prompt
            if not extracted_data or self._is_extraction_incomplete(extracted_data, side):
                logger.warning(f"Primary extraction incomplete for {side.value} side, trying fallback prompt")
                
                fallback_prompt = self.prompt_generator.get_front_side_fallback_prompt(
                ) if side == NIDSide.FRONT else self.prompt_generator.get_back_side_fallback_prompt()
                
                logger.info(f"Retrying {side.value} side with fallback prompt")
                fallback_response = self.vlm_client.extract_text(enhanced_image, fallback_prompt)
                fallback_data = self._parse_vlm_response(fallback_response)
                
                # Merge fallback data with primary data, preferring primary data
                if fallback_data:
                    extracted_data = {**fallback_data, **extracted_data}
                    logger.info(f"Fallback extraction provided additional data for {side.value} side")
            
            if extracted_data:
                logger.info(f"Cleaning and validating {side.value} side data")
                cleaned_data = self._clean_and_validate_extracted_data(extracted_data, side)
                
                # Add model name to the response
                cleaned_data['model_name'] = self.vlm_client.model
                
                logger.info(f"Successfully extracted {len(cleaned_data)} fields from {side.value} side")
                return cleaned_data
            else:
                logger.warning(f"No data extracted from {side.value} side after retry")
                return {}

        except Exception as e:
            logger.error(f"Error extracting from {side.value} side: {str(e)}")
            return {}

    def _is_extraction_incomplete(self, data: Dict, side: NIDSide) -> bool:
        """Check if extraction is incomplete and needs fallback prompt"""
        if side == NIDSide.FRONT:
            required_fields = ['english_name', 'nid_no', 'date_of_birth']
            missing_critical = sum(1 for field in required_fields if not data.get(field))
            return missing_critical > 1  # More than 1 critical field missing
        elif side == NIDSide.BACK:
            required_fields = ['address', 'blood_group']
            missing_critical = sum(1 for field in required_fields if not data.get(field))
            return missing_critical > 0  # Any critical field missing
        return False

    def get_extraction_feedback(self, extracted_data: Dict, side: NIDSide) -> Dict:
        """Provide detailed feedback on extraction quality and suggestions for improvement"""
        feedback = {
            "extraction_quality": "unknown",
            "missing_fields": [],
            "confidence_level": "unknown",
            "suggestions": [],
            "field_analysis": {},
            "model_name": self.vlm_client.model
        }
        
        try:
            if side == NIDSide.FRONT:
                expected_fields = ['english_name', 'bengali_name', 'father_name', 'mother_name', 'nid_no', 'date_of_birth']
                confidence = extracted_data.get('confidence', 'unknown')
                
                # Analyze each field
                for field in expected_fields:
                    value = extracted_data.get(field)
                    if not value or value == "NOT_FOUND":
                        feedback["missing_fields"].append(field)
                    
                    # Field-specific analysis
                    if field == 'nid_no' and value:
                        if not re.match(r'^\d{10,17}$', str(value)):
                            feedback["field_analysis"][field] = "Invalid format - should be 10-17 digits"
                        else:
                            feedback["field_analysis"][field] = "Valid format"
                    
                    elif field == 'date_of_birth' and value:
                        if not re.match(r'^\d{2}/\d{2}/\d{4}$', str(value)):
                            feedback["field_analysis"][field] = "Invalid format - should be DD/MM/YYYY"
                        else:
                            feedback["field_analysis"][field] = "Valid format"
                    
                    elif field in ['english_name', 'bengali_name', 'father_name', 'mother_name'] and value:
                        if len(str(value)) < 2:
                            feedback["field_analysis"][field] = "Too short - may be incomplete"
                        else:
                            feedback["field_analysis"][field] = "Appears complete"
                
                # Determine overall quality
                missing_count = len(feedback["missing_fields"])
                if missing_count == 0:
                    feedback["extraction_quality"] = "excellent"
                elif missing_count <= 2:
                    feedback["extraction_quality"] = "good"
                elif missing_count <= 4:
                    feedback["extraction_quality"] = "fair"
                else:
                    feedback["extraction_quality"] = "poor"
                
                feedback["confidence_level"] = confidence
                
                # Generate suggestions
                if missing_count > 0:
                    feedback["suggestions"].append("Ensure the image is well-lit and focused")
                    feedback["suggestions"].append("Avoid shadows and glare on the card")
                    feedback["suggestions"].append("Ensure the card fills most of the image frame")
                    feedback["suggestions"].append("Try capturing from a slightly different angle")
                
                if confidence == "low":
                    feedback["suggestions"].append("Image quality may be too low for reliable extraction")
                    feedback["suggestions"].append("Consider using a higher resolution camera")
                
            elif side == NIDSide.BACK:
                expected_fields = ['address', 'blood_group']
                confidence = extracted_data.get('confidence', 'unknown')
                
                for field in expected_fields:
                    value = extracted_data.get(field)
                    if not value or value == "NOT_FOUND":
                        feedback["missing_fields"].append(field)
                    
                    if field == 'blood_group' and value:
                        if not re.match(r'^(A|B|AB|O)[+-]$', str(value).upper()):
                            feedback["field_analysis"][field] = "Invalid format - should be A+, B+, O+, or AB+"
                        else:
                            feedback["field_analysis"][field] = "Valid format"
                    
                    elif field == 'address' and value:
                        if len(str(value)) < 10:
                            feedback["field_analysis"][field] = "Too short - may be incomplete"
                        else:
                            feedback["field_analysis"][field] = "Appears complete"
                
                missing_count = len(feedback["missing_fields"])
                if missing_count == 0:
                    feedback["extraction_quality"] = "excellent"
                elif missing_count == 1:
                    feedback["extraction_quality"] = "good"
                else:
                    feedback["extraction_quality"] = "poor"
                
                feedback["confidence_level"] = confidence
                
                if missing_count > 0:
                    feedback["suggestions"].append("Ensure the entire address section is visible")
                    feedback["suggestions"].append("Check that Bengali text is clearly readable")
                    feedback["suggestions"].append("Avoid cutting off any part of the address")
            
            logger.info(f"Generated feedback for {side.value} side: {feedback['extraction_quality']} quality")
            return feedback
            
        except Exception as e:
            logger.error(f"Error generating extraction feedback: {str(e)}")
            return feedback

    def extract_complete_nid_data(self, front_image_path: str, back_image_path: str, processed_dir: Path) -> NIDData:
        """
        Extract complete NID data from both sides of the card
        """
        logger.info("Starting complete NID data extraction")
        nid_data = NIDData()
        
        # Set the model name used for extraction
        nid_data.model_name = self.vlm_client.model

        try:
            # Extract front side data
            logger.info("Processing front side of NID card")
            front_data = self.extract_from_single_side(
                front_image_path,
                NIDSide.FRONT,
                processed_dir,
            )
            
            # Extract back side data
            logger.info("Processing back side of NID card")
            back_data = self.extract_from_single_side(
                back_image_path,
                NIDSide.BACK,
                processed_dir,
            )

            # Populate NID data structure with validated data
            if front_data:
                logger.info("Populating front side data")
                nid_data.english_name = front_data.get('english_name')
                nid_data.bengali_name = front_data.get('bengali_name')
                nid_data.father_name = front_data.get('father_name')
                nid_data.mother_name = front_data.get('mother_name')
                nid_data.nid_no = front_data.get('nid_no')
                nid_data.date_of_birth = front_data.get('date_of_birth')
                
                # Log extraction confidence
                confidence = front_data.get('confidence', 'unknown')
                logger.info(f"Front side extraction confidence: {confidence}")

            if back_data:
                logger.info("Populating back side data")
                nid_data.address = back_data.get('address')
                nid_data.blood_group = back_data.get('blood_group')
                
                # Log extraction confidence
                confidence = back_data.get('confidence', 'unknown')
                logger.info(f"Back side extraction confidence: {confidence}")

            # Validate extracted data quality
            extracted_fields = sum(1 for field in [
                nid_data.english_name, nid_data.bengali_name, nid_data.father_name,
                nid_data.mother_name, nid_data.nid_no, nid_data.date_of_birth,
                nid_data.address, nid_data.blood_group
            ] if field is not None)
            
            logger.info(f"Complete extraction completed. Extracted {extracted_fields}/8 fields")
            
            # Log any missing critical fields
            missing_fields = []
            if not nid_data.nid_no:
                missing_fields.append("NID Number")
            if not nid_data.date_of_birth:
                missing_fields.append("Date of Birth")
            if not nid_data.english_name:
                missing_fields.append("English Name")
                
            if missing_fields:
                logger.warning(f"Missing critical fields: {', '.join(missing_fields)}")
            
            return nid_data
            
        except Exception as e:
            logger.error(f"Complete extraction failed: {str(e)}")
            return nid_data
