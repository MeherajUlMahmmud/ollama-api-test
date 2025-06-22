import base64
import io
import json
import re
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance

from config import LLM_CONFIG, IMAGE_CONFIG, DECISION_METRICS, PROCESSED_DIR
from logger import Logger

logger = Logger.get_logger()


class LivenessResult(Enum):
    """Enumeration for liveness detection results"""
    LIVE = "live"
    FAKE = "fake"
    UNCERTAIN = "uncertain"


class FakeType(Enum):
    """Types of fake/spoof attacks"""
    PHOTO_ATTACK = "photo_attack"
    VIDEO_REPLAY = "video_replay"
    MASK_3D = "mask_3d"
    DEEPFAKE = "deepfake"
    SCREEN_DISPLAY = "screen_display"
    PRINTED_PHOTO = "printed_photo"
    UNKNOWN_FAKE = "unknown_fake"


@dataclass
class LivenessData:
    """Data structure for liveness detection results"""
    is_live: bool
    confidence: float
    result: LivenessResult
    fake_type: Optional[FakeType] = None
    reasoning: Optional[str] = None
    technical_analysis: Optional[Dict] = None

    def to_json(self) -> str:
        """Convert to JSON string"""
        data = asdict(self)
        data['result'] = self.result.value if self.result else None
        data['fake_type'] = self.fake_type.value if self.fake_type else None
        return json.dumps(data, indent=2, ensure_ascii=False)

    def is_reliable(self) -> bool:
        """Check if the result is reliable based on confidence"""
        return self.confidence >= DECISION_METRICS["liveness_confidence_threshold"]


class FaceImagePreprocessor:
    """Image preprocessing utilities for face liveness detection"""

    @staticmethod
    def enhance_face_image(image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """
        Enhance face image and save processed version
        """
        try:
            # Convert to PIL Image
            if isinstance(image, str):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image

            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Resize image
            width, height = pil_image.size
            target_size = IMAGE_CONFIG["max_size"]
            min_size = IMAGE_CONFIG["min_size"]

            if max(width, height) > target_size:
                if width > height:
                    new_width = target_size
                    new_height = int(height * (target_size / width))
                else:
                    new_height = target_size
                    new_width = int(width * (target_size / height))
                pil_image = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS)
            elif max(width, height) < min_size:
                scale_factor = min_size / max(width, height)
                new_size = (int(width * scale_factor),
                            int(height * scale_factor))
                pil_image = pil_image.resize(
                    new_size, Image.Resampling.LANCZOS)

            # Apply enhancements
            for enhancer_class, factor in [
                (ImageEnhance.Contrast, IMAGE_CONFIG["contrast_factor"]),
                (ImageEnhance.Sharpness, IMAGE_CONFIG["sharpness_factor"]),
                (ImageEnhance.Brightness, IMAGE_CONFIG["brightness_factor"])
            ]:
                enhancer = enhancer_class(pil_image)
                pil_image = enhancer.enhance(factor)

            # Save processed image
            processed_filename = f"liveness_processed_{uuid.uuid4().hex}.jpg"
            processed_path = PROCESSED_DIR / processed_filename
            pil_image.save(processed_path, quality=IMAGE_CONFIG["jpeg_quality"])
            logger.info(f"Processed image saved to: {processed_path}")

            return pil_image

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    @staticmethod
    def detect_face_region(image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop the main face region

        Args:
            image: Input image as numpy array

        Returns:
            Cropped face region or None if not detected
        """
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                padding = int(max(w, h) * IMAGE_CONFIG["face_padding_ratio"])
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                return image[y:y + h, x:x + w]

            logger.warning("No face detected in image")
            return None
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return None


class LivenessVLMClient:
    """Client for interacting with Qwen VLM for liveness detection"""

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

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=IMAGE_CONFIG["jpeg_quality"])
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def analyze_liveness(self, image: Image.Image, prompt: str) -> str:
        """Analyze face liveness using Qwen VLM"""
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
            logger.error(f"VLM analysis failed: {str(e)}")
            raise


class LivenessPromptGenerator:
    """Generates optimized prompts for face liveness detection"""

    @staticmethod
    def get_comprehensive_liveness_prompt() -> str:
        """Generate comprehensive prompt for face liveness detection"""
        return """
You are an expert computer vision system specialized in face liveness detection and anti-spoofing analysis. Your task is to determine if the face in this image is from a LIVE person or a FAKE/SPOOFED representation.

ANALYZE THE IMAGE FOR THESE CRITICAL LIVENESS INDICATORS:

🔍 LIVE FACE INDICATORS (Real Person):
1. **Skin Texture & Micro-details**: Natural skin pores, fine lines, skin imperfections, natural skin texture variations
2. **Eye Characteristics**: Natural eye moisture/reflections, iris details, pupil reactions, natural eye movement blur
3. **Lighting Consistency**: Natural shadows, consistent lighting across face, proper light interaction with 3D facial features
4. **Depth & Dimensionality**: Natural 3D facial contours, proper depth perception, realistic facial geometry
5. **Color Authenticity**: Natural skin tones, realistic color distribution, proper blood circulation signs
6. **Motion Artifacts**: Natural motion blur if present, consistent with live capture
7. **Facial Expression**: Natural micro-expressions, realistic muscle tension, authentic emotional display

⚠️ FAKE/SPOOF INDICATORS (Not Live):
1. **Photo Attack**: Flat appearance, paper/screen reflections, pixelation, printing artifacts, 2D characteristics
2. **Video Replay**: Screen bezels, pixel grid patterns, display artifacts, unnatural brightness/contrast
3. **3D Mask**: Artificial skin texture, unrealistic eye details, rigid facial features, material inconsistencies
4. **Deepfake/Digital**: Unnatural blending, digital artifacts, inconsistent lighting, temporal inconsistencies
5. **Screen Display**: Backlight bleeding, refresh rate artifacts, color banding, LCD/OLED characteristics
6. **Printed Photo**: Paper texture, printing dots, color saturation issues, folding marks

TECHNICAL ANALYSIS FOCUS:
- Examine pixel-level details and patterns
- Analyze lighting physics and shadows
- Check for digital manipulation artifacts
- Assess anatomical correctness and proportions
- Look for material inconsistencies
- Evaluate motion and temporal consistency

RESPONSE FORMAT (JSON):
{
    "is_live": true/false,
    "confidence": 85.5,
    "result": "live/fake/uncertain",
    "fake_type": "photo_attack/video_replay/mask_3d/deepfake/screen_display/printed_photo/unknown_fake" (only if fake),
    "reasoning": "Detailed explanation of why you determined live/fake with specific visual evidence",
    "technical_analysis": {
        "skin_texture_quality": "natural/artificial/unclear",
        "eye_authenticity": "realistic/fake/uncertain",
        "lighting_consistency": "natural/artificial/mixed",
        "depth_perception": "3d/2d/ambiguous",
        "color_authenticity": "natural/processed/artificial",
        "detected_artifacts": ["list of specific artifacts found"],
        "suspicious_patterns": ["list of suspicious visual patterns"],
        "quality_assessment": "high/medium/low image quality"
    }
}

CONFIDENCE SCORING GUIDELINES:
- 90-100%: Extremely confident with multiple clear indicators
- 80-89%: Highly confident with several strong indicators
- 70-79%: Confident with good supporting evidence
- 60-69%: Moderately confident with some indicators
- 50-59%: Low confidence, mixed or unclear signals
- Below 50%: Very uncertain, insufficient evidence

CRITICAL INSTRUCTIONS:
- Base your decision on MULTIPLE indicators, not just one factor
- Be especially careful with high-quality fakes that may have few obvious artifacts
- Consider the overall consistency of all visual elements
- If evidence is mixed or unclear, lean toward "uncertain" with lower confidence
- Provide specific visual evidence in your reasoning
- Be extremely thorough in your analysis

Analyze this image now and provide your detailed assessment:
"""


class FaceLivenessDetector:
    """Main class for face liveness detection"""

    def __init__(self):
        self.preprocessor = FaceImagePreprocessor()
        self.vlm_client = LivenessVLMClient()
        self.prompt_generator = LivenessPromptGenerator()

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

    @staticmethod
    def _validate_and_normalize_response(raw_data: Dict) -> LivenessData:
        """Validate and normalize the VLM response into LivenessData"""
        try:
            is_live = raw_data.get('is_live', False)
            confidence = float(raw_data.get('confidence', 0.0))
            confidence = max(0.0, min(100.0, confidence))
            result_str = raw_data.get('result', '').lower()
            result = {
                'live': LivenessResult.LIVE,
                'fake': LivenessResult.FAKE
            }.get(result_str, LivenessResult.UNCERTAIN)

            fake_type = None
            if not is_live and result == LivenessResult.FAKE:
                fake_type_str = raw_data.get('fake_type', '').lower()
                fake_type = {
                    'photo_attack': FakeType.PHOTO_ATTACK,
                    'video_replay': FakeType.VIDEO_REPLAY,
                    'mask_3d': FakeType.MASK_3D,
                    'deepfake': FakeType.DEEPFAKE,
                    'screen_display': FakeType.SCREEN_DISPLAY,
                    'printed_photo': FakeType.PRINTED_PHOTO
                }.get(fake_type_str, FakeType.UNKNOWN_FAKE)

            return LivenessData(
                is_live=is_live,
                confidence=confidence,
                result=result,
                fake_type=fake_type,
                reasoning=raw_data.get('reasoning', ''),
                technical_analysis=raw_data.get('technical_analysis', {}),
            )
        except Exception as e:
            logger.error(f"Error normalizing response: {str(e)}")
            return LivenessData(
                is_live=False,
                confidence=0.0,
                result=LivenessResult.UNCERTAIN,
                reasoning=f"Error processing response: {str(e)}"
            )

    def detect_liveness(self, image_path: str, focus_on_face: bool = True) -> LivenessData:
        """
        Detect face liveness from an image

        Args:
            image_path: Path to the image file
            focus_on_face: Crop to face region first

        Returns:
            LivenessData object with detection results
        """
        try:
            logger.info(f"Starting liveness detection for: {image_path}")
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            if focus_on_face:
                face_region = self.preprocessor.detect_face_region(image)
                image = face_region if face_region is not None else image
                logger.info("Face region detection attempted")

            enhanced_image = self.preprocessor.enhance_face_image(image)
            prompt = self.prompt_generator.get_comprehensive_liveness_prompt()
            vlm_response = self.vlm_client.analyze_liveness(enhanced_image, prompt)
            raw_data = self._parse_vlm_response(vlm_response)
            liveness_result = self._validate_and_normalize_response(raw_data)

            logger.info(
                f"Liveness detection completed - Result: {liveness_result.result.value}, Confidence: {liveness_result.confidence}%")
            return liveness_result

        except Exception as e:
            logger.error(f"Liveness detection failed: {str(e)}")
            return LivenessData(
                is_live=False,
                confidence=0.0,
                result=LivenessResult.UNCERTAIN,
                reasoning=f"Detection failed: {str(e)}"
            )
