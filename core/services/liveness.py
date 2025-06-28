import base64
import io
import json
import re
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import cv2
import mediapipe as mp
import numpy as np
import requests
from PIL import Image, ImageEnhance

from config import FLDConfig
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
        return self.confidence >= FLDConfig.DECISION_METRICS["liveness_confidence_threshold"]


class FaceImagePreprocessor:
    """Image preprocessing utilities for face liveness detection"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_faces_with_mediapipe(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Detect faces in the input image using MediaPipe FaceMesh.

        This method uses MediaPipe to detect faces and facial landmarks in the image.
        It also checks for multiple face detection and returns the cropped face region.

        Args:
            image: Input image as numpy array

        Returns:
            Tuple[bool, str, Optional[np.ndarray]]: A tuple containing:
                - is_success (bool): True if face detection was successful, False otherwise.
                - message (str): A message describing the result or any errors encountered.
        """
        try:
            h, w, _ = image.shape

            with self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,  # Treat the image as a static image (no video stream)
                    max_num_faces=3,  # Allow detection of up to 3 faces for checking multiple faces
                    refine_landmarks=True,  # Enable more accurate landmark detection and refinement
                    min_detection_confidence=0.5,  # Set the minimum confidence threshold for face detection
            ) as face_mesh:
                # Process the image, converting it from BGR to RGB as face_mesh expects RGB input
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # If no faces are detected, return an error message indicating no face found
                if not results.multi_face_landmarks:
                    return False, "No face detected in the image"

                # If more than one face is detected, return an error message indicating multiple faces found
                if len(results.multi_face_landmarks) > 1:
                    return False, "Multiple faces detected. Please ensure only one face is visible in the image"

                # Extract face landmarks for the single detected face
                face_landmarks = results.multi_face_landmarks[0]

                # Get bounding box coordinates from landmarks
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]

                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                # Add padding around the face
                padding_ratio = FLDConfig.IMAGE_CONFIG.get("face_padding_ratio", 0.2)
                padding_x = int((x_max - x_min) * padding_ratio)
                padding_y = int((y_max - y_min) * padding_ratio)

                # Ensure coordinates are within image bounds
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(w, x_max + padding_x)
                y_max = min(h, y_max + padding_y)

                # Crop the face region
                face_region = image[y_min:y_max, x_min:x_max]

                # Validate face region size
                if face_region.shape[0] < 50 or face_region.shape[1] < 50:
                    return False, "Detected face region is too small"

                return True, "Face detection successful"

        except Exception as e:
            logger.error(f"MediaPipe face detection failed: {str(e)}")
            return False, f"Face detection error: {str(e)}"

    @staticmethod
    def enhance_face_image(image: Union[str, np.ndarray, Image.Image], processed_dir: Path) -> Image.Image:
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
            target_size = FLDConfig.IMAGE_CONFIG["max_size"]
            min_size = FLDConfig.IMAGE_CONFIG["min_size"]

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
                (ImageEnhance.Contrast, FLDConfig.IMAGE_CONFIG["contrast_factor"]),
                (ImageEnhance.Sharpness, FLDConfig.IMAGE_CONFIG["sharpness_factor"]),
                (ImageEnhance.Brightness, FLDConfig.IMAGE_CONFIG["brightness_factor"])
            ]:
                enhancer = enhancer_class(pil_image)
                pil_image = enhancer.enhance(factor)

            # Save processed image
            processed_filename = f"liveness_processed_{uuid.uuid4().hex}.jpg"
            processed_path = processed_dir / processed_filename
            pil_image.save(processed_path, quality=FLDConfig.IMAGE_CONFIG["jpeg_quality"])
            logger.info(f"Processed image saved to: {processed_path}")

            return pil_image

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise


class LivenessVLMClient:
    """Client for interacting with Qwen VLM for liveness detection"""

    def __init__(self):
        self.base_url = FLDConfig.LLM_CONFIG["base_url"]
        self.model = FLDConfig.LLM_CONFIG["model"]
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
        image.save(buffer, format='JPEG', quality=FLDConfig.IMAGE_CONFIG["jpeg_quality"])
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
                "options": FLDConfig.LLM_CONFIG["options"],
            }
            response = requests.post(
                self.base_url, json=payload,
                timeout=FLDConfig.LLM_CONFIG["timeout"],
            )
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

    def detect_liveness(self, image_path: str, processed_dir: Path) -> LivenessData:
        """
        Detect face liveness from an image

        Args:
            image_path: Path to the image file
            processed_dir: Path to save the processed image

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

            face_success, face_message = self.preprocessor.detect_faces_with_mediapipe(image)

            if not face_success:
                # Return early with fake classification for multiple faces or no face
                logger.warning(f"Face detection failed: {face_message}")
                return LivenessData(
                    is_live=False,
                    confidence=95.0,  # High confidence in rejection
                    result=LivenessResult.FAKE,
                    fake_type=FakeType.UNKNOWN_FAKE,
                    reasoning=f"Face detection validation failed: {face_message}. "
                              f"For liveness detection, exactly one clear face must be visible in the image.",
                    technical_analysis={
                        "face_detection_status": "failed",
                        "face_detection_message": face_message,
                        "multiple_faces_detected": "multiple faces" in face_message.lower(),
                        "no_face_detected": "no face" in face_message.lower(),
                        "quality_assessment": "invalid_input"
                    }
                )

            logger.info("Face region successfully detected")

            enhanced_image = self.preprocessor.enhance_face_image(image, processed_dir)
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
