# NID Service API Documentation

## Overview
The NID (National ID) Service API provides endpoints for extracting and validating information from Bangladesh National ID cards using advanced OCR and Vision Language Model technology.

## Base URL
```
http://localhost:8000/api
```

## Authentication
The API uses Basic Authentication. Include your credentials in the request headers:
```
Authorization: Basic <base64_encoded_username:password>
```

**Default Credentials:**
- Username: `admin`
- Password: `password123`

## Endpoints

### 1. Health Check
**GET** `/health`

General health check for the entire service.

**Response:**
```json
{
    "status": "healthy",
    "message": "Service is running"
}
```

### 2. NID Service Health Check
**GET** `/nid-health`

Specific health check for the NID service.

**Response:**
```json
{
    "status": "healthy",
    "service": "nid",
    "message": "NID service is operational"
}
```

### 3. NID Data Extraction (Smart)
**POST** `/nid-data-extraction`

Smart endpoint that automatically determines whether to do complete or single side extraction based on the provided images.

**Request Options:**

**Option A: Complete Extraction (Both Images)**
```
front_img: <front_side_image_file>
back_img: <back_side_image_file>
```

**Option B: Single Side Extraction (One Image)**
```
front_img: <image_file> (OR back_img: <image_file>)
side: "front" | "back"
```

**Response Examples:**

**Complete Extraction Response:**
```json
{
    "success": true,
    "extraction_type": "complete",
    "message": "Complete NID data extracted from both sides",
    "data": {
        "english_name": "John Doe",
        "bengali_name": "জন ডো",
        "father_name": "মাইকেল ডো",
        "mother_name": "সারা ডো",
        "nid_no": "1234567890",
        "date_of_birth": "15/03/1990",
        "address": "ঠিকানা: বাসা/হোল্ডিং: ১২৩, গ্রাম/রাস্তা: মেইন রোড, ডাকঘর: ঢাকা",
        "blood_group": "A+"
    }
}
```

**Single Side Extraction Response:**
```json
{
    "success": true,
    "extraction_type": "single",
    "side": "front",
    "message": "Data extracted from front side",
    "data": {
        "english_name": "John Doe",
        "bengali_name": "জন ডো",
        "father_name": "মাইকেল ডো",
        "mother_name": "সারা ডো",
        "nid_no": "1234567890",
        "date_of_birth": "15/03/1990",
        "confidence": "high"
    }
}
```

### 4. NID Single Side Extraction
**POST** `/nid-single-extraction`

Dedicated endpoint for extracting data from a single side of an NID card.

**Request:**
```
image: <image_file>
side: "front" | "back"
```

**Response:**
```json
{
    "success": true,
    "extraction_type": "single",
    "side": "front",
    "message": "Data extracted from front side",
    "data": {
        "english_name": "John Doe",
        "bengali_name": "জন ডো",
        "father_name": "মাইকেল ডো",
        "mother_name": "সারা ডো",
        "nid_no": "1234567890",
        "date_of_birth": "15/03/1990",
        "confidence": "high"
    }
}
```

## Usage Examples

### cURL Examples

**Health Check:**
```bash
curl -X GET http://localhost:8000/api/health
```

**NID Service Health Check:**
```bash
curl -X GET http://localhost:8000/api/nid-health
```

**Complete NID Extraction (Both Images):**
```bash
curl -X POST http://localhost:8000/api/nid-data-extraction \
  -H "Authorization: Basic YWRtaW46cGFzc3dvcmQxMjM=" \
  -F "front_img=@front_side.jpg" \
  -F "back_img=@back_side.jpg"
```

**Single Side Extraction (Front):**
```bash
curl -X POST http://localhost:8000/api/nid-data-extraction \
  -H "Authorization: Basic YWRtaW46cGFzc3dvcmQxMjM=" \
  -F "front_img=@front_side.jpg" \
  -F "side=front"
```

**Single Side Extraction (Back):**
```bash
curl -X POST http://localhost:8000/api/nid-data-extraction \
  -H "Authorization: Basic YWRtaW46cGFzc3dvcmQxMjM=" \
  -F "back_img=@back_side.jpg" \
  -F "side=back"
```

**Dedicated Single Side Endpoint:**
```bash
curl -X POST http://localhost:8000/api/nid-single-extraction \
  -H "Authorization: Basic YWRtaW46cGFzc3dvcmQxMjM=" \
  -F "image=@front_side.jpg" \
  -F "side=front"
```

### Python Examples

```python
import requests

# Base URL and authentication
base_url = "http://localhost:8000/api"
auth = ("admin", "password123")

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# NID service health check
response = requests.get(f"{base_url}/nid-health")
print(response.json())

# Complete extraction with both images
with open("front_side.jpg", "rb") as front_f, open("back_side.jpg", "rb") as back_f:
    files = {
        "front_img": front_f,
        "back_img": back_f
    }
    response = requests.post(f"{base_url}/nid-data-extraction", 
                           files=files, auth=auth)
    print(response.json())

# Single side extraction (front)
with open("front_side.jpg", "rb") as f:
    files = {"front_img": f}
    data = {"side": "front"}
    response = requests.post(f"{base_url}/nid-data-extraction", 
                           files=files, data=data, auth=auth)
    print(response.json())

# Single side extraction (back)
with open("back_side.jpg", "rb") as f:
    files = {"back_img": f}
    data = {"side": "back"}
    response = requests.post(f"{base_url}/nid-data-extraction", 
                           files=files, data=data, auth=auth)
    print(response.json())

# Using dedicated single side endpoint
with open("front_side.jpg", "rb") as f:
    files = {"image": f}
    data = {"side": "front"}
    response = requests.post(f"{base_url}/nid-single-extraction", 
                           files=files, data=data, auth=auth)
    print(response.json())
```

## Smart Extraction Logic

The `/nid-data-extraction` endpoint automatically determines the extraction type:

1. **Complete Extraction**: When both `front_img` and `back_img` are provided
   - Extracts data from both sides
   - Returns comprehensive NID information
   - Uses `extract_complete_nid_data()` method

2. **Single Side Extraction**: When only one image is provided
   - Automatically detects which side the image represents
   - Uses the `side` parameter to determine processing logic
   - Uses `extract_from_single_side()` method
   - Returns data specific to that side

## Error Responses

All endpoints return consistent error responses:

```json
{
    "error": "Error description",
    "usage": {
        "complete_extraction": "Send both front_img and back_img",
        "single_extraction": "Send either front_img or back_img with side parameter"
    }
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (authentication required)
- `500` - Internal Server Error

## Supported Image Formats
- PNG
- JPEG/JPG
- BMP
- TIFF
- WebP

## Notes

1. **Image Quality**: For best results, use clear, high-resolution images with good lighting.
2. **File Size**: Large images are automatically resized and optimized for processing.
3. **Processing Time**: Extraction typically takes 5-30 seconds depending on image complexity and server performance.
4. **Temporary Files**: Images are temporarily saved and automatically cleaned up after processing.
5. **Error Handling**: The API provides detailed error messages and usage instructions.
6. **Side Parameter**: Always required for single side extraction, defaults to "front" if not specified.

## Configuration

The NID service can be configured through environment variables:

- `LLM_BASE_URL`: Ollama service URL (default: http://localhost:11434/api/generate)
- `LLM_MODEL`: Vision Language Model to use (default: qwen2.5vl:7b)
- `LLM_TIMEOUT`: Request timeout in seconds (default: 300)
- `IMAGE_MAX_SIZE`: Maximum image dimension (default: 1024)
- `IMAGE_JPEG_QUALITY`: JPEG quality for processed images (default: 400)
