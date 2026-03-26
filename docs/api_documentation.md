# API Documentation

## Endpoints

### GET /
Home page with upload form

### POST /predict
Upload image and get grade prediction

**Request:**
- Multipart form data with 'file' field containing image

**Response:**
```json
{
    "success": true,
    "grade": "A",
    "confidence": 0.95,
    "confidence_scores": {
        "A": 0.95,
        "B": 0.04,
        "C": 0.01
    }
}
```

### GET /about
About page with project information
