# Automatic Face Verification System

## Overview
Sistem verifikasi wajah otomatis yang sudah di-implement dengan pipeline lengkap:

```
1. OpenCV Haar Cascade → Detect wajah terdekat
2. InsightFace → Extract embedding berkualitas tinggi  
3. Silent-Face Anti-Spoofing → Liveness check (Real vs Fake)
4. Database Match → Cocokkan dengan embeddings terdaftar
```

## Key Features

### ✅ Automatic Verification Pipeline
- **Function**: `face_engine.verify_face_automatic(frame, require_liveness=True, threshold=None)`
- **Input**: Single frame (BGR image)
- **Output**: Dict with verification result
  - `success`: bool - Verifikasi berhasil?
  - `nik`: int - NIK yang cocok
  - `similarity`: float 0-1 - Tingkat kesamaan
  - `confidence`: int 0-100 - Confidence percentage
  - `liveness_passed`: bool - Passed liveness check?
  - `liveness_score`: float 0-1 - Liveness confidence
  - `message`: str - Result message
  - `details`: dict - Additional details

### ✅ Photo Storage in Database
- **Feature**: Foto disimpan langsung di embeddings database
- **Function**: `save_embedding(nik, embedding, quality_score, photo_bgr)`
- **Retrieval**: `get_photo_for_nik(nik)` - Get stored photo
- **Format**: JPEG 90% quality compression
- **Migration**: Automatic schema upgrade for existing databases

### ✅ Simplified Registration  
- **Function**: `enroll_face(img_bgr, nik)`
- **Pipeline**:
  1. Detect face dengan InsightFace
  2. Calculate quality score
  3. Extract embedding
  4. Save embedding + photo to database
- **Database**: Semua data langsung ke embeddings.db (unified)

### ✅ Automatic Verification Endpoint
- **Route**: `/api/recognize` (POST)
- **Improvement**: Now uses `verify_face_automatic()` for per-frame verification
- **Pipeline**:
  1. Process each frame dengan automatic verification
  2. Early exit on first successful match
  3. Return matched patient data + confidence scores

## API Changes

### Old vs New `/api/recognize`

**Old (Multi-Frame Heavy Processing)**:
```
recognize_face_multi_frame(frames, require_liveness=True)
- Memproses semua frame secara batch
- Complex voting system
```

**New (Automatic Per-Frame)**:
```
verify_face_automatic(frame, require_liveness=True)
- Simple per-frame verification
- OpenCV fast detection → InsightFace embedding → Liveness → Match
- Early exit on first success
- Clean result structure
```

## Response Format

### Successful Verification
```json
{
  "ok": true,
  "found": true,
  "nik": 5323600000000122,
  "name": "John Doe",
  "dob": "1990-01-15",
  "address": "Jl. Main St",
  "age": "34",
  "similarity": 0.95,
  "confidence": 95,
  "engine": "insightface_auto",
  "liveness_passed": true,
  "liveness_score": 0.87,
  "liveness_recommendation": "REAL - Score: 87%",
  "msg": "✓ Verifikasi berhasil! John Doe"
}
```

### Spoofing Detected
```json
{
  "ok": true,
  "found": false,
  "spoofing_detected": true,
  "liveness_score": 0.2,
  "liveness_recommendation": "FAKE - Detected as photo/screen",
  "msg": "⚠️ SPOOFING TERDETEKSI! Gunakan wajah asli, bukan foto."
}
```

### No Match
```json
{
  "ok": true,
  "found": false,
  "msg": "Wajah tidak dikenali."
}
```

## Code Changes Summary

### face_engine.py

1. **Database Schema Migration**
   - Added `photo BLOB` column to embeddings table
   - Auto-migration for existing databases

2. **New Functions**
   ```python
   # Photo extraction
   get_photo_for_nik(nik: int) -> Optional[np.ndarray]
   
   # Automatic verification
   verify_face_automatic(
       frame: np.ndarray,
       require_liveness: bool = True,
       threshold: float = None
   ) -> Dict[str, Any]
   ```

3. **Enhanced Enrollment**
   - Now saves photo with embedding
   - Single function for registration

### app.py

1. **Simplified `/api/recognize`**
   - Uses new `verify_face_automatic()` function
   - Per-frame processing with early exit
   - Better error handling for liveness failures
   - Cleaner response structure

## Usage Example

### Registration
```python
import cv2
from face_engine import enroll_face

# Read image
img = cv2.imread("person.jpg")

# Register
success, message, embedding = enroll_face(img, nik=5323600000000122)
if success:
    print(f"Registered: {message}")
```

### Verification
```python
import cv2
from face_engine import verify_face_automatic

# Read frame
frame = cv2.imread("verification.jpg")

# Verify
result = verify_face_automatic(frame, require_liveness=True)
if result['success']:
    print(f"Match: NIK={result['nik']}, Similarity={result['similarity']:.1%}")
else:
    print(f"Reason: {result['message']}")
```

### Via API
```bash
# Send frame for verification
curl -X POST http://localhost:5000/api/recognize \
  -F "files[]=@frame1.jpg" \
  -F "files[]=@frame2.jpg"
```

## Thresholds & Configuration

### Recognition Thresholds
```python
RECOGNITION_THRESHOLD = 0.4  # Min similarity for match
LIVENESS_THRESHOLD = 0.5     # Min score for real face
MIN_FACE_SIZE = 60            # Min face dimension
REGISTRATION_QUALITY_THRESHOLD = 0.1
```

### Detection Parameters
```python
REGISTRATION_DETECTION_THRESHOLD = 0.3  # More lenient for registration
DETECTION_THRESHOLD = 0.5                # Standard detection
```

## Database

### Embeddings Database
**Path**: `model/embeddings.db`

**Schema**:
```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    nik INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    photo BLOB,
    created_at TEXT NOT NULL,
    quality_score REAL DEFAULT 0.0
)
```

**Indices**:
- `idx_embeddings_nik` on nik column

## Performance Notes

1. **OpenCV Detection**: Very fast (~5-10ms)
2. **InsightFace Embedding**: ~200-300ms per frame
3. **Liveness Check**: ~100-150ms per frame
4. **Database Match**: ~10-20ms

**Total**: ~400ms per verification (single frame)

## Troubleshooting

### Issue: Photo not extracted
- Check if database has been migrated (run `init_embedding_db()`)
- Verify embeddings were saved with photo parameter

### Issue: Slow verification
- InsightFace CPU mode is slower (~500ms per frame)
- For real-time: Reduce frame size or use GPU

### Issue: High false rejection rate
- Lower `RECOGNITION_THRESHOLD` (e.g., 0.35)
- Ensure good lighting for registration

### Issue: Anti-spoofing false positives
- Adjust `LIVENESS_THRESHOLD` (e.g., 0.45)
- Ensure proper face visibility for liveness check

## Testing

### System Readiness Check
```bash
cd web-face
python -c "
from app import app, FACE_ENGINE
from face_engine import verify_face_automatic, get_photo_for_nik
print('✓ System ready')
print(f'✓ Engine: {FACE_ENGINE}')
"
```

### Test Verification
```bash
python -c "
import face_engine
import cv2
import numpy as np

# Load registered person's frame
frame = cv2.imread('registered_person.jpg')
result = face_engine.verify_face_automatic(frame)
print(f'Success: {result[\"success\"]}')
print(f'NIK: {result[\"nik\"]}')
"
```

## Next Steps

1. ✅ Photo storage in database - DONE
2. ✅ Automatic verification pipeline - DONE
3. ✅ Liveness detection integration - DONE
4. ✅ Simplified registration - DONE
5. Test with real camera feed
6. Monitor performance in production
7. Fine-tune thresholds based on results
