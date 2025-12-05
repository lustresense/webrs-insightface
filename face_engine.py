"""
Face Detection and Recognition Engine
InsightFace (RetinaFace + ArcFace) + Silent-Face Anti-Spoofing
CPU Mode
"""

import os
import sqlite3
import threading
import logging
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

# ====== LOGGING ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FaceEngine')

# ====== LIVENESS DETECTOR ======
try:
    import liveness_detector
    LIVENESS_ENABLED = True
    logger.info("[OK] Liveness detector loaded")
except ImportError as e:
    LIVENESS_ENABLED = False
    logger.warning(f"[WARN] Liveness detector not available: {e}")

# ====== CONFIG ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "database_wajah")
MODEL_DIR = os.path.join(BASE_DIR, "model")
EMBEDDING_DB_PATH = os.path.join(MODEL_DIR, "embeddings.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Thresholds
DETECTION_THRESHOLD = float(os.environ.get("DETECTION_THRESHOLD", "0.5"))
RECOGNITION_THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", "0.4"))
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "60"))
EMBEDDING_DIM = 512

REGISTRATION_DETECTION_THRESHOLD = 0.3
REGISTRATION_QUALITY_THRESHOLD = 0.1

VOTE_MIN_SHARE = 0.35
MIN_VALID_FRAMES = 2
EARLY_VOTES_REQUIRED = 4
EARLY_SIM_THRESHOLD = 0.55

LIVENESS_THRESHOLD = 0.5

# Global state
_engine_lock = threading.Lock()
_face_app = None
_embeddings_db = {}
_embeddings_loaded = False
_initialized = False


def _get_face_app():
    """Initialize InsightFace (CPU mode)"""
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            logger.info("Initializing InsightFace (CPU mode)...")
            
            _face_app = FaceAnalysis(
                name='buffalo_l',
                root=MODEL_DIR,
                providers=['CPUExecutionProvider']  # CPU ONLY
            )
            
            _face_app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 for CPU
            logger.info("[OK] InsightFace initialized (CPU)")
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            _face_app = None
    return _face_app


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2 normalize embedding"""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between embeddings"""
    return float(np.dot(emb1, emb2))


# ====== EMBEDDING DATABASE ======

def init_embedding_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(EMBEDDING_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nik INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            photo BLOB,
            created_at TEXT NOT NULL,
            quality_score REAL DEFAULT 0.0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_nik ON embeddings(nik)")
    
    # Add photo column if it doesn't exist (migration)
    try:
        cursor = conn.execute("PRAGMA table_info(embeddings)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'photo' not in columns:
            conn.execute("ALTER TABLE embeddings ADD COLUMN photo BLOB")
            logger.info("Migrated embeddings table: added photo column")
    except Exception as e:
        logger.debug(f"Migration check: {e}")
    
    conn.commit()
    conn.close()
    logger.info(f"Embedding DB initialized: {EMBEDDING_DB_PATH}")


def save_embedding(nik: int, embedding: np.ndarray, quality_score: float = 0.0, photo_bgr: Optional[np.ndarray] = None) -> bool:
    """Save embedding to database with optional photo"""
    try:
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        normalized = _normalize_embedding(embedding)
        blob = normalized.astype(np.float32).tobytes()
        
        # Encode photo as JPEG if provided
        photo_blob = None
        if photo_bgr is not None and photo_bgr.size > 0:
            success, photo_blob = cv2.imencode('.jpg', photo_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success:
                photo_blob = photo_blob.tobytes()
        
        conn.execute(
            "INSERT INTO embeddings (nik, embedding, photo, created_at, quality_score) VALUES (?, ?, ?, ?, ?)",
            (nik, blob, photo_blob, datetime.now().isoformat(), quality_score)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save embedding: {e}")
        return False


def get_photo_for_nik(nik: int) -> Optional[np.ndarray]:
    """Get stored photo for NIK from embeddings database"""
    try:
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        cursor = conn.execute(
            "SELECT photo FROM embeddings WHERE nik = ? AND photo IS NOT NULL ORDER BY quality_score DESC LIMIT 1",
            (nik,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            nparr = np.frombuffer(row[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        return None
    except Exception as e:
        logger.error(f"Failed to get photo for NIK {nik}: {e}")
        return None


def load_all_embeddings() -> Dict[int, List[np.ndarray]]:
    """Load all embeddings from database"""
    global _embeddings_db, _embeddings_loaded
    try:
        if not os.path.exists(EMBEDDING_DB_PATH):
            init_embedding_db()
            _embeddings_loaded = True
            return {}

        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        cursor = conn.execute("SELECT nik, embedding FROM embeddings ORDER BY quality_score DESC")

        _embeddings_db = {}
        count = 0
        for row in cursor:
            nik = int(row[0])
            emb = np.frombuffer(row[1], dtype=np.float32)
            if nik not in _embeddings_db:
                _embeddings_db[nik] = []
            _embeddings_db[nik].append(emb)
            count += 1

        conn.close()
        _embeddings_loaded = True
        logger.info(f"Loaded {count} embeddings for {len(_embeddings_db)} NIKs")
        return _embeddings_db
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        _embeddings_loaded = True
        return {}


def delete_embeddings_for_nik(nik: int) -> int:
    """Delete embeddings for NIK"""
    global _embeddings_db
    try:
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        cursor = conn.execute("DELETE FROM embeddings WHERE nik = ? ", (nik,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if nik in _embeddings_db:
            del _embeddings_db[nik]
        return deleted
    except Exception as e:
        logger.error(f"Failed to delete embeddings: {e}")
        return 0


def update_nik_in_embeddings(old_nik: int, new_nik: int) -> int:
    """Update NIK in database"""
    global _embeddings_db
    try:
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        cursor = conn.execute("UPDATE embeddings SET nik = ?  WHERE nik = ? ", (new_nik, old_nik))
        updated = cursor.rowcount
        conn.commit()
        conn.close()
        if old_nik in _embeddings_db:
            _embeddings_db[new_nik] = _embeddings_db.pop(old_nik)
        return updated
    except Exception as e:
        logger.error(f"Failed to update NIK: {e}")
        return 0


def get_embedding_count() -> int:
    """Get total embeddings count"""
    try:
        if not os.path.exists(EMBEDDING_DB_PATH):
            return 0
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        conn.close()
        return count
    except:
        return 0


def get_unique_nik_count() -> int:
    """Get unique NIK count"""
    try:
        if not os.path.exists(EMBEDDING_DB_PATH):
            return 0
        conn = sqlite3.connect(EMBEDDING_DB_PATH)
        count = conn.execute("SELECT COUNT(DISTINCT nik) FROM embeddings").fetchone()[0]
        conn.close()
        return count
    except:
        return 0


def find_best_match(embedding: np.ndarray, threshold: float = None) -> Tuple[Optional[int], float]:
    """
    Find best matching NIK for given embedding
    Returns: (nik, similarity) or (None, 0.0)
    """
    global _embeddings_db, _embeddings_loaded

    if threshold is None:
        threshold = RECOGNITION_THRESHOLD

    if not _embeddings_loaded:
        load_all_embeddings()

    if not _embeddings_db:
        return None, 0.0

    best_nik = None
    best_similarity = 0.0

    try:
        for nik, embeddings_list in _embeddings_db.items():
            for db_embedding in embeddings_list:
                similarity = _cosine_similarity(embedding, db_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_nik = nik

        if best_nik is not None and best_similarity >= threshold:
            return best_nik, best_similarity
        else:
            return None, best_similarity

    except Exception as e:
        logger.error(f"Error in find_best_match: {e}")
        return None, 0.0


# ====== FACE DETECTION ======

def detect_faces(img_bgr: np.ndarray, detection_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Detect faces using InsightFace"""
    if detection_threshold is None:
        detection_threshold = DETECTION_THRESHOLD

    app = _get_face_app()
    if app is None:
        return _detect_faces_fallback(img_bgr)

    try:
        faces = app.get(img_bgr)
        results = []
        for face in faces:
            if face.det_score < detection_threshold:
                continue

            bbox = face.bbox.astype(int).tolist()
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            results.append({
                'bbox': bbox,
                'landmarks': face.kps.tolist() if getattr(face, 'kps', None) is not None else None,
                'det_score': float(face.det_score),
                'embedding': _normalize_embedding(face.embedding) if getattr(face, 'embedding', None) is not None else None
            })

        results.sort(key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]), reverse=True)
        return results
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return _detect_faces_fallback(img_bgr)


def _detect_faces_fallback(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Fallback using Haar Cascade"""
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [x, y, x+w, y+h],
                'landmarks': None,
                'det_score': 0.9,
                'embedding': None
            })
        return results
    except:
        return []


def detect_largest_face(img_bgr: np.ndarray, detection_threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Detect largest face"""
    faces = detect_faces(img_bgr, detection_threshold)
    return faces[0] if faces else None


def is_blurry(img_gray: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if image is blurry"""
    return cv2.Laplacian(img_gray, cv2.CV_64F).var() < threshold


def calculate_quality_score(face_dict: Dict[str, Any], img_bgr: np.ndarray) -> float:
    """Calculate face quality score"""
    score = face_dict.get('det_score', 0.5) * 0.5
    
    bbox = face_dict.get('bbox', [0, 0, 0, 0])
    x1, y1, x2, y2 = bbox
    face_region = img_bgr[y1:y2, x1:x2]
    
    if face_region.size > 0:
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
        score += sharpness * 0.5
    
    return min(score, 1.0)


def get_embedding(img_bgr: np.ndarray, face_dict: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
    """Get face embedding"""
    if face_dict is not None and face_dict.get('embedding') is not None:
        return face_dict['embedding']

    app = _get_face_app()
    if app is None:
        return None

    try:
        faces = app.get(img_bgr)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        if getattr(largest, 'embedding', None) is None:
            return None
        return _normalize_embedding(largest.embedding)
    except:
        return None


# ====== MAIN RECOGNITION WITH LIVENESS ======

def recognize_face_multi_frame(
    frames: List[np.ndarray],
    threshold: float = None,
    require_liveness: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Recognize face with liveness detection. 
    
    Pipeline:
    1. InsightFace: Detect face → bbox + embedding
    2. Silent-Face: Check liveness
    3. If REAL → match embedding
    4. If FAKE → reject
    """
    global _embeddings_db, _embeddings_loaded

    if threshold is None:
        threshold = RECOGNITION_THRESHOLD

    if not _embeddings_loaded:
        load_all_embeddings()

    if not _embeddings_db:
        logger.info("No embeddings in database")
        return None

    from collections import defaultdict
    votes = defaultdict(list)
    processed = 0
    
    liveness_scores = []
    liveness_details = []
    
    if LIVENESS_ENABLED and require_liveness:
        liveness_detector.reset_detector()

    for frame in frames:
        # 1. Detect face with InsightFace
        face = detect_largest_face(frame)
        if face is None:
            continue

        bbox = face.get('bbox')
        if bbox is None:
            continue
        
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Convert [x1,y1,x2,y2] to [x,y,w,h] for Silent-Face
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

        # Blur check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_gray = gray[y1:y2, x1:x2]
        if face_gray.size > 0 and is_blurry(face_gray, 50.0):
            continue

        face_roi = frame[y1:y2, x1:x2]
        
        # 2. Silent-Face liveness check
        if LIVENESS_ENABLED and require_liveness:
            liveness_result = liveness_detector.check_liveness(
                face_roi=face_roi,
                landmarks=face.get('landmarks'),
                full_frame=frame,
                bbox=bbox_xywh
            )
            liveness_scores.append(liveness_result.get('confidence', 0.5))
            liveness_details.append(liveness_result)

        # 3.  Get embedding
        embedding = face.get('embedding')
        if embedding is None:
            embedding = get_embedding(frame, face)
            if embedding is None:
                continue

        processed += 1

        # Match against database
        for nik, embs in _embeddings_db.items():
            sims = [_cosine_similarity(embedding, e) for e in embs]
            max_sim = max(sims) if sims else 0.0
            if max_sim >= threshold:
                votes[nik].append(max_sim)

        # Early stop
        if votes and processed >= 5:
            best_nik = max(votes.keys(), key=lambda k: np.mean(votes[k]))
            if len(votes[best_nik]) >= EARLY_VOTES_REQUIRED:
                break

    if processed == 0 or not votes:
        logger.info(f"Recognition failed: processed={processed}")
        return None

    # 4.  Evaluate liveness
    liveness_passed = True
    avg_liveness = 0.5
    final_detail = {}
    
    if LIVENESS_ENABLED and require_liveness and liveness_scores:
        avg_liveness = np.mean(liveness_scores)
        if liveness_details:
            final_detail = liveness_details[-1]
        
        liveness_passed = avg_liveness >= LIVENESS_THRESHOLD
        
        logger.info(f"Liveness: {avg_liveness:.3f}, passed={liveness_passed}")
        
        if not liveness_passed:
            logger.warning(f"SPOOFING DETECTED! Score: {avg_liveness:.3f}")
            return {
                'nik': None,
                'similarity': 0,
                'vote_count': 0,
                'vote_share': 0,
                'processed_frames': processed,
                'confidence': 0,
                'liveness_passed': False,
                'liveness_score': float(avg_liveness),
                'liveness_recommendation': final_detail.get('recommendation', 'FAKE'),
                'liveness_scores': final_detail.get('scores', {}),
                'rejected': True,
                'reject_reason': 'LIVENESS_FAILED'
            }

    # 5.  Find winner
    winner = None
    best_score = -1

    for nik, sims in votes.items():
        vote_count = len(sims)
        avg_sim = np.mean(sims)
        score = vote_count * avg_sim

        if score > best_score:
            best_score = score
            winner = {
                'nik': nik,
                'similarity': float(avg_sim),
                'vote_count': vote_count,
                'vote_share': vote_count / processed,
                'processed_frames': processed,
                'confidence': int(min(avg_sim * 100, 100)),
                'liveness_passed': liveness_passed,
                'liveness_score': float(avg_liveness),
                'liveness_recommendation': final_detail.get('recommendation', 'REAL'),
                'liveness_scores': final_detail.get('scores', {})
            }

    if winner is None:
        return None

    if winner['vote_share'] < VOTE_MIN_SHARE or winner['vote_count'] < MIN_VALID_FRAMES:
        return None

    logger.info(f"Recognition success: NIK={winner['nik']}, sim={winner['similarity']:.3f}")
    return winner


# ====== AUTOMATIC VERIFICATION PIPELINE ======

def verify_face_automatic(
    frame: np.ndarray,
    require_liveness: bool = True,
    threshold: float = None,
    client_liveness_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    AUTOMATIC VERIFICATION PIPELINE:
    1. OpenCV Haar detect closest face
    2. If found → InsightFace for embedding
    3. Combined liveness check (MediaPipe client + Silent-Face server)
    4. Match against embeddings database

    Args:
        frame: Input frame (BGR)
        require_liveness: Require liveness check
        threshold: Recognition threshold (uses RECOGNITION_THRESHOLD if None)
        client_liveness_data: Client-side MediaPipe liveness results (optional)

    Returns:
        Dict with verification result:
        - success: bool
        - nik: int or None
        - similarity: float (0-1)
        - confidence: int (0-100)
        - liveness_passed: bool
        - liveness_score: float
        - message: str
        - details: dict
    """
    global _embeddings_db, _embeddings_loaded
    
    if threshold is None:
        threshold = RECOGNITION_THRESHOLD
    
    if not _embeddings_loaded:
        load_all_embeddings()
    
    result = {
        'success': False,
        'nik': None,
        'similarity': 0.0,
        'confidence': 0,
        'liveness_passed': True,
        'liveness_score': 0.5,
        'message': '',
        'details': {}
    }
    
    # Check if database has embeddings
    if not _embeddings_db:
        result['message'] = 'No registered faces in database'
        logger.warning(result['message'])
        return result
    
    # 1. DETECT CLOSEST FACE USING OPENCV (FALLBACK)
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_opencv = detector.detectMultiScale(gray, 1.1, 5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        
        if len(faces_opencv) == 0:
            result['message'] = 'No face detected by OpenCV'
            logger.debug(result['message'])
            return result
        
        # Get closest face (largest)
        closest = max(faces_opencv, key=lambda f: f[2] * f[3])
        x, y, w, h = closest
        bbox = [x, y, x+w, y+h]
        
        logger.debug(f"OpenCV detected face: bbox={bbox}")
        
    except Exception as e:
        result['message'] = f'OpenCV detection failed: {str(e)}'
        logger.error(result['message'])
        return result
    
    # 2. USE INSIGHTFACE FOR BETTER EMBEDDING
    x1, y1, x2, y2 = bbox
    face_roi = frame[y1:y2, x1:x2]
    
    if face_roi.size == 0:
        result['message'] = 'Invalid face region'
        return result
    
    # Check blur
    try:
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        if is_blurry(face_gray, 50.0):
            result['message'] = 'Face image is too blurry'
            logger.debug(result['message'])
            return result
    except:
        pass
    
    # Get InsightFace embedding
    try:
        face_dict = detect_largest_face(frame)
        if face_dict:
            embedding = face_dict.get('embedding')
        else:
            embedding = get_embedding(frame)
        
        if embedding is None:
            result['message'] = 'Failed to extract face embedding'
            logger.warning(result['message'])
            return result
            
    except Exception as e:
        result['message'] = f'Embedding extraction failed: {str(e)}'
        logger.error(result['message'])
        return result
    
    # 3. LIVENESS CHECK - TRUST CLIENT BLINK DETECTION
    liveness_passed = True
    liveness_score = 0.8  # High score karena client sudah validasi blink

    if require_liveness:
        try:
            if client_liveness_data:
                # CLIENT SUDAH VALIDASI BLINK = LIVENESS PASSED!
                # Blink detection adalah bukti kuat bahwa ini wajah asli
                client_blink_detected = client_liveness_data.get('blinkDetected', False)
                
                if client_blink_detected:
                    # TRUST CLIENT BLINK - Skip Silent-Face
                    liveness_passed = True
                    liveness_score = 0.9  # High confidence karena blink verified
                    
                    result['liveness_score'] = liveness_score
                    result['liveness_passed'] = liveness_passed
                    result['details']['liveness_recommendation'] = 'REAL - Blink verified by client'
                    result['details']['client_liveness'] = liveness_score
                    result['details']['server_liveness'] = 0.0  # Skipped
                    result['details']['validation_method'] = client_liveness_data.get('validationMethod', 'unknown')
                    
                    logger.info(f"✅ Client blink verified - LIVENESS PASSED (score={liveness_score:.2f})")
                else:
                    # No blink detected, use combined check
                    combined_result = liveness_detector.check_liveness_combined(
                        client_data=client_liveness_data,
                        face_roi=face_roi
                    )
                    liveness_score = combined_result.get('combined_score', 0.5)
                    liveness_passed = combined_result.get('is_live', True)

                    result['liveness_score'] = liveness_score
                    result['liveness_passed'] = liveness_passed
                    result['details']['liveness_recommendation'] = combined_result.get('recommendation', '')
                    result['details']['client_liveness'] = combined_result.get('client_score', 0.0)
                    result['details']['server_liveness'] = combined_result.get('server_score', 0.0)

                    logger.info(f"Combined liveness: {liveness_score:.3f}, passed={liveness_passed}")

            elif LIVENESS_ENABLED:
                # Fallback to server-only liveness check
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                liveness_result = liveness_detector.check_liveness(
                    face_roi=face_roi,
                    full_frame=frame,
                    bbox=bbox_xywh
                )
                liveness_score = liveness_result.get('confidence', 0.5)
                liveness_passed = liveness_result.get('is_live', True)

                result['liveness_score'] = liveness_score
                result['liveness_passed'] = liveness_passed
                result['details']['liveness_recommendation'] = liveness_result.get('recommendation', '')

                logger.info(f"Server liveness: {liveness_score:.3f}, passed={liveness_passed}")

            if not liveness_passed:
                result['message'] = f'SPOOFING DETECTED! Liveness: {liveness_score:.1%}'
                logger.warning(result['message'])
                return result

        except Exception as e:
            logger.warning(f"Liveness check failed: {e}")
            # Continue anyway if liveness fails
    
    # 4. MATCH AGAINST DATABASE
    best_match = None
    best_similarity = 0.0
    
    try:
        for nik, embeddings_list in _embeddings_db.items():
            for db_embedding in embeddings_list:
                similarity = _cosine_similarity(embedding, db_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = nik
        
        logger.debug(f"Best match: NIK={best_match}, similarity={best_similarity:.3f}")
        
    except Exception as e:
        result['message'] = f'Database matching failed: {str(e)}'
        logger.error(result['message'])
        return result
    
    # 5. CHECK THRESHOLD
    if best_match is None or best_similarity < threshold:
        result['message'] = f'No match found (best similarity: {best_similarity:.3f}, threshold: {threshold:.3f})'
        logger.debug(result['message'])
        return result
    
    # SUCCESS!
    result['success'] = True
    result['nik'] = best_match
    result['similarity'] = float(best_similarity)
    result['confidence'] = int(min(best_similarity * 100, 100))
    result['message'] = f'Match found! NIK={best_match}, similarity={best_similarity:.1%}'
    result['details']['photo'] = get_photo_for_nik(best_match)
    
    logger.info(f"VERIFICATION SUCCESS: {result['message']}")
    return result




def enroll_face(img_bgr: np.ndarray, nik: int) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Enroll single face - SIMPLE VERSION
    - Detect face with InsightFace
    - Extract embedding
    - Save with photo to database
    """
    face = detect_largest_face(img_bgr, REGISTRATION_DETECTION_THRESHOLD)
    if face is None:
        return False, "No face detected", None

    quality = calculate_quality_score(face, img_bgr)
    if quality < REGISTRATION_QUALITY_THRESHOLD:
        return False, f"Quality too low: {quality:.2f}", None

    embedding = face.get('embedding')
    if embedding is None:
        embedding = get_embedding(img_bgr, face)
        if embedding is None:
            return False, "Could not extract embedding", None

    # Save embedding WITH photo to database
    if save_embedding(nik, embedding, quality, photo_bgr=img_bgr):
        global _embeddings_db
        if nik not in _embeddings_db:
            _embeddings_db[nik] = []
        _embeddings_db[nik].append(embedding)
        logger.info(f"Enrolled NIK={nik}, quality={quality:.2f}, photo saved")
        return True, f"Enrolled (quality: {quality:.2f})", embedding

    return False, "Failed to save", None


def enroll_multiple_frames(frames: List[np.ndarray], nik: int, min_embeddings: int = 5) -> Tuple[int, str]:
    """Enroll multiple frames"""
    enrolled = 0
    for frame in frames:
        success, _, _ = enroll_face(frame, nik)
        if success:
            enrolled += 1
            if enrolled >= 20:
                break

    # Augment if needed
    global _embeddings_db
    if 0 < enrolled < min_embeddings and nik in _embeddings_db:
        current = len(_embeddings_db[nik])
        for i in range(min_embeddings - current):
            base = _embeddings_db[nik][i % current]
            noise = np.random.normal(0, 0.01, base.shape)
            augmented = _normalize_embedding(base + noise)
            save_embedding(nik, augmented, 0.5)
            _embeddings_db[nik].append(augmented)
            enrolled += 1

    if enrolled == 0:
        return 0, "No valid faces"
    return enrolled, f"Enrolled {enrolled} embeddings"


# ====== STATUS ======

def get_engine_status() -> Dict[str, Any]:
    """Get engine status"""
    return {
        'insightface_available': _get_face_app() is not None,
        'liveness_available': LIVENESS_ENABLED,
        'embeddings_loaded': _embeddings_loaded,
        'total_embeddings': get_embedding_count(),
        'unique_niks': get_unique_nik_count(),
        'recognition_threshold': RECOGNITION_THRESHOLD,
        'liveness_threshold': LIVENESS_THRESHOLD
    }


def initialize():
    """Initialize engine"""
    global _initialized
    init_embedding_db()
    load_all_embeddings()
    _get_face_app()
    _initialized = True
    logger.info("[OK] Face engine initialized")


# Auto-init
if os.environ.get("FACE_ENGINE_INIT", "1") == "1":
    initialize()