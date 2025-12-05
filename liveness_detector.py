"""
Silent-Face Anti-Spoofing Integration
Menggunakan model MiniFASNet dari infinityglow/Silent-Face-Anti-Spoofing
CPU Mode untuk compatibility
"""

import os
import sys
import cv2
import math
import numpy as np
import logging
from typing import Dict, Optional, Any, List

logger = logging.getLogger('LivenessDetector')

# ====== PATH CONFIG ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SILENT_FACE_DIR = os.path.join(BASE_DIR, "Silent-Face-Anti-Spoofing")

# Add Silent-Face to Python path
if SILENT_FACE_DIR not in sys.path:
    sys.path.insert(0, SILENT_FACE_DIR)

# ====== LIVENESS THRESHOLD ======
LIVENESS_THRESHOLD = 0.5  # Score >= 0.5 = REAL, < 0.5 = FAKE

# ====== GLOBAL STATE ======
_predictor = None
_models_loaded = False


def _get_model_paths() -> List[str]:
    """Get list of anti-spoof model paths"""
    models_dir = os.path.join(SILENT_FACE_DIR, "resources", "anti_spoof_models")
    
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return []
    
    model_files = []
    for f in os.listdir(models_dir):
        if f.endswith('.pth'):
            model_files.append(os.path.join(models_dir, f))
    
    logger.info(f"Found {len(model_files)} anti-spoof models")
    return sorted(model_files)


def _init_predictor():
    """Initialize Silent-Face predictor (CPU mode)"""
    global _predictor, _models_loaded
    
    if _models_loaded:
        return _predictor is not None
    
    try:
        # Import torch first
        import torch
        
        # Force CPU mode
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Import Silent-Face components
        from src.anti_spoof_predict import AntiSpoofPredict
        
        # Initialize with device_id=0 (will use CPU because CUDA disabled)
        _predictor = AntiSpoofPredict(device_id=0)
        _models_loaded = True
        
        logger.info("[OK] Silent-Face Anti-Spoofing initialized (CPU mode)")
        return True
        
    except ImportError as e:
        logger.error(f"[ERROR] Failed to import Silent-Face: {e}")
        logger.error("Make sure you cloned: git clone https://github.com/infinityglow/Silent-Face-Anti-Spoofing.git")
        _models_loaded = True
        return False
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize Silent-Face: {e}")
        _models_loaded = True
        return False


def _crop_face_with_scale(image: np.ndarray, bbox: List[int], scale: float) -> np.ndarray:
    """
    Crop face region with scale factor for anti-spoofing. 
    Silent-Face models expect specific crop scales.
    
    Args:
        image: Full frame (BGR)
        bbox: [x, y, w, h] format
        scale: Scale factor (e.g., 2.7 or 4. 0)
    
    Returns:
        Cropped and scaled face region
    """
    x, y, w, h = bbox
    
    # Calculate center
    cx = x + w // 2
    cy = y + h // 2
    
    # Calculate new size with scale
    new_size = int(max(w, h) * scale)
    half_size = new_size // 2
    
    # Calculate crop coordinates
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(image.shape[1], cx + half_size)
    y2 = min(image.shape[0], cy + half_size)
    
    # Crop
    cropped = image[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return image[y:y+h, x:x+w] if y+h <= image.shape[0] and x+w <= image.shape[1] else image
    
    return cropped


def _parse_model_scale(model_name: str) -> float:
    """Parse scale factor from model filename"""
    # Model names like: "2.7_80x80_MiniFASNetV2.pth" or "4_0_0_80x80_MiniFASNetV1SE.pth"
    try:
        parts = model_name.split('_')
        if '.' in parts[0]:
            return float(parts[0])
        else:
            # Format like 4_0_0 means 4.0
            return float(parts[0])
    except:
        return 2.7  # Default scale


class LivenessDetector:
    """
    Liveness detector using Silent-Face Anti-Spoofing. 
    Detects photo attacks, screen attacks, printed photos. 
    """
    
    def __init__(self):
        self.initialized = _init_predictor()
        self.model_paths = _get_model_paths() if self.initialized else []
        self.last_scores = {}
        
        if self.initialized and self.model_paths:
            logger.info(f"[OK] LivenessDetector ready with {len(self.model_paths)} models")
        else:
            logger.warning("[WARN] LivenessDetector not fully initialized")
    
    def reset(self):
        """Reset state for new session"""
        self.last_scores = {}
    
    def check_liveness(
        self,
        face_roi: np.ndarray,
        landmarks: Optional[List] = None,
        full_frame: Optional[np.ndarray] = None,
        bbox: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Check if face is real or fake. 
        
        Args:
            face_roi: Cropped face region (BGR) - from InsightFace
            landmarks: Facial landmarks (optional)
            full_frame: Full image frame (optional, used if bbox provided)
            bbox: Face bounding box [x, y, w, h] (optional)
        
        Returns:
            Dict with:
            - is_live: bool
            - confidence: float (0-1)
            - scores: dict of per-model scores
            - recommendation: str
        """
        results = {
            'is_live': True,  # Default allow (fail-open)
            'confidence': 0.5,
            'scores': {},
            'details': {},
            'recommendation': ''
        }
        
        # Check if initialized
        if not self.initialized or _predictor is None:
            results['recommendation'] = 'PASS - Anti-spoof not available'
            logger.warning("Silent-Face not initialized, allowing through")
            return results
        
        if not self.model_paths:
            results['recommendation'] = 'PASS - No models found'
            return results
        
        try:
            # Determine image to use
            if full_frame is not None and bbox is not None:
                image = full_frame
                face_bbox = bbox  # [x, y, w, h]
            elif face_roi is not None and face_roi.size > 0:
                h, w = face_roi.shape[:2]
                image = face_roi
                face_bbox = [0, 0, w, h]
            else:
                results['recommendation'] = 'PASS - No valid face image'
                return results
            
            # Check minimum size
            if image.shape[0] < 80 or image.shape[1] < 80:
                results['is_live'] = True
                results['confidence'] = 0.6
                results['recommendation'] = 'PASS - Image too small'
                return results
            
            # Run prediction with each model
            predictions = []
            model_scores = {}
            
            for model_path in self.model_paths:
                try:
                    model_name = os.path.basename(model_path)
                    scale = _parse_model_scale(model_name)
                    
                    # Crop with appropriate scale
                    cropped = _crop_face_with_scale(image, face_bbox, scale)
                    
                    if cropped.size == 0:
                        continue
                    
                    # Resize to 80x80 (model input size)
                    resized = cv2.resize(cropped, (80, 80))
                    
                    # Predict using Silent-Face
                    prediction = _predictor.predict(resized, model_path)
                    
                    # prediction shape: [[fake_prob, real_prob]]
                    if prediction is not None and len(prediction) > 0:
                        real_score = float(prediction[0][1])
                        predictions.append(real_score)
                        model_scores[model_name] = real_score
                        
                except Exception as e:
                    logger.debug(f"Model {os.path.basename(model_path)} error: {e}")
                    continue
            
            if not predictions:
                results['recommendation'] = 'PASS - Prediction failed'
                return results
            
            # Aggregate predictions
            avg_score = np.mean(predictions)
            is_live = avg_score >= LIVENESS_THRESHOLD
            
            results['is_live'] = is_live
            results['confidence'] = float(avg_score)
            results['scores'] = model_scores
            results['details'] = {
                'num_models': len(predictions),
                'avg_score': float(avg_score),
                'min_score': float(min(predictions)),
                'max_score': float(max(predictions)),
                'threshold': LIVENESS_THRESHOLD
            }
            
            if is_live:
                results['recommendation'] = f'REAL - Score: {avg_score:.2%}'
            else:
                results['recommendation'] = f'FAKE - Detected as photo/screen ({avg_score:.2%})'
            
            self.last_scores = results['scores']
            
            logger.info(f"Liveness: {avg_score:.3f} | Live: {is_live} | Models: {len(predictions)}")
            
        except Exception as e:
            logger.error(f"Liveness check error: {e}")
            results['is_live'] = True
            results['confidence'] = 0.5
            results['recommendation'] = f'PASS - Error: {str(e)}'
        
        return results


# ====== SINGLETON ======
_detector_instance = None


def get_detector() -> LivenessDetector:
    """Get singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LivenessDetector()
    return _detector_instance


def check_liveness(
    face_roi: np.ndarray,
    landmarks: Optional[List] = None,
    full_frame: Optional[np.ndarray] = None,
    bbox: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Convenience function for liveness check"""
    detector = get_detector()
    return detector.check_liveness(face_roi, landmarks, full_frame, bbox)


def check_liveness_combined(client_data: Dict[str, Any], face_roi: np.ndarray) -> Dict[str, Any]:
    """
    Combine client-side MediaPipe liveness scores with server-side analysis.

    Args:
        client_data: Dict containing client-side liveness results:
            - depthScore: float (0-1)
            - movementScore: float (0-1)
            - poseScore: float (0-1)
            - clientConfidence: float (0-1)
            - clientIsLive: bool
            - clientMessage: str
        face_roi: Face region for server-side analysis

    Returns:
        Combined liveness result with final decision
    """
    result = {
        'is_live': False,
        'confidence': 0.0,
        'recommendation': '',
        'client_score': client_data.get('clientConfidence', 0.0),
        'server_score': 0.0,
        'combined_score': 0.0,
        'details': {}
    }

    try:
        # Get client scores
        depth_score = client_data.get('depthScore', 0.0)
        movement_score = client_data.get('movementScore', 0.0)
        pose_score = client_data.get('poseScore', 0.0)
        client_confidence = client_data.get('clientConfidence', 0.0)
        client_is_live = client_data.get('clientIsLive', False)

        # Run server-side analysis if face_roi is available
        server_result = None
        if face_roi is not None and face_roi.size > 0:
            try:
                detector = get_detector()
                server_result = detector.check_liveness(face_roi)
                result['server_score'] = server_result.get('confidence', 0.0)
            except Exception as e:
                logger.warning(f"Server-side liveness check failed: {e}")
                result['server_score'] = 0.5  # Neutral score on failure

        # Combine scores with weights
        # Client-side (MediaPipe): 70% weight (more reliable for depth analysis)
        # Server-side (Silent-Face): 30% weight (additional validation)
        CLIENT_WEIGHT = 0.7
        SERVER_WEIGHT = 0.3

        combined_score = (
            client_confidence * CLIENT_WEIGHT +
            result['server_score'] * SERVER_WEIGHT
        )

        result['combined_score'] = combined_score

        # Decision logic
        # If client says FAKE with high confidence, reject immediately
        if client_confidence < 0.3:
            result['is_live'] = False
            result['confidence'] = client_confidence
            result['recommendation'] = f'FAKE - Client detected spoofing ({client_confidence:.1%})'
        # If client says REAL and server agrees, accept
        elif combined_score >= 0.5:
            result['is_live'] = True
            result['confidence'] = combined_score
            result['recommendation'] = f'REAL - Combined score: {combined_score:.1%}'
        # Borderline cases - be conservative
        else:
            result['is_live'] = False
            result['confidence'] = combined_score
            result['recommendation'] = f'FAKE - Low confidence ({combined_score:.1%})'

        # Add detailed breakdown
        result['details'] = {
            'client_depth': depth_score,
            'client_movement': movement_score,
            'client_pose': pose_score,
            'client_overall': client_confidence,
            'server_overall': result['server_score'],
            'weights': {
                'client': CLIENT_WEIGHT,
                'server': SERVER_WEIGHT
            },
            'client_message': client_data.get('clientMessage', ''),
            'server_details': server_result.get('details', {}) if server_result else {}
        }

        logger.info(f"Combined liveness: Client={client_confidence:.3f}, Server={result['server_score']:.3f}, Combined={combined_score:.3f}, Result={result['recommendation']}")

    except Exception as e:
        logger.error(f"Combined liveness check error: {e}")
        result['is_live'] = False
        result['confidence'] = 0.0
        result['recommendation'] = f'ERROR - {str(e)}'

    return result


def reset_detector():
    """Reset detector state"""
    detector = get_detector()
    detector.reset()