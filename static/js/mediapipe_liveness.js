/**
 * MediaPipe FaceMesh Liveness Detection
 * Client-side liveness analysis using depth variance, micro-movements, and head pose
 */

class MediaPipeLiveness {
    constructor() {
        this.faceMesh = null;
        this.landmarkHistory = [];
        this.maxHistoryFrames = 30; // ~1 second at 30fps
        this.isInitialized = false;
        this.blinkHistory = [];
        this.eyeAspectRatios = [];
    }

    async initialize() {
        if (this.isInitialized) return true;

        try {
            // Load MediaPipe FaceMesh from CDN
            if (typeof FaceMesh === 'undefined') {
                console.error('MediaPipe FaceMesh not loaded. Make sure to include the CDN script.');
                return false;
            }

            // Initialize FaceMesh
            this.faceMesh = new FaceMesh({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
                }
            });

            this.faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            this.isInitialized = true;
            console.log('MediaPipe FaceMesh initialized for liveness detection');
            return true;

        } catch (error) {
            console.error('Failed to initialize MediaPipe FaceMesh:', error);
            return false;
        }
    }

    analyzeDepthVariance(landmarks) {
        // Analyze Z-coordinate variance for depth detection
        // Real faces have depth variation (nose closer, ears farther)
        // Photos are relatively flat

        const keyPoints = [
            landmarks[1],   // nose tip (closest)
            landmarks[234], // left cheek
            landmarks[454], // right cheek
            landmarks[10],  // forehead
            landmarks[152], // chin
            landmarks[127], // left ear area
            landmarks[356]  // right ear area
        ];

        const zValues = keyPoints.map(p => p.z);

        // Calculate depth range
        const zMin = Math.min(...zValues);
        const zMax = Math.max(...zValues);
        const depthRange = zMax - zMin;

        // Normalize depth score (higher = more depth variation = real face)
        // Real face: depthRange > 0.02, Photo: depthRange < 0.01
        const depthScore = Math.min(depthRange / 0.03, 1.0);

        return {
            score: depthScore,
            range: depthRange,
            isReal: depthScore > 0.3
        };
    }

    analyzeMicroMovements() {
        // Analyze landmark position changes across frames
        // Real faces have subtle micro-movements
        // Photos are perfectly still

        if (this.landmarkHistory.length < 2) {
            return { score: 0.5, movement: 0 };
        }

        const current = this.landmarkHistory[this.landmarkHistory.length - 1];
        const previous = this.landmarkHistory[this.landmarkHistory.length - 2];

        // Calculate average movement across key landmarks
        const keyIndices = [1, 234, 454, 10, 152]; // nose, cheeks, forehead, chin
        let totalMovement = 0;

        for (const idx of keyIndices) {
            const dx = current[idx].x - previous[idx].x;
            const dy = current[idx].y - previous[idx].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            totalMovement += distance;
        }

        const avgMovement = totalMovement / keyIndices.length;

        // Normalize movement score (higher = more movement = real face)
        // Real face: avgMovement > 0.0005, Photo: avgMovement < 0.0002
        const movementScore = Math.min(avgMovement / 0.001, 1.0);

        return {
            score: movementScore,
            movement: avgMovement,
            isReal: movementScore > 0.2
        };
    }

    analyzeHeadPose(landmarks) {
        // Analyze head pose stability
        // Real faces have subtle pose changes
        // Photos have very stable pose

        if (this.landmarkHistory.length < 5) {
            return { score: 0.5, poseChange: 0 };
        }

        // Calculate pose using nose, eyes, chin
        const calculatePose = (lm) => {
            const nose = lm[1];
            const leftEye = lm[33];
            const rightEye = lm[263];
            const chin = lm[152];

            // Simple yaw calculation based on eye positions
            const eyeCenterX = (leftEye.x + rightEye.x) / 2;
            const yaw = Math.abs(nose.x - eyeCenterX);

            // Simple pitch calculation
            const pitch = Math.abs(nose.y - chin.y);

            return { yaw, pitch };
        };

        const currentPose = calculatePose(landmarks);
        const poses = this.landmarkHistory.slice(-5).map(lm => calculatePose(lm));

        // Calculate pose variation
        let yawVariance = 0;
        let pitchVariance = 0;

        for (const pose of poses) {
            yawVariance += Math.pow(pose.yaw - currentPose.yaw, 2);
            pitchVariance += Math.pow(pose.pitch - currentPose.pitch, 2);
        }

        yawVariance /= poses.length;
        pitchVariance /= poses.length;

        const poseVariation = Math.sqrt(yawVariance + pitchVariance);

        // Normalize pose score (higher = more pose variation = real face)
        // Real face: poseVariation > 0.001, Photo: poseVariation < 0.0005
        const poseScore = Math.min(poseVariation / 0.002, 1.0);

        return {
            score: poseScore,
            poseChange: poseVariation,
            isReal: poseScore > 0.2
        };
    }

    calculateEyeAspectRatio(landmarks) {
        // Calculate Eye Aspect Ratio (EAR) for blink detection
        // EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        // Left eye landmarks: [33, 160, 158, 133, 153, 144]
        const leftEye = [
            landmarks[33],   // left corner
            landmarks[160],  // top
            landmarks[158],  // bottom
            landmarks[133],  // right corner
            landmarks[153],  // left of top
            landmarks[144]   // right of bottom
        ];

        // Right eye landmarks: [362, 385, 387, 263, 373, 380]
        const rightEye = [
            landmarks[362],  // right corner
            landmarks[385],  // top
            landmarks[387],  // bottom
            landmarks[263],  // left corner
            landmarks[373],  // right of top
            landmarks[380]   // left of bottom
        ];

        const calculateEAR = (eye) => {
            // Vertical distances
            const v1 = Math.sqrt(Math.pow(eye[1].x - eye[5].x, 2) + Math.pow(eye[1].y - eye[5].y, 2));
            const v2 = Math.sqrt(Math.pow(eye[2].x - eye[4].x, 2) + Math.pow(eye[2].y - eye[4].y, 2));

            // Horizontal distance
            const h = Math.sqrt(Math.pow(eye[0].x - eye[3].x, 2) + Math.pow(eye[0].y - eye[3].y, 2));

            return h > 0 ? (v1 + v2) / (2 * h) : 0;
        };

        const leftEAR = calculateEAR(leftEye);
        const rightEAR = calculateEAR(rightEye);

        return (leftEAR + rightEAR) / 2; // Average of both eyes
    }

    analyzeBlinks() {
        // Analyze blink patterns over time
        // Real humans blink periodically, photos don't

        if (this.eyeAspectRatios.length < 10) {
            return { score: 0.5, blinkCount: 0, blinkRate: 0 };
        }

        // Detect blinks (EAR drops below threshold)
        const BLINK_THRESHOLD = 0.25; // EAR threshold for closed eye
        let blinkCount = 0;
        let lastBlinkFrame = -10;

        for (let i = 1; i < this.eyeAspectRatios.length; i++) {
            const currentEAR = this.eyeAspectRatios[i];
            const prevEAR = this.eyeAspectRatios[i-1];

            // Detect blink: EAR drops below threshold and then recovers
            if (prevEAR > BLINK_THRESHOLD && currentEAR <= BLINK_THRESHOLD && (i - lastBlinkFrame) > 5) {
                blinkCount++;
                lastBlinkFrame = i;
            }
        }

        // Calculate blink rate (blinks per second)
        // Assuming 30fps, we have ~1 second of data (30 frames)
        const timeSpan = this.eyeAspectRatios.length / 30; // seconds
        const blinkRate = blinkCount / timeSpan;

        // Normal blink rate: 15-20 blinks per minute = 0.25-0.33 blinks per second
        // Score based on blink rate being in normal range
        let blinkScore = 0;
        if (blinkRate >= 0.1 && blinkRate <= 0.5) {
            blinkScore = 1.0; // Normal blink rate
        } else if (blinkRate >= 0.05 && blinkRate <= 0.6) {
            blinkScore = 0.7; // Acceptable range
        } else if (blinkCount > 0) {
            blinkScore = 0.4; // Some blinking detected
        } else {
            blinkScore = 0.1; // No blinking (likely photo)
        }

        return {
            score: blinkScore,
            blinkCount: blinkCount,
            blinkRate: blinkRate,
            isReal: blinkScore > 0.3
        };
    }

    async runLivenessCheck(videoElement, durationMs = 1500) {
        return new Promise(async (resolve) => {
            if (!this.isInitialized) {
                const initSuccess = await this.initialize();
                if (!initSuccess) {
                    resolve({
                        isLive: false,
                        confidence: 0,
                        message: 'Failed to initialize MediaPipe'
                    });
                    return;
                }
            }

            const results = {
                isLive: false,
                confidence: 0,
                depthScore: 0,
                movementScore: 0,
                poseScore: 0,
                frames: [],
                message: ''
            };

            let frameCount = 0;
            const targetFrames = Math.ceil(durationMs / 33); // ~30fps
            const startTime = Date.now();

            // Set up FaceMesh results handler
            this.faceMesh.onResults((faceMeshResults) => {
                if (faceMeshResults.multiFaceLandmarks && faceMeshResults.multiFaceLandmarks.length > 0) {
                    const landmarks = faceMeshResults.multiFaceLandmarks[0];

                    // Store landmark history
                    this.landmarkHistory.push(landmarks);
                    if (this.landmarkHistory.length > this.maxHistoryFrames) {
                        this.landmarkHistory.shift();
                    }
        
                    // Calculate and store eye aspect ratio for blink detection
                    const ear = this.calculateEyeAspectRatio(landmarks);
                    this.eyeAspectRatios.push(ear);
                    if (this.eyeAspectRatios.length > this.maxHistoryFrames) {
                        this.eyeAspectRatios.shift();
                    }
        
                    // Capture frame for server
                    if (videoElement.videoWidth > 0) {
                        const canvas = document.createElement('canvas');
                        canvas.width = 320;
                        canvas.height = (videoElement.videoHeight / videoElement.videoWidth) * 320;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                        canvas.toBlob((blob) => {
                            if (blob) results.frames.push(blob);
                        }, 'image/jpeg', 0.7);
                    }
        
                    frameCount++;
                }
            });

            // Process frames
            const processFrame = async () => {
                if (Date.now() - startTime >= durationMs || frameCount >= targetFrames) {
                    // Analyze collected data
                    if (this.landmarkHistory.length >= 10) {
                        const depthAnalysis = this.analyzeDepthVariance(this.landmarkHistory[this.landmarkHistory.length - 1]);
                        const movementAnalysis = this.analyzeMicroMovements();
                        const poseAnalysis = this.analyzeHeadPose(this.landmarkHistory[this.landmarkHistory.length - 1]);
                        const blinkAnalysis = this.analyzeBlinks();

                        results.depthScore = depthAnalysis.score;
                        results.movementScore = movementAnalysis.score;
                        results.poseScore = poseAnalysis.score;
                        results.blinkScore = blinkAnalysis.score;

                        // Weighted scoring with blink detection
                        const WEIGHTS = {
                            depth: 0.35,    // Depth variance (most important for 3D detection)
                            movement: 0.25, // Micro-movements
                            pose: 0.20,     // Head pose changes
                            blink: 0.20     // Blink detection (crucial for live vs static)
                        };

                        results.confidence = (
                            results.depthScore * WEIGHTS.depth +
                            results.movementScore * WEIGHTS.movement +
                            results.poseScore * WEIGHTS.pose +
                            results.blinkScore * WEIGHTS.blink
                        );

                        results.isLive = results.confidence >= 0.45;

                        if (results.isLive) {
                            results.message = `REAL - Score: ${(results.confidence * 100).toFixed(1)}% (Blinks: ${blinkAnalysis.blinkCount})`;
                        } else {
                            results.message = `FAKE - Detected as photo/screen (${(results.confidence * 100).toFixed(1)}%)`;
                        }

                        // Add blink details
                        results.blinkCount = blinkAnalysis.blinkCount;
                        results.blinkRate = blinkAnalysis.blinkRate;

                    } else {
                        results.message = 'Insufficient face data';
                    }

                    resolve(results);
                    return;
                }

                // Send video frame to FaceMesh
                await this.faceMesh.send({ image: videoElement });
                requestAnimationFrame(processFrame);
            };

            processFrame();
        });
    }

    reset() {
        this.landmarkHistory = [];
        this.blinkHistory = [];
        this.eyeAspectRatios = [];
    }
}

// Export for use in other scripts
window.MediaPipeLiveness = MediaPipeLiveness;