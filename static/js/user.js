// ============================================================
// BLINK-GATED FACE RECOGNITION SYSTEM v2.1
// ============================================================
// Pipeline:
// 1. Webcam aktif (client)
// 2. MediaPipe FaceMesh REAL-TIME (mesh VISIBLE)
// 3. Mesh valid → TUNGGU BLINK (EAR detection)
// 4. Blink detected (EAR < 0.1) → Kirim frame ke backend InsightFace
// 5. Backend return hasil verifikasi
// ============================================================

(() => {
  'use strict';
  console.log('🚀 Blink-Gated Face Recognition System v2.1');

  // ============================================================
  // CONFIGURATION
  // ============================================================
  const CONFIG = {
    // EAR Blink Detection - INSTANT TRIGGER
    EAR_THRESHOLD: 0.10,           // Kedip = EAR di bawah 0.1
    EAR_CONSEC_FRAMES: 1,          // 1 FRAME AJA LANGSUNG TRIGGER!
    BLINK_COOLDOWN: 800,           // Cooldown antar blink (ms)
    
    // MediaPipe
    MESH_CONFIDENCE: 0.5,
    
    // Timing
    NEXT_SCAN_DELAY: 10000,
  };

  // MediaPipe FaceMesh Eye Landmark Indices
  const LEFT_EYE = [33, 160, 158, 133, 153, 144];
  const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

  // ============================================================
  // STATE
  // ============================================================
  const STATE = {
    streamReg: null,
    streamVerif: null,
    facingMode: 'user',
    
    faceMesh: null,
    isDetecting: false,
    animationFrameId: null,
    
    // Blink Detection
    earHistory: [],
    closedFrameCount: 0,
    lastBlinkTime: 0,
    isWaitingForBlink: false,
    blinkDetected: false,
    
    currentLandmarks: null,
    meshValid: false,
    
    isProcessing: false,
    activePatient: null,
    nextScanTimer: null,
  };

  // ============================================================
  // DOM ELEMENTS
  // ============================================================
  const $ = (id) => document.getElementById(id);
  
  const pageHome = $('page-home');
  const pageRegistrasi = $('page-registrasi');
  const pagePoli = $('page-poli');
  const pagePoliGateway = $('page-poli-gateway');
  
  const navHome = $('nav-home');
  const navRegistrasi = $('nav-registrasi');
  const navPoli = $('nav-poli');
  const btnHomeRegistrasi = $('btn-home-registrasi');
  const btnHomeKePoli = $('btn-home-ke-poli');
  
  const formRegistrasi = $('form-registrasi');
  const inputNik = $('reg-nik');
  const inputNama = $('reg-nama');
  const inputDob = $('reg-ttl');
  const inputAlamat = $('reg-alamat');
  const videoReg = $('video-reg');
  const statusReg = $('status-reg');
  const countReg = $('count-reg');
  
  const videoVerif = $('video-verif');
  const canvasVerif = $('canvas-verif');
  const btnScan = $('btn-scan');
  const btnNikFallback = $('btn-nik-fallback');
  const verifResult = $('verif-result');
  const verifData = $('verif-data');
  const verifNikBox = $('verif-nik');
  const fallbackNik = $('fallback-nik');
  const btnCariNik = $('btn-cari-nik');
  const statusVerif = $('status-verif');
  const btnLanjutForm = $('btn-lanjut-form');
  const btnDetailData = $('btn-detail-data');
  const focusBox = $('verif-focus-box');
  
  const btnSwitchCamReg = $('btn-switch-cam-reg');
  const btnSwitchCamVerif = $('btn-switch-cam-verif');
  
  const formPoliGateway = $('form-poli-gateway');
  const gwNama = $('gw-nama');
  const gwUmur = $('gw-umur');
  const gwAlamat = $('gw-alamat');
  const gwPoli = $('gw-poli');
  
  const modalAlert = $('modal-alert');
  const alertMessage = $('alert-message');
  const btnAlertOk = $('btn-modal-alert-ok');
  const modalLoading = $('modal-loading');
  const loadingText = $('loading-text');
  const progressInner = $('progress-inner');
  const modalAntrian = $('modal-antrian');
  const antrianPoli = $('antrian-poli');
  const antrianNomor = $('antrian-nomor');
  const btnAntrianTutup = $('btn-modal-antrian-tutup');
  const modalRegisSuccess = $('modal-regis-success');
  const btnModalRegisTutup = $('btn-modal-regis-tutup');
  const btnModalLanjutPoli = $('btn-modal-lanjut-poli');
  const modalCameraPermission = $('modal-camera-permission');
  const btnCameraPermission = $('btn-camera-permission');
  const btnCameraPermissionLater = $('btn-camera-permission-later');
  const cameraStatus = $('camera-status');
  const cameraStatusDot = $('camera-status-dot');
  const cameraStatusText = $('camera-status-text');

  let ctx = null;

  // ============================================================
  // UTILITY FUNCTIONS
  // ============================================================
  
  function showAlert(msg) {
    if (alertMessage) alertMessage.textContent = msg;
    if (modalAlert) modalAlert.classList.remove('hidden');
  }
  
  function showLoading(text) {
    if (loadingText) loadingText.textContent = text;
    if (progressInner) progressInner.style.width = '0%';
    if (modalLoading) modalLoading.classList.remove('hidden');
  }
  
  function hideLoading() {
    if (modalLoading) modalLoading.classList.add('hidden');
  }
  
  function updateProgress(current, total, label) {
    const pct = Math.round((current / total) * 100);
    if (progressInner) progressInner.style.width = pct + '%';
    if (loadingText) loadingText.textContent = `${label} ${current}/${total} (${pct}%)`;
  }
  
  function computeAge(dob) {
    if (!dob) return '-';
    const d = new Date(dob);
    const today = new Date();
    let age = today.getFullYear() - d.getFullYear();
    const m = today.getMonth() - d.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < d.getDate())) age--;
    return `${age} Tahun`;
  }

  function updateStatus(text, type = 'info') {
    if (!statusVerif) return;
    statusVerif.textContent = text;
    
    if (focusBox) {
      focusBox.classList.remove('border-red-600', 'border-yellow-500', 'border-green-500', 'border-primary-500');
      switch(type) {
        case 'waiting': focusBox.classList.add('border-yellow-500'); break;
        case 'success': focusBox.classList.add('border-green-500'); break;
        case 'error': focusBox.classList.add('border-red-600'); break;
        default: focusBox.classList.add('border-primary-500');
      }
    }
  }

  // ============================================================
  // EYE ASPECT RATIO (EAR) CALCULATION
  // ============================================================
  
  function distance(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
  }
  
  function calculateEAR(landmarks, eyeIndices) {
    try {
      const eye = eyeIndices.map(i => landmarks[i]);
      
      // Vertical distances
      const v1 = distance(eye[1], eye[5]);
      const v2 = distance(eye[2], eye[4]);
      
      // Horizontal distance
      const h = distance(eye[0], eye[3]);
      
      if (h === 0) return 0.3; // Default open eye
      
      return (v1 + v2) / (2.0 * h);
    } catch (e) {
      return 0.3;
    }
  }
  
  function getAverageEAR(landmarks) {
    const leftEAR = calculateEAR(landmarks, LEFT_EYE);
    const rightEAR = calculateEAR(landmarks, RIGHT_EYE);
    return (leftEAR + rightEAR) / 2.0;
  }

  // ============================================================
  // BLINK DETECTION (EAR < 0.1)
  // ============================================================
  
  function resetBlinkState() {
    STATE.earHistory = [];
    STATE.closedFrameCount = 0;
    STATE.blinkDetected = false;
    STATE.isWaitingForBlink = false;
  }
  
  function detectBlink(landmarks) {
    const ear = getAverageEAR(landmarks);
    const now = Date.now();
    
    // Save history
    STATE.earHistory.push({ ear, time: now });
    if (STATE.earHistory.length > 30) STATE.earHistory.shift();
    
    // Check cooldown
    if (now - STATE.lastBlinkTime < CONFIG.BLINK_COOLDOWN) {
      console.log('⏳ Cooldown active, skipping...');
      return false;
    }
    
    // SIMPLE: EAR < 0.1 = LANGSUNG TRIGGER!
    const isEyeClosed = ear < CONFIG.EAR_THRESHOLD;
    
    console.log(`👁️ EAR: ${ear.toFixed(3)} | Closed: ${isEyeClosed} | Counter: ${STATE.closedFrameCount}`);
    
    if (isEyeClosed) {
      STATE.closedFrameCount++;
      console.log(`🔴 EYE CLOSED! Frame: ${STATE.closedFrameCount}/${CONFIG.EAR_CONSEC_FRAMES}`);
      
      // 1 FRAME = LANGSUNG TRIGGER!
      if (STATE.closedFrameCount >= CONFIG.EAR_CONSEC_FRAMES) {
        STATE.lastBlinkTime = now;
        console.log(`✅✅✅ BLINK CONFIRMED! TRIGGERING BACKEND! ✅✅✅`);
        return true;
      }
    } else {
      STATE.closedFrameCount = 0;
    }
    
    return false;
  }

  // ============================================================
  // MEDIAPIPE FACEMESH
  // ============================================================
  
  async function initializeFaceMesh() {
    if (STATE.faceMesh) return true;
    
    try {
      if (typeof FaceMesh === 'undefined') {
        console.error('❌ MediaPipe FaceMesh not loaded');
        return false;
      }
      
      STATE.faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
      });
      
      STATE.faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: CONFIG.MESH_CONFIDENCE,
        minTrackingConfidence: CONFIG.MESH_CONFIDENCE
      });
      
      STATE.faceMesh.onResults(onFaceMeshResults);
      
      console.log('✅ MediaPipe FaceMesh initialized');
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize FaceMesh:', error);
      return false;
    }
  }

  // ============================================================
  // FACEMESH RESULTS HANDLER
  // ============================================================
  
  function onFaceMeshResults(results) {
    if (!ctx || !canvasVerif) return;
    
    ctx.clearRect(0, 0, canvasVerif.width, canvasVerif.height);
    
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      STATE.meshValid = false;
      STATE.currentLandmarks = null;
      resetBlinkState();
      
      if (STATE.isDetecting && !STATE.isProcessing) {
        updateStatus('👤 Arahkan wajah ke kamera...', 'info');
      }
      return;
    }
    
    const landmarks = results.multiFaceLandmarks[0];
    STATE.currentLandmarks = landmarks;
    STATE.meshValid = true;
    
    // ALWAYS draw mesh (visible to user)
    drawFaceMesh(landmarks);
    
    // Workflow
    if (!STATE.isProcessing && STATE.isDetecting) {
      if (!STATE.isWaitingForBlink) {
        STATE.isWaitingForBlink = true;
        resetBlinkState();
        console.log('🟡 Started waiting for blink...');
      }
      
      if (!STATE.blinkDetected) {
        const blinked = detectBlink(landmarks);
        
        if (blinked) {
          STATE.blinkDetected = true;
          console.log('🚀🚀🚀 CALLING processVerification() NOW! 🚀🚀🚀');
          updateStatus('✅ Kedip terdeteksi! Memproses...', 'success');
          processVerification();
        } else {
          const ear = getAverageEAR(landmarks);
          updateStatus(`👁️ KEDIPKAN MATA! (EAR: ${ear.toFixed(2)} | Need < 0.10)`, 'waiting');
        }
      }
    }
  }

  // ============================================================
  // DRAW FACE MESH
  // ============================================================
  
  function drawFaceMesh(landmarks) {
    if (!ctx || !canvasVerif) return;
    
    const w = canvasVerif.width;
    const h = canvasVerif.height;
    
    // Draw mesh points (green)
    ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
    landmarks.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x * w, point.y * h, 1, 0, 2 * Math.PI);
      ctx.fill();
    });
    
    // Draw eye landmarks (RED)
    ctx.fillStyle = 'red';
    [...LEFT_EYE, ...RIGHT_EYE].forEach(idx => {
      const point = landmarks[idx];
      ctx.beginPath();
      ctx.arc(point.x * w, point.y * h, 4, 0, 2 * Math.PI);
      ctx.fill();
    });
    
    // Draw eye contours (YELLOW)
    ctx.strokeStyle = 'yellow';
    ctx.lineWidth = 2;
    
    [LEFT_EYE, RIGHT_EYE].forEach(eyeIndices => {
      ctx.beginPath();
      eyeIndices.forEach((idx, i) => {
        const point = landmarks[idx];
        if (i === 0) ctx.moveTo(point.x * w, point.y * h);
        else ctx.lineTo(point.x * w, point.y * h);
      });
      ctx.closePath();
      ctx.stroke();
    });
    
    // Draw bounding box
    let minX = Infinity, minY = Infinity, maxX = 0, maxY = 0;
    landmarks.forEach(point => {
      minX = Math.min(minX, point.x * w);
      minY = Math.min(minY, point.y * h);
      maxX = Math.max(maxX, point.x * w);
      maxY = Math.max(maxY, point.y * h);
    });
    
    const pad = 20;
    ctx.strokeStyle = STATE.isWaitingForBlink ? 'yellow' : 'lime';
    ctx.lineWidth = 3;
    ctx.strokeRect(minX - pad, minY - pad, maxX - minX + pad * 2, maxY - minY + pad * 2);
    
    // Show EAR value
    const ear = getAverageEAR(landmarks);
    const isClosing = ear < CONFIG.EAR_THRESHOLD;
    
    // Background for text
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(5, 5, 220, 90);
    
    // EAR value - HIJAU kalau di bawah threshold
    ctx.fillStyle = isClosing ? '#00ff00' : 'white';
    ctx.font = 'bold 20px Arial';
    ctx.fillText(`EAR: ${ear.toFixed(3)}`, 15, 30);
    
    // Threshold indicator
    ctx.fillStyle = 'yellow';
    ctx.font = '14px Arial';
    ctx.fillText(`Need: < ${CONFIG.EAR_THRESHOLD} (merem)`, 15, 50);
    
    // Progress counter
    ctx.fillStyle = isClosing ? '#00ff00' : 'white';
    ctx.font = 'bold 16px Arial';
    ctx.fillText(`Progress: ${STATE.closedFrameCount}/${CONFIG.EAR_CONSEC_FRAMES} frames`, 15, 75);
    
    // Instruction at bottom
    if (STATE.isWaitingForBlink && !STATE.blinkDetected) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(w/2 - 160, h - 55, 320, 45);
      
      ctx.fillStyle = isClosing ? '#00ff00' : 'yellow';
      ctx.font = 'bold 24px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(isClosing ? '✅ TAHAN MEREM...' : '👁️ PEJAMKAN MATA!', w/2, h - 22);
      ctx.textAlign = 'left';
    }
  }

  // ============================================================
  // DETECTION LOOP
  // ============================================================
  
  function startDetectionLoop() {
    if (STATE.animationFrameId) return;
    
    STATE.isDetecting = true;
    
    const detect = async () => {
      if (!STATE.isDetecting) return;
      
      if (videoVerif && videoVerif.readyState >= 2 && STATE.faceMesh) {
        if (canvasVerif) {
          canvasVerif.width = videoVerif.videoWidth || 640;
          canvasVerif.height = videoVerif.videoHeight || 480;
        }
        
        try {
          await STATE.faceMesh.send({ image: videoVerif });
        } catch (e) {}
      }
      
      STATE.animationFrameId = requestAnimationFrame(detect);
    };
    
    detect();
    console.log('🔄 Detection loop started');
  }
  
  function stopDetectionLoop() {
    STATE.isDetecting = false;
    
    if (STATE.animationFrameId) {
      cancelAnimationFrame(STATE.animationFrameId);
      STATE.animationFrameId = null;
    }
    
    resetBlinkState();
    console.log('🛑 Detection loop stopped');
  }

  // ============================================================
  // CAMERA
  // ============================================================
  
  async function initCamera(videoEl, mode = 'user') {
    if (videoEl.srcObject) {
      videoEl.srcObject.getTracks().forEach(t => t.stop());
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: mode, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });
      
      videoEl.srcObject = stream;
      await videoEl.play();
      
      return stream;
    } catch (e) {
      if (mode === 'environment') return initCamera(videoEl, 'user');
      showAlert('Gagal akses kamera: ' + e.message);
      return null;
    }
  }
  
  async function ensureCamera(type) {
    const status = localStorage.getItem('cameraPermissionStatus');
    if (status !== 'granted') {
      showCameraPermissionModal();
      return false;
    }
    
    if (type === 'verif') {
      if (!STATE.streamVerif || !STATE.streamVerif.active) {
        STATE.streamVerif = await initCamera(videoVerif, STATE.facingMode);
      }
    } else if (type === 'reg') {
      if (!STATE.streamReg || !STATE.streamReg.active) {
        STATE.streamReg = await initCamera(videoReg, STATE.facingMode);
      }
    }
    
    updateCameraStatus('granted');
    return true;
  }
  
  async function switchCamera() {
    STATE.facingMode = STATE.facingMode === 'user' ? 'environment' : 'user';
    
    if (!pageRegistrasi.classList.contains('hidden')) {
      await ensureCamera('reg');
    } else if (!pagePoli.classList.contains('hidden')) {
      stopDetectionLoop();
      await ensureCamera('verif');
      startAutoVerification();
    }
  }

  // ============================================================
  // FRAME CAPTURE
  // ============================================================
  
  function captureFrame(videoEl, width = 400, quality = 0.85) {
    return new Promise((resolve) => {
      if (!videoEl.videoWidth) { resolve(null); return; }
      
      const canvas = document.createElement('canvas');
      const scale = width / videoEl.videoWidth;
      canvas.width = width;
      canvas.height = videoEl.videoHeight * scale;
      
      const c = canvas.getContext('2d');
      c.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', quality);
    });
  }
  
  function captureMultipleFrames(videoEl, count = 15, delay = 100, counterEl = null) {
    return new Promise((resolve) => {
      const frames = [];
      let captured = 0;
      
      const grab = () => {
        if (!videoEl.videoWidth) { requestAnimationFrame(grab); return; }
        
        const canvas = document.createElement('canvas');
        const scale = 400 / videoEl.videoWidth;
        canvas.width = 400;
        canvas.height = videoEl.videoHeight * scale;
        
        const c = canvas.getContext('2d');
        c.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
          frames.push(blob);
          captured++;
          if (counterEl) counterEl.textContent = captured;
          updateProgress(captured, count, 'Foto');
          
          if (captured >= count) resolve(frames);
          else setTimeout(grab, delay);
        }, 'image/jpeg', 0.8);
      };
      
      grab();
    });
  }

  // ============================================================
  // BACKEND VERIFICATION
  // ============================================================
  
  async function processVerification() {
    if (STATE.isProcessing) return;
    
    STATE.isProcessing = true;
    stopDetectionLoop();
    
    const startTime = Date.now();
    showLoading('🔄 Mengirim ke server InsightFace...');
    
    try {
      const frameBlob = await captureFrame(videoVerif);
      if (!frameBlob) throw new Error('Gagal capture frame');
      
      const formData = new FormData();
      formData.append('files[]', frameBlob, 'verify.jpg');
      formData.append('liveness_data', JSON.stringify({
        blinkDetected: true,
        validationMethod: 'blink-gated-workflow',
        earThreshold: CONFIG.EAR_THRESHOLD,
        timestamp: Date.now()
      }));
      
      const response = await fetch('/api/recognize', { method: 'POST', body: formData });
      const result = await response.json();
      hideLoading();
      
      console.log('📥 Backend result:', result);
      
      if (!result.ok || !result.found) {
        if (result.spoofing_detected) {
          updateStatus('⚠️ SPOOFING TERDETEKSI! Gunakan wajah asli.', 'error');
          showAlert('Spoofing terdeteksi! Pastikan menggunakan wajah asli.');
        } else {
          updateStatus('❌ Wajah tidak dikenali. Pastikan sudah terdaftar.', 'error');
        }
        
        setTimeout(() => {
          STATE.isProcessing = false;
          startAutoVerification();
        }, 3000);
        return;
      }
      
      // SUCCESS
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      
      STATE.activePatient = {
        nik: result.nik,
        name: result.name,
        address: result.address,
        dob: result.dob,
        age: result.age
      };
      
      updateStatus('✅ Verifikasi berhasil!', 'success');
      
      if (verifData) {
        verifData.innerHTML = `
          <div class="space-y-2">
            <p><strong>NIK:</strong> <span class="font-mono">${result.nik}</span></p>
            <p><strong>Nama:</strong> ${result.name}</p>
            <p><strong>Tanggal Lahir:</strong> ${result.dob || '-'}</p>
            <p><strong>Umur:</strong> ${result.age}</p>
            <p><strong>Alamat:</strong> ${result.address}</p>
            <p><strong>Waktu:</strong> ${elapsed} detik</p>
            <p><strong>Similarity:</strong> ${(result.similarity * 100).toFixed(1)}%</p>
            <p><strong>Liveness:</strong> ${result.liveness_passed ? '✅ REAL' : '⚠️'} (${(result.liveness_score * 100).toFixed(0)}%)</p>
            
            <div class="bg-green-50 dark:bg-[#1e293b] text-green-800 dark:text-green-300 text-sm px-4 py-2 rounded-lg mt-2 flex items-center gap-2 animate-pulse border dark:border-border">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <span id="next-scan-text">Scan berikutnya dalam 10 detik...</span>
            </div>
          </div>
        `;
      }
      
      if (verifResult) verifResult.classList.remove('hidden');
      startNextScanCountdown();
      
    } catch (error) {
      console.error('Verification error:', error);
      hideLoading();
      updateStatus('❌ Error koneksi server.', 'error');
      showAlert('Error: ' + error.message);
      
      setTimeout(() => {
        STATE.isProcessing = false;
        startAutoVerification();
      }, 2000);
    }
  }

  // ============================================================
  // AUTO VERIFICATION WORKFLOW
  // ============================================================
  
  async function startAutoVerification() {
    console.log('🚀 Starting auto-verification...');
    
    // Stop any existing detection first
    stopDetectionLoop();
    
    if (!STATE.faceMesh) {
      const ok = await initializeFaceMesh();
      if (!ok) {
        updateStatus('❌ Gagal memuat MediaPipe', 'error');
        return;
      }
    }
    
    if (canvasVerif && !ctx) {
      ctx = canvasVerif.getContext('2d');
    }
    
    // RESET ALL BLINK STATES FOR FRESH START
    STATE.isProcessing = false;
    STATE.blinkDetected = false;
    STATE.isWaitingForBlink = false;
    STATE.closedFrameCount = 0;
    STATE.earHistory = [];
    
    // Start fresh detection loop
    startDetectionLoop();
    
    updateStatus('👤 Arahkan wajah ke kamera...', 'info');
  }
  
  function stopAutoVerification() {
    stopDetectionLoop();
    stopNextScanCountdown();
    if (ctx && canvasVerif) {
      ctx.clearRect(0, 0, canvasVerif.width, canvasVerif.height);
    }
  }

  // ============================================================
  // MANUAL VERIFICATION
  // ============================================================
  
  async function performManualVerification() {
    console.log('🔘 Manual verification triggered');
    stopDetectionLoop();
    resetBlinkState();
    STATE.isProcessing = false;
    updateStatus('🔍 Memulai verifikasi manual...', 'info');
    startAutoVerification();
  }

  // ============================================================
  // NEXT SCAN COUNTDOWN
  // ============================================================
  
  function startNextScanCountdown() {
    stopNextScanCountdown();
    
    let seconds = 10;
    
    const updateText = () => {
      const el = $('next-scan-text');
      if (el) el.textContent = `Scan berikutnya dalam ${seconds} detik...`;
    };
    
    updateText();
    
    STATE.nextScanTimer = setInterval(() => {
      seconds--;
      updateText();
      
      if (seconds <= 0) {
        stopNextScanCountdown();
        resetVerification();
      }
    }, 1000);
  }
  
  function stopNextScanCountdown() {
    if (STATE.nextScanTimer) {
      clearInterval(STATE.nextScanTimer);
      STATE.nextScanTimer = null;
    }
  }

  // ============================================================
  // PAGE NAVIGATION
  // ============================================================
  
  function showPage(id) {
    console.log('📄 Showing page:', id);
    
    stopAutoVerification();
    
    [pageHome, pageRegistrasi, pagePoli, pagePoliGateway].forEach(p => {
      if (p) p.classList.add('hidden');
    });
    
    document.querySelectorAll('.nav-button').forEach(b => b.classList.remove('active'));
    
    switch(id) {
      case 'page-home':
        if (pageHome) pageHome.classList.remove('hidden');
        break;
      case 'page-registrasi':
        if (pageRegistrasi) pageRegistrasi.classList.remove('hidden');
        if (navRegistrasi) navRegistrasi.classList.add('active');
        break;
      case 'page-poli':
        if (pagePoli) pagePoli.classList.remove('hidden');
        if (navPoli) navPoli.classList.add('active');
        resetVerification();
        break;
      case 'page-poli-gateway':
        if (pagePoliGateway) pagePoliGateway.classList.remove('hidden');
        if (navPoli) navPoli.classList.add('active');
        break;
    }
  }
  
  function resetVerification() {
    console.log('🔄 Resetting verification for new scan...');
    
    // Hide results
    if (verifResult) verifResult.classList.add('hidden');
    if (verifNikBox) verifNikBox.classList.add('hidden');
    if (verifData) verifData.innerHTML = '';
    
    // RESET ALL STATES
    STATE.isProcessing = false;
    STATE.blinkDetected = false;
    STATE.isWaitingForBlink = false;
    STATE.closedFrameCount = 0;
    STATE.earHistory = [];
    
    stopNextScanCountdown();
    
    // Clear canvas
    if (ctx && canvasVerif) {
      ctx.clearRect(0, 0, canvasVerif.width, canvasVerif.height);
    }
    
    ensureCamera('verif').then(() => {
      if (STATE.streamVerif && !pagePoli.classList.contains('hidden')) {
        console.log('🚀 Starting fresh auto-verification...');
        startAutoVerification();
      }
    });
  }

  // ============================================================
  // CAMERA PERMISSION
  // ============================================================
  
  function showCameraPermissionModal() {
    if (modalCameraPermission) modalCameraPermission.classList.remove('hidden');
  }
  
  function hideCameraPermissionModal() {
    if (modalCameraPermission) modalCameraPermission.classList.add('hidden');
  }
  
  function updateCameraStatus(status) {
    if (!cameraStatus || !cameraStatusDot || !cameraStatusText) return;
    
    if (status === 'granted') {
      cameraStatusDot.className = 'w-2 h-2 rounded-full bg-green-500';
      cameraStatusText.textContent = 'Camera: Siap';
      cameraStatus.className = 'flex items-center gap-2 mb-8 px-3 py-1 rounded-full text-xs font-medium bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 w-fit mx-auto';
    } else {
      cameraStatusDot.className = 'w-2 h-2 rounded-full bg-red-500';
      cameraStatusText.textContent = 'Camera: Belum Izin';
      cameraStatus.className = 'flex items-center gap-2 mb-8 px-3 py-1 rounded-full text-xs font-medium bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 w-fit mx-auto';
    }
  }

  // ============================================================
  // EVENT LISTENERS
  // ============================================================
  
  // Navigation
  if (navHome) navHome.addEventListener('click', () => showPage('page-home'));
  if (navRegistrasi) navRegistrasi.addEventListener('click', () => showPage('page-registrasi'));
  if (navPoli) navPoli.addEventListener('click', () => showPage('page-poli'));
  if (btnHomeRegistrasi) btnHomeRegistrasi.addEventListener('click', () => showPage('page-registrasi'));
  if (btnHomeKePoli) btnHomeKePoli.addEventListener('click', () => showPage('page-poli'));
  
  // Camera switch
  if (btnSwitchCamReg) btnSwitchCamReg.addEventListener('click', (e) => { e.preventDefault(); switchCamera(); });
  if (btnSwitchCamVerif) btnSwitchCamVerif.addEventListener('click', (e) => { e.preventDefault(); switchCamera(); });
  
  // Manual verification
  if (btnScan) btnScan.addEventListener('click', performManualVerification);
  
  // NIK Fallback
  if (btnNikFallback) {
    btnNikFallback.addEventListener('click', () => {
      if (verifNikBox) verifNikBox.classList.toggle('hidden');
    });
  }
  
  if (btnCariNik) {
    btnCariNik.addEventListener('click', async () => {
      const nik = fallbackNik?.value?.trim();
      if (!nik || nik.length !== 16) {
        showAlert('NIK harus 16 digit');
        return;
      }
      
      showLoading('Mencari data...');
      try {
        const r = await fetch(`/api/patient/${nik}`);
        const d = await r.json();
        hideLoading();
        
        if (d.ok && d.patient) {
          STATE.activePatient = d.patient;
          updateStatus('✅ Data ditemukan!', 'success');
          
          if (verifData) {
            verifData.innerHTML = `
              <div class="space-y-2">
                <p><strong>NIK:</strong> ${d.patient.nik}</p>
                <p><strong>Nama:</strong> ${d.patient.name}</p>
                <p><strong>TTL:</strong> ${d.patient.dob || '-'}</p>
                <p><strong>Alamat:</strong> ${d.patient.address}</p>
              </div>
            `;
          }
          if (verifResult) verifResult.classList.remove('hidden');
          if (verifNikBox) verifNikBox.classList.add('hidden');
        } else {
          showAlert('NIK tidak ditemukan');
        }
      } catch (e) {
        hideLoading();
        showAlert('Error: ' + e.message);
      }
    });
  }
  
  // Modals
  if (btnAlertOk) btnAlertOk.addEventListener('click', () => modalAlert?.classList.add('hidden'));
  if (btnAntrianTutup) btnAntrianTutup.addEventListener('click', () => { modalAntrian?.classList.add('hidden'); showPage('page-home'); });
  if (btnDetailData) btnDetailData.addEventListener('click', stopNextScanCountdown);
  
  if (btnLanjutForm) {
    btnLanjutForm.addEventListener('click', () => {
      stopNextScanCountdown();
      if (STATE.activePatient) {
        if (gwNama) gwNama.textContent = STATE.activePatient.name;
        if (gwUmur) gwUmur.textContent = STATE.activePatient.age || computeAge(STATE.activePatient.dob);
        if (gwAlamat) gwAlamat.textContent = STATE.activePatient.address;
        showPage('page-poli-gateway');
      }
    });
  }
  
  if (btnModalRegisTutup) {
    btnModalRegisTutup.addEventListener('click', () => {
      modalRegisSuccess?.classList.add('hidden');
      showPage('page-home');
    });
  }
  
  if (btnModalLanjutPoli) {
    btnModalLanjutPoli.addEventListener('click', () => {
      modalRegisSuccess?.classList.add('hidden');
      if (STATE.activePatient) {
        if (gwNama) gwNama.textContent = STATE.activePatient.name;
        if (gwUmur) gwUmur.textContent = computeAge(STATE.activePatient.dob);
        if (gwAlamat) gwAlamat.textContent = STATE.activePatient.address;
        showPage('page-poli-gateway');
      }
    });
  }
  
  // Camera permission
  if (btnCameraPermission) {
    btnCameraPermission.addEventListener('click', async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: true });
        s.getTracks().forEach(t => t.stop());
        localStorage.setItem('cameraPermissionStatus', 'granted');
        hideCameraPermissionModal();
        showAlert('Kamera diizinkan! Silakan lanjutkan.');
        updateCameraStatus('granted');
      } catch (e) {
        localStorage.setItem('cameraPermissionStatus', 'denied');
        hideCameraPermissionModal();
        showAlert('Akses ditolak. Fitur scan wajah tidak bisa digunakan.');
        updateCameraStatus('denied');
      }
    });
  }
  
  if (btnCameraPermissionLater) {
    btnCameraPermissionLater.addEventListener('click', hideCameraPermissionModal);
  }
  
  // Registration form
  if (formRegistrasi) {
    formRegistrasi.addEventListener('submit', async (e) => {
      e.preventDefault();
      
      const status = localStorage.getItem('cameraPermissionStatus');
      if (status !== 'granted') {
        showCameraPermissionModal();
        return;
      }
      
      await ensureCamera('reg');
      if (!STATE.streamReg) {
        if (statusReg) statusReg.textContent = 'Gagal kamera';
        return;
      }
      
      if (statusReg) statusReg.textContent = 'Mengambil foto...';
      if (countReg) countReg.textContent = '0';
      showLoading('Registrasi: mengambil foto...');
      
      const frames = await captureMultipleFrames(videoReg, 15, 100, countReg);
      updateProgress(15, 15, 'Mengirim');
      
      const fd = new FormData();
      fd.append('nik', inputNik?.value?.trim() || '');
      fd.append('name', inputNama?.value?.trim() || '');
      fd.append('dob', inputDob?.value || '');
      fd.append('address', inputAlamat?.value?.trim() || '');
      frames.forEach((b, i) => fd.append('frames[]', b, `frame_${i}.jpg`));
      
      try {
        const r = await fetch('/api/register', { method: 'POST', body: fd });
        const d = await r.json();
        hideLoading();
        
        if (!d.ok) {
          showAlert(d.msg || 'Gagal');
          if (statusReg) statusReg.textContent = 'Gagal';
          return;
        }
        
        STATE.activePatient = {
          nik: inputNik?.value?.trim(),
          name: inputNama?.value?.trim(),
          address: inputAlamat?.value?.trim(),
          dob: inputDob?.value
        };
        
        formRegistrasi.reset();
        if (countReg) countReg.textContent = '0';
        if (modalRegisSuccess) modalRegisSuccess.classList.remove('hidden');
        
      } catch (err) {
        hideLoading();
        showAlert('Error: ' + err.message);
      }
    });
  }
  
  // Poli form
  if (formPoliGateway) {
    formPoliGateway.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!STATE.activePatient) return;
      
      showLoading('Mengambil antrian...');
      try {
        const r = await fetch('/api/queue/assign', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ poli: gwPoli?.value })
        });
        const d = await r.json();
        hideLoading();
        
        if (d.ok) {
          if (antrianPoli) antrianPoli.textContent = d.poli;
          if (antrianNomor) antrianNomor.textContent = d.nomor;
          if (modalAntrian) modalAntrian.classList.remove('hidden');
          formPoliGateway.reset();
          STATE.activePatient = null;
        } else {
          showAlert('Gagal ambil nomor');
        }
      } catch (err) {
        hideLoading();
        showAlert('Error: ' + err.message);
      }
    });
  }

  // ============================================================
  // INITIALIZATION
  // ============================================================
  
  if (localStorage.getItem('cameraPermissionStatus') === 'granted') {
    updateCameraStatus('granted');
  } else {
    updateCameraStatus('denied');
    setTimeout(showCameraPermissionModal, 1500);
  }
  
  showPage('page-home');
  
  console.log('✅ Blink-Gated Face Recognition System v2.1 initialized');
  console.log('📋 Config: EAR Threshold = 0.10 (kedip jika EAR < 0.10)');

})();
