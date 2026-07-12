
document.addEventListener("DOMContentLoaded", () => {
    // --- Supported upload types ---
    const SUPPORTED_AUDIO_EXTENSIONS = [
        ".aac",
        ".aif",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
    ];
    const SUPPORTED_MODEL_EXTENSIONS = [".torch", ".pt", ".pth"];

    // --- DOM Elements ---
    const modelSelect = document.getElementById("model-select");
    const audioFileInput = document.getElementById("audio-file");
    const modelFileInput = document.getElementById("model-file-input");
    const fileNameSpan = document.getElementById("file-name");
    const recordBtn = document.getElementById("record-btn");
    const stopBtn = document.getElementById("stop-btn");
    const recordTimerSpan = document.getElementById("record-timer");
    const transcribeBtn = document.getElementById("transcribe-btn");
    const statusDiv = document.getElementById("status");
    const statusMessage = document.getElementById("status-message");
    const loadingSpinner = document.getElementById("loading-spinner");
    const resultsDiv = document.getElementById("results");
    const playBtn = document.getElementById("play-btn");
    const canvas = document.getElementById("piano-roll-canvas");
    const ctx = canvas.getContext("2d");
    const pianoRollContainer = canvas.parentElement;
    const dropZone = document.getElementById("drop-zone");
    const uploadFileLabel = document.querySelector("label[for='audio-file']");

    const progressContainer = document.createElement("div");
    const progressBar = document.createElement("progress");
    const progressText = document.createElement("span");
    progressContainer.id = "upload-progress";
    progressContainer.style.display = "none";
    progressContainer.style.marginTop = "0.75rem";
    progressContainer.style.gap = "0.5rem";
    progressContainer.style.alignItems = "center";
    progressBar.max = 100;
    progressBar.value = 0;
    progressBar.style.width = "100%";
    progressText.textContent = "";
    progressContainer.append(progressBar, progressText);
    statusDiv.appendChild(progressContainer);

    const cancelBtn = document.createElement("button");
    cancelBtn.id = "cancel-transcribe-btn";
    cancelBtn.type = "button";
    cancelBtn.className = "button";
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.display = "none";
    statusDiv.appendChild(cancelBtn);

    // --- State ---
    let audioBlob = null;
    let modelFile = null;
    let mediaRecorder;
    let recordingChunks = [];
    let timerInterval;
    let transcriptionData = null;
    let isTranscribing = false;
    let currentRequest = null;
    let playbackSynth = null;
    let playbackAnimationRequest = null;
    let playbackState = "stopped";
    let playbackMode = "none";
    let playbackPosition = 0;
    let playbackDuration = 0;
    let synthPlaybackClockStartTime = 0;
    let audioElement = null;
    let audioObjectUrl = null;
    let pianoRollSpacer = null;
    let pianoRollFrameRequest = null;
    let pianoRollState = null;
    let pianoRollResizeTimeout = null;

    const PIANO_ROLL_HEIGHT = 400;
    const PIANO_ROLL_MIN_WIDTH = 1000;
    const PIANO_ROLL_PX_PER_SECOND = 50;
    const PIANO_ROLL_MAX_BACKING_WIDTH = 2400;
    const PIANO_ROLL_RESIZE_DEBOUNCE_MS = 100;
    const PIANO_ROLL_PEDAL_LANE_HEIGHT = 42;
    const PLAYHEAD_COLOR = "#f97316";

    // --- Initialization ---
    const init = async () => {
        setupEventListeners();
        await fetchModels();
    };

    const fetchModels = async () => {
        try {
            const response = await fetch("/api/models");
            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }
            const models = await response.json();
            modelSelect.innerHTML = 
                '<option value="">-- Select a model --</option>';
            if (models.length > 0) {
                models.forEach((model) => {
                    const option = document.createElement("option");
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
                modelSelect.selectedIndex = 1; // Select the first model by default

                const browseOption = document.createElement("option");
                browseOption.value = "browse";
                browseOption.textContent = "Browse for model...";
                modelSelect.appendChild(browseOption);
            } else {
                modelSelect.innerHTML = 
                    '<option value="">No models found</option>';
                updateStatus(
                    "No models found in `assets/`. Please add a model to the assets folder.",
                    "error"
                );
            }
        } catch (error) {
            console.error("Error fetching models:", error);
            modelSelect.innerHTML = 
                '<option value="">Error loading models</option>';
            updateStatus(
                "Could not load models. Is the server running correctly?",
                "error"
            );
        }
    };

    // --- Event Listeners ---
    function setupEventListeners() {
        modelSelect.addEventListener("change", () => {
            if (isTranscribing) return;
            if (modelSelect.value === "browse") {
                modelFileInput.click();
            }
            checkTranscribeButtonState();
        });

        modelFileInput.addEventListener("change", (event) => {
            const selectedModelFile = event.target.files[0];
            if (!selectedModelFile) {
                checkTranscribeButtonState();
                return;
            }

            if (!isAllowedExtension(selectedModelFile.name, SUPPORTED_MODEL_EXTENSIONS)) {
                modelFile = null;
                modelFileInput.value = "";
                updateStatus(
                    `Unsupported model type. Use ${SUPPORTED_MODEL_EXTENSIONS.join(", ")} checkpoints.`,
                    "error"
                );
                checkTranscribeButtonState();
                return;
            }

            modelFile = selectedModelFile;
            if (modelFile) {
                const option = document.createElement("option");
                option.value = modelFile.name;
                option.textContent = modelFile.name;
                option.selected = true;
                modelSelect.insertBefore(
                    option,
                    modelSelect.lastChild.previousSibling
                );
                modelSelect.value = modelFile.name;
            }
            checkTranscribeButtonState();
        });

        audioFileInput.addEventListener("change", handleFileSelect);
        recordBtn.addEventListener("click", startRecording);
        stopBtn.addEventListener("click", stopRecording);
        transcribeBtn.addEventListener("click", handleTranscribe);
        cancelBtn.addEventListener("click", cancelTranscription);
        playBtn.addEventListener("click", togglePlayback);
        pianoRollContainer.addEventListener("click", handlePianoRollSeek);
        pianoRollContainer.addEventListener("scroll", requestPianoRollRender);
        window.addEventListener("resize", handlePianoRollResize);

        // Drag and drop events
        dropZone.addEventListener("dragenter", (e) => {
            e.preventDefault();
            if (isTranscribing) return;
            dropZone.classList.add("drag-over");
        });

        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            if (isTranscribing) return;
            dropZone.classList.add("drag-over");
        });

        dropZone.addEventListener("dragleave", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
            if (isTranscribing) return;
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect({ target: { files: files } });
            }
        });
    }

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) {
            checkTranscribeButtonState();
            return;
        }

        const validationError = validateAudioFile(file);
        if (validationError) {
            clearSelectedAudio();
            updateStatus(validationError, "error");
            return;
        }

        audioBlob = file;
        prepareAudioPlaybackSource(file);
        transcriptionData = null;
        resultsDiv.style.display = "none";
        fileNameSpan.textContent = `${file.name} (${formatBytes(file.size)})`;
        updateStatus("Audio file ready. Select a model and transcribe.", "info");
        checkTranscribeButtonState();
    }

    // --- UI Updates ---
    function checkTranscribeButtonState() {
        transcribeBtn.disabled = isTranscribing || !(modelSelect.value && audioBlob);
    }

    function updateStatus(message, type = "info") {
        statusMessage.textContent = message;
        statusDiv.className = `status-area ${type}`; // for potential styling
        if (type === "loading") {
            loadingSpinner.style.display = "block";
        } else {
            loadingSpinner.style.display = "none";
        }
        checkTranscribeButtonState();
    }

    function setBusyState(isBusy) {
        isTranscribing = isBusy;
        modelSelect.disabled = isBusy;
        modelFileInput.disabled = isBusy;
        audioFileInput.disabled = isBusy;
        recordBtn.disabled = isBusy;
        stopBtn.disabled = true;
        playBtn.disabled = isBusy;
        cancelBtn.style.display = isBusy ? "inline-block" : "none";
        dropZone.style.opacity = isBusy ? "0.65" : "1";
        dropZone.style.cursor = isBusy ? "not-allowed" : "pointer";
        if (uploadFileLabel) {
            uploadFileLabel.style.pointerEvents = isBusy ? "none" : "auto";
            uploadFileLabel.setAttribute("aria-disabled", String(isBusy));
        }
        checkTranscribeButtonState();
    }

    function setProgress(percent, message = "") {
        const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
        progressContainer.style.display = "flex";
        progressBar.value = safePercent;
        progressText.textContent = message || `${safePercent}%`;
    }

    function hideProgress() {
        progressContainer.style.display = "none";
        progressBar.value = 0;
        progressText.textContent = "";
    }

    function clearSelectedAudio() {
        audioBlob = null;
        transcriptionData = null;
        clearAudioPlaybackSource();
        resultsDiv.style.display = "none";
        audioFileInput.value = "";
        fileNameSpan.textContent = " or drag and drop an audio file here";
        checkTranscribeButtonState();
    }

    function prepareAudioPlaybackSource(blob) {
        stopPlayback(true);
        clearAudioPlaybackSource(false);

        if (!blob) return;

        audioObjectUrl = URL.createObjectURL(blob);
        audioElement = new Audio(audioObjectUrl);
        audioElement.preload = "metadata";
        audioElement.addEventListener("ended", handleSourceAudioEnded);
        audioElement.addEventListener("loadedmetadata", handleSourceAudioMetadataLoaded);
        audioElement.addEventListener("error", () => {
            console.warn("Uploaded audio could not be loaded for synchronized playback.");
        });
        updatePlayButton();
    }

    function clearAudioPlaybackSource(stopExistingPlayback = true) {
        if (stopExistingPlayback) {
            stopPlayback(true);
        }

        if (audioElement) {
            audioElement.pause();
            audioElement.removeAttribute("src");
            audioElement.load();
            audioElement = null;
        }

        if (audioObjectUrl) {
            URL.revokeObjectURL(audioObjectUrl);
            audioObjectUrl = null;
        }
        updatePlayButton();
    }

    function hasPlayableSourceAudio() {
        return Boolean(audioElement && audioObjectUrl);
    }

    function handleSourceAudioEnded() {
        if (playbackMode !== "audio") return;
        stopPlayback(true);
    }

    function handleSourceAudioMetadataLoaded() {
        if (!transcriptionData || !pianoRollState) return;
        playbackDuration = getVisualizationDuration(
            transcriptionData.notes,
            transcriptionData.pedals,
            transcriptionData.duration
        );
        drawPianoRoll(
            transcriptionData.notes,
            transcriptionData.pedals,
            playbackDuration,
            false
        );
        updatePlayButton();
    }

    function validateAudioFile(file) {
        if (!isAudioFile(file)) {
            return `Unsupported file type. Please choose an audio file (${SUPPORTED_AUDIO_EXTENSIONS.join(", ")}).`;
        }

        return "";
    }

    function isAudioFile(file) {
        const hasAudioMimeType = file.type && file.type.toLowerCase().startsWith("audio/");
        return hasAudioMimeType || isAllowedExtension(file.name, SUPPORTED_AUDIO_EXTENSIONS);
    }

    function isAllowedExtension(filename, allowedExtensions) {
        const lowerName = filename.toLowerCase();
        return allowedExtensions.some((extension) => lowerName.endsWith(extension));
    }

    function formatBytes(bytes) {
        if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
        const units = ["B", "KB", "MB", "GB"];
        const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
        const value = bytes / Math.pow(1024, index);
        return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
    }

    // --- Recording Logic ---
    async function startRecording() {
        if (isTranscribing) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                recordingChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                audioBlob = new Blob(recordingChunks, { type: "audio/webm" });
                recordingChunks = [];
                prepareAudioPlaybackSource(audioBlob);
                transcriptionData = null;
                resultsDiv.style.display = "none";
                fileNameSpan.textContent = `recording_${new Date().toISOString()}.webm`;
                stream.getTracks().forEach((track) => track.stop()); // Stop mic access
                checkTranscribeButtonState();
            };

            mediaRecorder.start();
            recordBtn.disabled = true;
            stopBtn.disabled = false;
            recordBtn.classList.add("recording");
            recordBtn.textContent = "Recording...";
            startTimer();
        } catch (error) {
            console.error("Error starting recording:", error);
            updateStatus("Could not access microphone.", "error");
        }
    }

    function stopRecording() {
        if (!mediaRecorder || mediaRecorder.state === "inactive") return;
        mediaRecorder.stop();
        recordBtn.disabled = false;
        stopBtn.disabled = true;
        recordBtn.classList.remove("recording");
        recordBtn.textContent = "Record";
        stopTimer();
    }

    function startTimer() {
        let seconds = 0;
        recordTimerSpan.textContent = "00:00";
        timerInterval = setInterval(() => {
            seconds++;
            const mins = String(Math.floor(seconds / 60)).padStart(2, "0");
            const secs = String(seconds % 60).padStart(2, "0");
            recordTimerSpan.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    function stopTimer() {
        clearInterval(timerInterval);
    }

    // --- Transcription Logic ---
    async function handleTranscribe() {
        if (isTranscribing) return;

        if (!modelSelect.value || !audioBlob) {
            updateStatus("Please select a model and provide audio.", "error");
            return;
        }

        if (audioBlob instanceof File) {
            const validationError = validateAudioFile(audioBlob);
            if (validationError) {
                clearSelectedAudio();
                updateStatus(validationError, "error");
                return;
            }
        }

        setBusyState(true);
        hideProgress();
        updateStatus("Preparing upload...", "loading");
        stopPlayback(true);
        resultsDiv.style.display = "none";

        const formData = new FormData();
        formData.append("audio", audioBlob);
        if (modelFile) {
            formData.append("model_file", modelFile);
        } else {
            formData.append("model", modelSelect.value);
        }

        try {
            const data = await postTranscription(formData);

            transcriptionData = data;
            hideProgress();
            updateStatus("Transcription complete!", "success");
            displayResults(data);
        } catch (error) {
            console.error("Transcription error:", error);
            hideProgress();
            if (error.name === "AbortError") {
                updateStatus("Transcription cancelled. You can adjust inputs and try again.", "info");
            } else {
                updateStatus(`Error: ${error.message}`, "error");
            }
        } finally {
            currentRequest = null;
            setBusyState(false);
        }
    }

    function postTranscription(formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            currentRequest = xhr;

            xhr.open("POST", "/api/transcribe");
            xhr.responseType = "json";

            xhr.upload.addEventListener("progress", (event) => {
                if (event.lengthComputable) {
                    const percent = (event.loaded / event.total) * 100;
                    setProgress(percent, `Uploading ${Math.round(percent)}%`);
                    if (percent >= 100) {
                        updateStatus("Upload complete. Transcribing audio...", "loading");
                    }
                } else {
                    setProgress(0, "Uploading...");
                }
            });

            xhr.addEventListener("load", () => {
                const data = xhr.response;
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(data);
                    return;
                }
                reject(new Error((data && data.error) || `Transcription failed with status ${xhr.status}.`));
            });

            xhr.addEventListener("error", () => {
                reject(new Error("Network error while uploading audio. Check the server and try again."));
            });

            xhr.addEventListener("abort", () => {
                const abortError = new Error("Request aborted");
                abortError.name = "AbortError";
                reject(abortError);
            });

            setProgress(0, "Starting upload...");
            xhr.send(formData);
        });
    }

    function cancelTranscription() {
        if (currentRequest) {
            currentRequest.abort();
        }
    }

    // --- Results Display and Playback ---
    function displayResults(data) {
        try {
            resultsDiv.style.display = "block";
            resetPlaybackForResults(data);
            drawPianoRoll(data.notes, data.pedals, data.duration);
        } catch (error) {
            console.error("Error displaying results:", error);
            updateStatus(`Error displaying results: ${error.message}`, "error");
        }
    }

    async function togglePlayback() {
        if (!transcriptionData || !Array.isArray(transcriptionData.notes)) return;

        if (playbackState === "playing") {
            pausePlayback();
            return;
        }

        playBtn.disabled = true;
        playBtn.textContent = "Starting audio...";

        try {
            if (!hasPlayableSourceAudio()) {
                if (typeof Tone === "undefined") {
                    throw new Error("Tone.js is not available for synthesized transcription playback.");
                }
                // Browser autoplay policies require AudioContext startup from a user gesture.
                await Tone.start();
            }

            const notes = getPlayableNotes();
            const duration = getVisualizationDuration(
                transcriptionData.notes,
                transcriptionData.pedals,
                transcriptionData.duration
            );
            const startAt = playbackPosition >= duration ? 0 : playbackPosition;
            await startPlaybackFrom(startAt, notes, duration);
        } catch (error) {
            console.error("Playback start failed:", error);
            updateStatus(`Audio playback could not start: ${error.message}`, "error");
            updatePlayButton();
        } finally {
            playBtn.disabled = false;
        }
    }

    async function startPlaybackFrom(position, notes = getPlayableNotes(), duration = playbackDuration) {
        const boundedDuration = Math.max(Number(duration) || 0, 0.01);
        playbackDuration = boundedDuration;
        playbackPosition = clamp(Number(position) || 0, 0, boundedDuration);

        if (hasPlayableSourceAudio()) {
            await startSourceAudioPlayback(playbackPosition, boundedDuration);
            return;
        }

        startSynthPlaybackFrom(playbackPosition, notes, boundedDuration);
    }

    async function startSourceAudioPlayback(position, duration) {
        stopPlayback(false, false);

        playbackDuration = Math.max(Number(duration) || 0, 0.01);
        playbackPosition = clamp(Number(position) || 0, 0, playbackDuration);
        playbackMode = "audio";

        await ensureSourceAudioMetadata();
        seekSourceAudio(playbackPosition);

        playbackState = "playing";
        updatePlayButton();
        keepPlaybackPositionInView(playbackPosition, true);
        renderPianoRollViewport();
        startPlaybackAnimation();

        try {
            await audioElement.play();
        } catch (error) {
            playbackState = "paused";
            playbackMode = "none";
            cancelPlaybackAnimation();
            updatePlayButton();
            renderPianoRollViewport();
            throw error;
        }
    }

    function startSynthPlaybackFrom(position, notes = getPlayableNotes(), duration = playbackDuration) {
        const boundedDuration = Math.max(Number(duration) || 0, 0.01);
        stopPlayback(false, false);

        playbackDuration = boundedDuration;
        playbackPosition = clamp(Number(position) || 0, 0, boundedDuration);
        playbackMode = "synth";

        if (typeof Tone === "undefined") {
            throw new Error("Tone.js is not available for synthesized transcription playback.");
        }

        playbackSynth = new Tone.PolySynth(Tone.Synth, {
            oscillator: { type: "sine" },
            envelope: { attack: 0.01, decay: 0.1, sustain: 0.3, release: 1 },
        }).toDestination();

        const scheduleNow = Tone.now();
        synthPlaybackClockStartTime = scheduleNow - playbackPosition;

        notes.forEach((note) => {
            const noteStart = Math.max(0, Number(note.start) || 0);
            const noteDuration = Math.max(0.01, Number(note.duration) || 0.01);
            const noteEnd = noteStart + noteDuration;
            if (noteEnd <= playbackPosition || noteStart > boundedDuration) return;

            const offsetWithinNote = Math.max(0, playbackPosition - noteStart);
            const remainingDuration = Math.max(
                0.01,
                Math.min(noteDuration - offsetWithinNote, boundedDuration - Math.max(noteStart, playbackPosition))
            );
            const scheduledTime = scheduleNow + Math.max(0, noteStart - playbackPosition);

            playbackSynth.triggerAttackRelease(
                Tone.Frequency(note.pitch, "midi"),
                remainingDuration,
                scheduledTime,
                clamp(Number(note.velocity) || 0.8, 0, 1)
            );
        });

        playbackState = "playing";
        updatePlayButton();
        keepPlaybackPositionInView(playbackPosition, true);
        renderPianoRollViewport();
        startPlaybackAnimation();
    }

    function pausePlayback() {
        if (playbackState !== "playing") return;

        playbackPosition = getCurrentPlaybackPosition();
        if (playbackMode === "audio" && audioElement) {
            audioElement.pause();
        }
        disposePlaybackSynth();

        playbackState = "paused";
        playbackMode = "none";
        cancelPlaybackAnimation();
        updatePlayButton();
        renderPianoRollViewport();
    }

    function stopPlayback(resetButton = true, resetPosition = true) {
        cancelPlaybackAnimation();

        if (audioElement) {
            audioElement.pause();
            if (resetPosition) {
                seekSourceAudio(0);
            }
        }

        disposePlaybackSynth();
        if (typeof Tone !== "undefined" && Tone.Transport) {
            Tone.Transport.stop();
            Tone.Transport.cancel(0);
        }

        playbackState = "stopped";
        playbackMode = "none";
        synthPlaybackClockStartTime = 0;
        if (resetPosition) {
            playbackPosition = 0;
        }

        if (resetButton) {
            updatePlayButton();
        }
        renderPianoRollViewport();
    }

    function disposePlaybackSynth() {
        if (!playbackSynth) return;

        try {
            const releaseTime = typeof Tone !== "undefined" ? Tone.now() : undefined;
            playbackSynth.releaseAll(releaseTime);
            playbackSynth.dispose();
        } finally {
            playbackSynth = null;
        }
    }

    function ensureSourceAudioMetadata() {
        if (!audioElement) return Promise.reject(new Error("No source audio is available for playback."));
        if (Number.isFinite(audioElement.duration) || audioElement.readyState >= 1) {
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            const cleanup = () => {
                audioElement.removeEventListener("loadedmetadata", handleLoaded);
                audioElement.removeEventListener("error", handleError);
            };
            const handleLoaded = () => {
                cleanup();
                resolve();
            };
            const handleError = () => {
                cleanup();
                reject(new Error("Could not load the uploaded audio for playback."));
            };

            audioElement.addEventListener("loadedmetadata", handleLoaded);
            audioElement.addEventListener("error", handleError);
            audioElement.load();
        });
    }

    function getSourceAudioDuration() {
        return audioElement && Number.isFinite(audioElement.duration)
            ? Math.max(0, audioElement.duration)
            : 0;
    }

    function getSourceAudioSeekLimit() {
        return getSourceAudioDuration() || Math.max(playbackDuration || 0, 0.01);
    }

    function seekSourceAudio(position) {
        if (!audioElement) return;

        try {
            audioElement.currentTime = clamp(Number(position) || 0, 0, getSourceAudioSeekLimit());
        } catch (error) {
            console.debug("Could not seek source audio yet:", error);
        }
    }

    function resetPlaybackForResults(data) {
        stopPlayback(false);
        playbackDuration = getVisualizationDuration(data.notes, data.pedals, data.duration);
        playbackPosition = 0;
        playbackState = "stopped";
        updatePlayButton();
    }

    function startPlaybackAnimation() {
        cancelPlaybackAnimation();

        const animate = () => {
            if (playbackState !== "playing") return;

            playbackPosition = getCurrentPlaybackPosition();
            if (playbackPosition >= playbackDuration) {
                stopPlayback(true);
                return;
            }

            keepPlaybackPositionInView(playbackPosition);
            renderPianoRollViewport();
            playbackAnimationRequest = window.requestAnimationFrame(animate);
        };

        playbackAnimationRequest = window.requestAnimationFrame(animate);
    }

    function cancelPlaybackAnimation() {
        if (playbackAnimationRequest) {
            window.cancelAnimationFrame(playbackAnimationRequest);
            playbackAnimationRequest = null;
        }
    }

    function getCurrentPlaybackPosition() {
        if (playbackState !== "playing") {
            return clamp(playbackPosition, 0, playbackDuration || 0);
        }

        if (playbackMode === "audio" && audioElement) {
            return clamp(audioElement.currentTime || 0, 0, playbackDuration || 0);
        }

        if (playbackMode === "synth") {
            if (typeof Tone === "undefined") {
                return clamp(playbackPosition, 0, playbackDuration || 0);
            }
            return clamp(Tone.now() - synthPlaybackClockStartTime, 0, playbackDuration || 0);
        }

        return clamp(playbackPosition, 0, playbackDuration || 0);
    }

    function updatePlayButton() {
        if (playbackState === "playing") {
            playBtn.textContent = "Pause";
        } else if (playbackState === "paused") {
            playBtn.textContent = "Resume";
        } else {
            playBtn.textContent = hasPlayableSourceAudio()
                ? "Play Audio + Tracker"
                : "Play Transcription";
        }
    }

    function getPlayableNotes() {
        return transcriptionData && Array.isArray(transcriptionData.notes)
            ? transcriptionData.notes.filter(isFiniteNote)
            : [];
    }

    function handlePianoRollSeek(event) {
        if (!pianoRollState) return;

        const rect = pianoRollContainer.getBoundingClientRect();
        if (event.clientY > rect.top + pianoRollState.height) return;

        const xInViewport = clamp(event.clientX - rect.left, 0, pianoRollState.backingWidth);
        const virtualX = clamp(
            xInViewport + (pianoRollContainer.scrollLeft || 0),
            0,
            pianoRollState.virtualWidth
        );
        const targetTime = (virtualX / pianoRollState.virtualWidth) * pianoRollState.duration;
        seekPlayback(targetTime);
    }

    function seekPlayback(targetTime) {
        if (!pianoRollState) return;

        playbackDuration = pianoRollState.duration;
        playbackPosition = clamp(Number(targetTime) || 0, 0, playbackDuration);

        if (playbackState === "playing") {
            if (playbackMode === "audio" && audioElement) {
                seekSourceAudio(playbackPosition);
                keepPlaybackPositionInView(playbackPosition, true);
                renderPianoRollViewport();
            } else {
                startSynthPlaybackFrom(playbackPosition, getPlayableNotes(), playbackDuration);
            }
            return;
        }

        if (audioElement) {
            seekSourceAudio(playbackPosition);
        }
        keepPlaybackPositionInView(playbackPosition, true);
        updatePlayButton();
        renderPianoRollViewport();
    }

    function keepPlaybackPositionInView(currentTime, force = false) {
        if (!pianoRollState) return;

        const { duration, virtualWidth, backingWidth } = pianoRollState;
        if (duration <= 0 || virtualWidth <= backingWidth) return;

        const globalX = (currentTime / duration) * virtualWidth;
        const scrollLeft = pianoRollContainer.scrollLeft || 0;
        const rightEdge = scrollLeft + backingWidth;
        const margin = Math.min(180, backingWidth * 0.2);
        const maxScroll = Math.max(0, virtualWidth - backingWidth);

        if (force && (globalX < scrollLeft || globalX > rightEdge)) {
            pianoRollContainer.scrollLeft = clamp(globalX - backingWidth / 2, 0, maxScroll);
        } else if (globalX > rightEdge - margin) {
            pianoRollContainer.scrollLeft = clamp(globalX - backingWidth * 0.35, 0, maxScroll);
        } else if (globalX < scrollLeft + margin) {
            pianoRollContainer.scrollLeft = clamp(globalX - backingWidth * 0.15, 0, maxScroll);
        }
    }

    function isFiniteNote(note) {
        return Number.isFinite(Number(note.pitch)) &&
            Number.isFinite(Number(note.start)) &&
            Number.isFinite(Number(note.duration));
    }

    function getLastNoteEnd(notes) {
        return notes.reduce((lastEnd, note) => {
            const end = (Number(note.start) || 0) + Math.max(0, Number(note.duration) || 0);
            return Math.max(lastEnd, end);
        }, 0);
    }

    function getLastPedalEnd(pedals) {
        return pedals.reduce((lastEnd, pedal) => {
            const end = (Number(pedal.start) || 0) + Math.max(0, Number(pedal.duration) || 0);
            return Math.max(lastEnd, end);
        }, 0);
    }

    function getVisualizationDuration(notes, pedals, duration) {
        const safeNotes = Array.isArray(notes) ? notes.filter(isFiniteNote) : [];
        const safePedals = Array.isArray(pedals) ? pedals : [];
        return Math.max(
            Number(duration) || 0,
            getSourceAudioDuration(),
            getLastNoteEnd(safeNotes),
            getLastPedalEnd(safePedals),
            1
        );
    }

    // --- Piano Roll Visualization ---
    function drawPianoRoll(notes, pedals, duration, resetScroll = true) {
        const minPitch = 21; // A0
        const maxPitch = 108; // C8
        const pitchCount = maxPitch - minPitch + 1;
        const safeDuration = getVisualizationDuration(notes || [], pedals || [], duration);
        const virtualWidth = Math.max(PIANO_ROLL_MIN_WIDTH, safeDuration * PIANO_ROLL_PX_PER_SECOND);
        const viewportWidth = Math.max(
            PIANO_ROLL_MIN_WIDTH,
            Math.ceil(pianoRollContainer.clientWidth || PIANO_ROLL_MIN_WIDTH)
        );
        const backingWidth = Math.min(PIANO_ROLL_MAX_BACKING_WIDTH, viewportWidth);
        const height = PIANO_ROLL_HEIGHT;

        canvas.width = backingWidth;
        canvas.height = height;
        canvas.style.width = `${backingWidth}px`;
        canvas.style.height = `${height}px`;
        canvas.style.position = "sticky";
        canvas.style.left = "0";

        ensurePianoRollSpacer(virtualWidth, height);

        pianoRollState = {
            notes: Array.isArray(notes) ? notes.filter(isFiniteNote) : [],
            pedals: Array.isArray(pedals) ? pedals : [],
            duration: safeDuration,
            minPitch,
            maxPitch,
            pitchCount,
            virtualWidth,
            backingWidth,
            height,
        };

        pianoRollContainer.scrollLeft = resetScroll
            ? 0
            : Math.min(pianoRollContainer.scrollLeft || 0, Math.max(0, virtualWidth - backingWidth));
        renderPianoRollViewport();
    }

    function ensurePianoRollSpacer(width) {
        if (!pianoRollSpacer) {
            pianoRollSpacer = document.createElement("div");
            pianoRollSpacer.className = "piano-roll-spacer";
            pianoRollContainer.appendChild(pianoRollSpacer);
        }

        pianoRollSpacer.style.width = `${width}px`;
        pianoRollSpacer.style.height = "0";
    }

    function handlePianoRollResize() {
        if (!pianoRollState) return;

        clearTimeout(pianoRollResizeTimeout);
        pianoRollResizeTimeout = setTimeout(() => {
            drawPianoRoll(
                pianoRollState.notes,
                pianoRollState.pedals,
                pianoRollState.duration,
                false
            );
        }, PIANO_ROLL_RESIZE_DEBOUNCE_MS);
    }

    function requestPianoRollRender() {
        if (!pianoRollState || pianoRollFrameRequest) return;

        pianoRollFrameRequest = window.requestAnimationFrame(() => {
            pianoRollFrameRequest = null;
            renderPianoRollViewport();
        });
    }

    function renderPianoRollViewport() {
        if (!pianoRollState) return;

        const {
            notes,
            pedals,
            duration,
            minPitch,
            maxPitch,
            pitchCount,
            virtualWidth,
            backingWidth,
            height,
        } = pianoRollState;

        const scrollLeft = pianoRollContainer.scrollLeft || 0;
        const visibleWidth = backingWidth;
        const startTime = (scrollLeft / virtualWidth) * duration;
        const endTime = ((scrollLeft + visibleWidth) / virtualWidth) * duration;

        const pedalHeight = PIANO_ROLL_PEDAL_LANE_HEIGHT;
        const noteAreaHeight = height - pedalHeight;
        const keyHeight = noteAreaHeight / pitchCount;

        // Clear canvas
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, backingWidth, height);

        // Draw grid (octave lines and key labels)
        ctx.strokeStyle = "#eee";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#999";
        ctx.font = "12px sans-serif";

        for (let p = minPitch; p <= maxPitch; p++) {
            const y = noteAreaHeight - (p - minPitch) * keyHeight;
            if (p % 12 === 0) {
                // True C notes only. A0 is the lowest piano key, so no extra bottom C row is shown.
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(backingWidth, y);
                ctx.stroke();
                ctx.fillText(`C${Math.floor(p / 12) - 1}`, 5, clamp(y - 5, 12, noteAreaHeight - 2));
            }
        }

        drawTimeTicks(startTime, endTime, duration, virtualWidth, scrollLeft, backingWidth);

        drawPedalLane(pedals, startTime, endTime, duration, virtualWidth, scrollLeft, backingWidth, height, pedalHeight);

        // Draw note onsets. The current model predicts onset/velocity only, so
        // durations are treated as short visual extents unless explicitly marked
        // as real by the backend.
        notes.forEach((note) => {
            const noteStart = Number(note.start) || 0;
            const noteDuration = Math.max(0, Number(note.duration) || 0);
            const hasEstimatedDuration = note.duration_estimated === true;
            const noteEnd = noteStart + (hasEstimatedDuration ? Math.max(0.05, noteDuration) : noteDuration);
            if (hasEstimatedDuration) {
                if (noteStart < startTime || noteStart > endTime) return;
            } else if (noteEnd < startTime || noteStart > endTime) {
                return;
            }
            if (note.pitch < minPitch || note.pitch > maxPitch) return;

            const y = noteAreaHeight - (note.pitch - minPitch) * keyHeight;
            const x = (noteStart / duration) * virtualWidth - scrollLeft;
            const w = hasEstimatedDuration
                ? Math.max(2, Math.min(6, keyHeight * 1.5))
                : Math.max(1, (noteDuration / duration) * virtualWidth);

            ctx.fillStyle = `rgba(0, 0, 0, ${clamp(Number(note.velocity) || 0.8, 0, 1) * 0.8 + 0.2})`;
            ctx.fillRect(x, y - keyHeight, w, keyHeight);
        });

        drawPlaybackIndicator(duration, virtualWidth, scrollLeft, backingWidth, height);
    }

    function drawPedalLane(pedals, startTime, endTime, duration, virtualWidth, scrollLeft, backingWidth, height, pedalHeight) {
        const laneTop = height - pedalHeight;

        // Pedal lane background and label
        ctx.fillStyle = "#eef6ff";
        ctx.fillRect(0, laneTop, backingWidth, pedalHeight);
        ctx.fillStyle = "#1e3a8a";
        ctx.font = "bold 12px sans-serif";
        ctx.fillText("Sustain pedal", 8, laneTop + 15);

        // Sustain-held intervals
        ctx.fillStyle = "rgba(37, 99, 235, 0.25)";
        pedals.forEach((pedal) => {
            const pedalStart = Number(pedal.start) || 0;
            const pedalDuration = Math.max(0, Number(pedal.duration) || 0);
            const pedalEnd = pedalStart + pedalDuration;
            if (pedalEnd < startTime || pedalStart > endTime) return;

            const x = (pedalStart / duration) * virtualWidth - scrollLeft;
            const w = Math.max(1, (pedalDuration / duration) * virtualWidth);
            ctx.fillRect(x, laneTop + 4, w, pedalHeight - 8);
        });

        ctx.strokeStyle = "rgba(37, 99, 235, 0.65)";
        ctx.lineWidth = 1;
        ctx.strokeRect(0, laneTop, backingWidth, pedalHeight);

        // Compact onset markers: a small red arrow pointing up from the pedal lane.
        pedals.forEach((pedal) => {
            const pedalStart = Number(pedal.start) || 0;
            if (pedalStart < startTime || pedalStart > endTime) return;

            const x = (pedalStart / duration) * virtualWidth - scrollLeft;
            if (x < -4 || x > backingWidth + 4) return;

            const arrowTipY = height - 13;
            const arrowBaseY = height - 5;
            ctx.fillStyle = "rgba(220, 38, 38, 0.65)";
            ctx.beginPath();
            ctx.moveTo(x, arrowTipY);
            ctx.lineTo(x - 4, arrowBaseY);
            ctx.lineTo(x + 4, arrowBaseY);
            ctx.closePath();
            ctx.fill();
        });

        // Compact offset markers: a green arrow pointing down at pedal release.
        pedals.forEach((pedal) => {
            if (pedal.offset_estimated === true) return;

            const pedalStart = Number(pedal.start) || 0;
            const pedalDuration = Math.max(0, Number(pedal.duration) || 0);
            const pedalEnd = Number.isFinite(Number(pedal.end))
                ? Number(pedal.end)
                : pedalStart + pedalDuration;
            if (pedalEnd < startTime || pedalEnd > endTime) return;

            const x = (pedalEnd / duration) * virtualWidth - scrollLeft;
            if (x < -4 || x > backingWidth + 4) return;

            const arrowTipY = laneTop + 13;
            const arrowBaseY = laneTop + 5;
            ctx.fillStyle = "rgba(22, 163, 74, 0.75)";
            ctx.beginPath();
            ctx.moveTo(x, arrowTipY);
            ctx.lineTo(x - 4, arrowBaseY);
            ctx.lineTo(x + 4, arrowBaseY);
            ctx.closePath();
            ctx.fill();
        });
    }

    function drawPlaybackIndicator(duration, virtualWidth, scrollLeft, backingWidth, height) {
        if (!pianoRollState || playbackDuration <= 0) return;

        const currentTime = clamp(playbackPosition, 0, duration);
        const x = (currentTime / duration) * virtualWidth - scrollLeft;
        if (x < -8 || x > backingWidth + 8) return;

        ctx.save();
        ctx.strokeStyle = PLAYHEAD_COLOR;
        ctx.fillStyle = PLAYHEAD_COLOR;
        ctx.lineWidth = 2;
        ctx.shadowColor = "rgba(249, 115, 22, 0.35)";
        ctx.shadowBlur = playbackState === "playing" ? 8 : 0;

        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x, 11);
        ctx.lineTo(x - 7, 1);
        ctx.lineTo(x + 7, 1);
        ctx.closePath();
        ctx.fill();

        ctx.shadowBlur = 0;
        const label = formatPlaybackTime(currentTime);
        ctx.font = "bold 11px sans-serif";
        const labelWidth = ctx.measureText(label).width + 10;
        const labelX = clamp(x + 6, 2, Math.max(2, backingWidth - labelWidth - 2));
        ctx.fillStyle = "rgba(249, 115, 22, 0.95)";
        ctx.fillRect(labelX, 20, labelWidth, 18);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, labelX + 5, 33);
        ctx.restore();
    }

    function formatPlaybackTime(seconds) {
        const safeSeconds = Math.max(0, Number(seconds) || 0);
        const minutes = Math.floor(safeSeconds / 60);
        const wholeSeconds = Math.floor(safeSeconds % 60);
        const tenths = Math.floor((safeSeconds % 1) * 10);
        return `${minutes}:${String(wholeSeconds).padStart(2, "0")}.${tenths}`;
    }

    function drawTimeTicks(startTime, endTime, duration, virtualWidth, scrollLeft, backingWidth) {
        const pixelsPerSecond = virtualWidth / duration;
        const tickStep = pixelsPerSecond >= 80 ? 1 : pixelsPerSecond >= 25 ? 5 : 10;
        const firstTick = Math.floor(startTime / tickStep) * tickStep;

        ctx.strokeStyle = "#f3f4f6";
        ctx.fillStyle = "#777";
        ctx.font = "11px sans-serif";

        for (let t = firstTick; t <= endTime + tickStep; t += tickStep) {
            if (t < 0) continue;
            const x = (t / duration) * virtualWidth - scrollLeft;
            if (x < -1 || x > backingWidth + 1) continue;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, PIANO_ROLL_HEIGHT);
            ctx.stroke();
            ctx.fillText(`${Math.round(t)}s`, x + 4, 14);
        }
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    // --- Start the app ---
    init();
});
