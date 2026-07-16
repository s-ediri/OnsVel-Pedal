
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
    const audioSourceSelect = document.getElementById("audio-source-select");
    const syncStatus = document.getElementById("sync-status");
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
    let playbackAnimationRequest = null;
    let playbackState = "stopped";
    let playbackMode = "none";
    let playbackPosition = 0;
    let playbackDuration = 0;
    let audioElement = null;
    let audioObjectUrl = null;
    let generatedAudioElement = null;
    let generatedAudioUrl = null;
    let generatedAudioBalanceInfo = null;
    let activeAudioSource = "original";
    let audioSourceFadeRequest = null;
    let lastAudioSyncCorrectionTime = 0;
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
    const PIANO_ROLL_THEME = {
        background: "#07101f",
        octaveLine: "rgba(148, 163, 184, 0.36)",
        timeLine: "rgba(148, 163, 184, 0.22)",
        text: "#d7e3f3",
        mutedText: "#a8b3c7",
        textHalo: "rgba(7, 16, 31, 0.86)",
        pedalLaneBackground: "#0b1d32",
        pedalLaneText: "#dbeafe",
        pedalFill: "rgba(56, 189, 248, 0.24)",
        pedalBorder: "rgba(56, 189, 248, 0.62)",
        pedalOnset: "rgba(34, 197, 94, 0.95)",
        pedalOffset: "rgba(248, 113, 113, 0.9)",
        playhead: "#f59e0b",
        playheadShadow: "rgba(245, 158, 11, 0.38)",
        playheadLabel: "rgba(245, 158, 11, 0.95)",
        playheadText: "#0b1120",
    };
    const PLAYHEAD_COLOR = PIANO_ROLL_THEME.playhead;
    const ACTIVE_AUDIO_VOLUME = 1;
    const INACTIVE_AUDIO_VOLUME = 0;
    const AUDIO_SOURCE_SWITCH_FADE_MS = 80;
    const AUDIO_SYNC_TOLERANCE_SECS = 0.012;
    const AUDIO_SYNC_CHECK_INTERVAL_MS = 80;
    const GENERATED_AUDIO_ENGINE_NAME = "pre-rendered sampled grand piano WAV";

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
        if (audioSourceSelect) {
            audioSourceSelect.addEventListener("change", () => {
                switchPlaybackAudioSource(audioSourceSelect.value);
            });
        }
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
        clearGeneratedAudioPlaybackSource(false);
        resultsDiv.style.display = "none";
        audioFileInput.value = "";
        fileNameSpan.textContent = " or drag and drop an audio file here";
        checkTranscribeButtonState();
    }

    function prepareAudioPlaybackSource(blob) {
        stopPlayback(true);
        clearAudioPlaybackSource(false);
        clearGeneratedAudioPlaybackSource(false);
        activeAudioSource = "original";

        if (!blob) return;

        audioObjectUrl = URL.createObjectURL(blob);
        audioElement = new Audio(audioObjectUrl);
        audioElement.preload = "metadata";
        audioElement.volume = activeAudioSource === "original" ? ACTIVE_AUDIO_VOLUME : INACTIVE_AUDIO_VOLUME;
        audioElement.addEventListener("ended", () => handleMediaAudioEnded("original"));
        audioElement.addEventListener("loadedmetadata", handleMediaAudioMetadataLoaded);
        audioElement.addEventListener("error", () => {
            console.warn("Uploaded audio could not be loaded for synchronized playback.");
            updateAudioSourceControls();
        });
        updateAudioSourceControls();
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

    function prepareGeneratedAudioPlaybackSource(generatedAudio) {
        clearGeneratedAudioPlaybackSource(false);
        generatedAudioBalanceInfo = generatedAudio && generatedAudio.balance
            ? generatedAudio.balance
            : null;

        const url = generatedAudio && typeof generatedAudio.url === "string"
            ? generatedAudio.url
            : "";
        if (!url) {
            const message = generatedAudio && generatedAudio.error
                ? generatedAudio.error
                : "Generated piano audio was not rendered; playback will use the original audio if possible.";
            updateAudioSourceControls(message);
            return;
        }

        generatedAudioUrl = url;
        activeAudioSource = "generated";
        generatedAudioElement = new Audio(generatedAudioUrl);
        generatedAudioElement.preload = "auto";
        generatedAudioElement.volume = activeAudioSource === "generated" ? ACTIVE_AUDIO_VOLUME : INACTIVE_AUDIO_VOLUME;
        generatedAudioElement.addEventListener("ended", () => handleMediaAudioEnded("generated"));
        generatedAudioElement.addEventListener("loadedmetadata", handleMediaAudioMetadataLoaded);
        generatedAudioElement.addEventListener("canplaythrough", () => updateAudioSourceControls());
        generatedAudioElement.addEventListener("error", () => {
            console.warn("Generated piano audio could not be loaded for synchronized playback.");
            clearGeneratedAudioPlaybackSource(false);
            updateAudioSourceControls("Generated piano audio could not be loaded.");
        });
        generatedAudioElement.load();
        updateAudioSourceControls();
    }

    function clearGeneratedAudioPlaybackSource(stopExistingPlayback = true) {
        if (stopExistingPlayback) {
            stopPlayback(true);
        }

        if (generatedAudioElement) {
            generatedAudioElement.pause();
            generatedAudioElement.removeAttribute("src");
            generatedAudioElement.load();
            generatedAudioElement = null;
        }
        generatedAudioUrl = null;
        if (activeAudioSource === "generated") {
            activeAudioSource = hasPlayableSourceAudio() ? "original" : "generated";
        }
        updateAudioSourceControls();
        updatePlayButton();
    }

    function hasPlayableSourceAudio() {
        return Boolean(audioElement && audioObjectUrl);
    }

    function hasPlayableGeneratedAudioElement() {
        return Boolean(generatedAudioElement && generatedAudioUrl);
    }

    function hasPlayableGeneratedAudio() {
        return hasPlayableGeneratedAudioElement();
    }

    function hasPlayableMediaAudio() {
        return hasPlayableSourceAudio() || hasPlayableGeneratedAudioElement();
    }

    function handleMediaAudioEnded(sourceName) {
        if (playbackMode !== "audio" || sourceName !== activeAudioSource) return;
        stopPlayback(true);
    }

    function handleMediaAudioMetadataLoaded() {
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
        updateAudioSourceControls();
        updatePlayButton();
    }

    function updateAudioSourceControls(message = "", resetVolumes = true) {
        const hasOriginal = hasPlayableSourceAudio();
        const hasGenerated = hasPlayableGeneratedAudio();

        if (activeAudioSource === "original" && !hasOriginal && hasGenerated) {
            activeAudioSource = "generated";
        } else if (activeAudioSource === "generated" && !hasGenerated && hasOriginal) {
            activeAudioSource = "original";
        }

        if (audioSourceSelect) {
            const originalOption = audioSourceSelect.querySelector("option[value='original']");
            const generatedOption = audioSourceSelect.querySelector("option[value='generated']");
            if (originalOption) originalOption.disabled = !hasOriginal;
            if (generatedOption) generatedOption.disabled = !hasGenerated;
            audioSourceSelect.value = activeAudioSource;
            audioSourceSelect.disabled = isTranscribing || !(hasOriginal && hasGenerated);
        }

        if (resetVolumes) {
            setMediaVolumesForActiveSource(false);
        }

        if (syncStatus) {
            if (message) {
                syncStatus.textContent = message;
                syncStatus.className = "sync-status warning";
            } else if (hasOriginal && hasGenerated) {
                syncStatus.textContent = `Original audio and generated grand piano are ready. Both are media-synced to the piano roll using the ${GENERATED_AUDIO_ENGINE_NAME}.${getGeneratedAudioBalanceMessage()}`;
                syncStatus.className = "sync-status ready";
            } else if (hasGenerated) {
                syncStatus.textContent = `Generated grand piano playback is ready via the ${GENERATED_AUDIO_ENGINE_NAME}.${getGeneratedAudioBalanceMessage()}`;
                syncStatus.className = "sync-status warning";
            } else {
                syncStatus.textContent = "Generated grand piano playback is not available; playback will use the original audio if possible.";
                syncStatus.className = "sync-status warning";
            }
        }
    }

    function getGeneratedAudioBalanceMessage() {
        if (!generatedAudioBalanceInfo || typeof generatedAudioBalanceInfo !== "object") {
            return "";
        }

        if (generatedAudioBalanceInfo.applied) {
            const gain = Number(generatedAudioBalanceInfo.gain);
            const gainMessage = Number.isFinite(gain) && gain > 0
                ? ` (${formatGainDb(gain)})`
                : "";
            return ` Volume balanced to the original audio${gainMessage}.`;
        }

        const reason = generatedAudioBalanceInfo.reason || "";
        if (reason === "disabled" || reason === "missing_reference") {
            return "";
        }
        return " Volume balancing to the original audio was unavailable.";
    }

    function formatGainDb(gain) {
        const safeGain = Math.max(Number(gain) || 0, 0.000001);
        const db = 20 * Math.log10(safeGain);
        const sign = db >= 0 ? "+" : "";
        return `${sign}${db.toFixed(1)} dB`;
    }

    async function switchPlaybackAudioSource(requestedSource) {
        const nextSource = requestedSource === "generated" ? "generated" : "original";
        if (nextSource === "generated" && !hasPlayableGeneratedAudio()) {
            updateAudioSourceControls("Generated grand piano playback is still loading or unavailable.");
            return;
        }
        if (nextSource === "original" && !hasPlayableSourceAudio()) {
            updateAudioSourceControls("Original audio is unavailable for synced playback.");
            return;
        }

        const timelinePosition = getCurrentPlaybackPosition();
        activeAudioSource = nextSource;
        playbackPosition = clamp(timelinePosition, 0, playbackDuration || 0);
        seekMediaAudio(playbackPosition);
        updateAudioSourceControls("", false);
        setMediaVolumesForActiveSource(true);
        updatePlayButton();

        if (playbackState === "playing") {
            playBtn.disabled = true;
            playBtn.textContent = "Syncing audio...";
            try {
                await startPlaybackFrom(playbackPosition, playbackDuration);
            } catch (error) {
                console.warn("Could not switch playback source:", error);
                updateStatus(`Audio source switch failed: ${error.message}`, "error");
                updatePlayButton();
            } finally {
                playBtn.disabled = false;
            }
        }

        renderPianoRollViewport();
    }

    function getPlayableMediaAudioElements() {
        const elements = [];
        if (hasPlayableSourceAudio()) {
            elements.push({ name: "original", element: audioElement });
        }
        if (hasPlayableGeneratedAudioElement()) {
            elements.push({ name: "generated", element: generatedAudioElement });
        }
        return elements;
    }

    function getActiveAudioElement() {
        if (activeAudioSource === "generated" && hasPlayableGeneratedAudioElement()) {
            return generatedAudioElement;
        }
        if (hasPlayableSourceAudio()) {
            return audioElement;
        }
        if (hasPlayableGeneratedAudioElement()) {
            return generatedAudioElement;
        }
        return null;
    }

    function pauseMediaAudio() {
        getPlayableMediaAudioElements().forEach(({ element }) => {
            element.pause();
        });
    }

    function setMediaVolumesForActiveSource(animated = true) {
        if (audioSourceFadeRequest) {
            window.cancelAnimationFrame(audioSourceFadeRequest);
            audioSourceFadeRequest = null;
        }

        const mediaElements = getPlayableMediaAudioElements().map(({ name, element }) => ({
            element,
            startVolume: Number.isFinite(element.volume) ? element.volume : 0,
            targetVolume: name === activeAudioSource ? ACTIVE_AUDIO_VOLUME : INACTIVE_AUDIO_VOLUME,
        }));
        if (!animated || mediaElements.length === 0 || AUDIO_SOURCE_SWITCH_FADE_MS <= 0) {
            mediaElements.forEach(({ element, targetVolume }) => {
                element.volume = targetVolume;
            });
            return;
        }

        const fadeStart = performance.now();
        const step = (now) => {
            const progress = clamp((now - fadeStart) / AUDIO_SOURCE_SWITCH_FADE_MS, 0, 1);
            mediaElements.forEach(({ element, startVolume, targetVolume }) => {
                element.volume = startVolume + (targetVolume - startVolume) * progress;
            });
            if (progress < 1) {
                audioSourceFadeRequest = window.requestAnimationFrame(step);
            } else {
                audioSourceFadeRequest = null;
            }
        };
        audioSourceFadeRequest = window.requestAnimationFrame(step);
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
            prepareGeneratedAudioPlaybackSource(data.generated_audio);
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
        playBtn.textContent = "Starting synced audio...";

        try {
            if (!hasPlayableMediaAudio()) {
                throw new Error("No original or generated audio source is available for synchronized playback.");
            }

            const duration = getVisualizationDuration(
                transcriptionData.notes,
                transcriptionData.pedals,
                transcriptionData.duration
            );
            const startAt = playbackPosition >= duration ? 0 : playbackPosition;
            await startPlaybackFrom(startAt, duration);
        } catch (error) {
            console.error("Playback start failed:", error);
            updateStatus(`Audio playback could not start: ${error.message}`, "error");
            updatePlayButton();
        } finally {
            playBtn.disabled = false;
        }
    }

    async function startPlaybackFrom(position, duration = playbackDuration) {
        const boundedDuration = Math.max(Number(duration) || 0, 0.01);
        playbackDuration = boundedDuration;
        playbackPosition = clamp(Number(position) || 0, 0, boundedDuration);

        if (hasPlayableMediaAudio()) {
            await startSynchronizedAudioPlayback(playbackPosition, boundedDuration);
            return;
        }

        throw new Error("No original or generated audio source is available for synchronized playback.");
    }

    async function startSynchronizedAudioPlayback(position, duration) {
        stopPlayback(false, false);

        playbackDuration = Math.max(Number(duration) || 0, 0.01);
        playbackPosition = clamp(Number(position) || 0, 0, playbackDuration);
        playbackMode = "audio";

        await ensureAvailableAudioMetadata();
        seekMediaAudio(playbackPosition);
        setMediaVolumesForActiveSource(false);

        playbackState = "playing";
        updatePlayButton();
        updateAudioSourceControls();
        keepPlaybackPositionInView(playbackPosition, true);
        renderPianoRollViewport();

        try {
            const started = await playAvailableMediaAudio();
            if (!started) {
                throw new Error("No original or generated audio source is available for playback.");
            }
            startPlaybackAnimation();
        } catch (error) {
            playbackState = "paused";
            playbackMode = "none";
            pauseMediaAudio();
            cancelPlaybackAnimation();
            updatePlayButton();
            renderPianoRollViewport();
            throw error;
        }
    }

    function pausePlayback() {
        if (playbackState !== "playing") return;

        playbackPosition = getCurrentPlaybackPosition();
        pauseMediaAudio();

        playbackState = "paused";
        playbackMode = "none";
        cancelPlaybackAnimation();
        updatePlayButton();
        renderPianoRollViewport();
    }

    function stopPlayback(resetButton = true, resetPosition = true) {
        cancelPlaybackAnimation();

        pauseMediaAudio();
        if (resetPosition) {
            seekMediaAudio(0);
        }

        playbackState = "stopped";
        playbackMode = "none";
        if (resetPosition) {
            playbackPosition = 0;
        }

        if (resetButton) {
            updatePlayButton();
        }
        renderPianoRollViewport();
    }

    function ensureSourceAudioMetadata() {
        return ensureAudioElementMetadata(audioElement, "uploaded audio");
    }

    function ensureGeneratedAudioMetadata() {
        return ensureAudioElementMetadata(generatedAudioElement, "generated piano audio");
    }

    function ensureAvailableAudioMetadata() {
        const metadataPromises = [];
        if (hasPlayableSourceAudio()) metadataPromises.push(ensureSourceAudioMetadata());
        if (hasPlayableGeneratedAudioElement()) metadataPromises.push(ensureGeneratedAudioMetadata());
        if (metadataPromises.length === 0) {
            return Promise.reject(new Error("No audio is available for playback."));
        }
        return Promise.all(metadataPromises);
    }

    function ensureAudioElementMetadata(element, label) {
        if (!element) return Promise.reject(new Error(`No ${label} is available for playback.`));
        if (Number.isFinite(element.duration) || element.readyState >= 1) {
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            const cleanup = () => {
                element.removeEventListener("loadedmetadata", handleLoaded);
                element.removeEventListener("error", handleError);
            };
            const handleLoaded = () => {
                cleanup();
                resolve();
            };
            const handleError = () => {
                cleanup();
                reject(new Error(`Could not load the ${label} for playback.`));
            };

            element.addEventListener("loadedmetadata", handleLoaded);
            element.addEventListener("error", handleError);
            element.load();
        });
    }

    function getSourceAudioDuration() {
        return audioElement && Number.isFinite(audioElement.duration)
            ? Math.max(0, audioElement.duration)
            : 0;
    }

    function getGeneratedAudioDuration() {
        return generatedAudioElement && Number.isFinite(generatedAudioElement.duration)
            ? Math.max(0, generatedAudioElement.duration)
            : 0;
    }

    function getSourceAudioSeekLimit() {
        return getSourceAudioDuration() || Math.max(playbackDuration || 0, 0.01);
    }

    function getGeneratedAudioSeekLimit() {
        return getGeneratedAudioDuration() || Math.max(playbackDuration || 0, 0.01);
    }

    function seekSourceAudio(position) {
        seekAudioElement(audioElement, position, getSourceAudioSeekLimit());
    }

    function seekGeneratedAudio(position) {
        seekAudioElement(generatedAudioElement, position, getGeneratedAudioSeekLimit());
    }

    function seekMediaAudio(position) {
        if (hasPlayableSourceAudio()) seekSourceAudio(position);
        if (hasPlayableGeneratedAudioElement()) seekGeneratedAudio(position);
    }

    function seekAudioElement(element, position, seekLimit) {
        if (!element) return;

        try {
            element.currentTime = clamp(Number(position) || 0, 0, seekLimit);
        } catch (error) {
            console.debug("Could not seek audio yet:", error);
        }
    }

    async function playAvailableMediaAudio(resetVolumes = true) {
        const mediaElements = getPlayableMediaAudioElements();
        if (mediaElements.length === 0) return false;

        if (resetVolumes) {
            setMediaVolumesForActiveSource(false);
        }
        const results = await Promise.allSettled(
            mediaElements.map(({ name, element }) => {
                if (Math.abs((element.currentTime || 0) - playbackPosition) > AUDIO_SYNC_TOLERANCE_SECS) {
                    seekAudioElement(
                        element,
                        playbackPosition,
                        name === "generated" ? getGeneratedAudioSeekLimit() : getSourceAudioSeekLimit()
                    );
                }
                return element.play();
            })
        );

        const activeIndex = mediaElements.findIndex(({ name }) => name === activeAudioSource);
        const activeResult = activeIndex >= 0 ? results[activeIndex] : null;
        const activeStarted = activeResult && activeResult.status === "fulfilled";
        const fallbackIndex = results.findIndex((result) => result.status === "fulfilled");

        results.forEach((result, index) => {
            if (result.status === "rejected") {
                console.warn(`Could not start ${mediaElements[index].name} audio:`, result.reason);
            }
        });

        if (!activeStarted && fallbackIndex >= 0) {
            activeAudioSource = mediaElements[fallbackIndex].name;
            updateAudioSourceControls("", false);
            setMediaVolumesForActiveSource(false);
        }

        return Boolean(activeStarted || fallbackIndex >= 0);
    }

    function keepMediaAudioInSync(timelinePosition) {
        if (playbackMode !== "audio" || playbackState !== "playing") return;
        const now = performance.now();
        if (now - lastAudioSyncCorrectionTime < AUDIO_SYNC_CHECK_INTERVAL_MS) return;

        const activeElement = getActiveAudioElement();
        if (!activeElement) return;

        const targetTime = clamp(
            Number(timelinePosition) || activeElement.currentTime || 0,
            0,
            playbackDuration || 0
        );

        getPlayableMediaAudioElements().forEach(({ name, element }) => {
            const drift = Math.abs((element.currentTime || 0) - targetTime);
            if (drift > AUDIO_SYNC_TOLERANCE_SECS) {
                seekAudioElement(
                    element,
                    targetTime,
                    name === "generated" ? getGeneratedAudioSeekLimit() : getSourceAudioSeekLimit()
                );
            }
            if (element.paused && targetTime < playbackDuration) {
                element.play().catch((error) => {
                    console.debug(`Could not resume ${name} audio during sync check:`, error);
                });
            }
        });

        lastAudioSyncCorrectionTime = now;
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
            keepMediaAudioInSync(playbackPosition);
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

        if (playbackMode === "audio") {
            const activeElement = getActiveAudioElement();
            return clamp(activeElement ? activeElement.currentTime || 0 : playbackPosition, 0, playbackDuration || 0);
        }

        return clamp(playbackPosition, 0, playbackDuration || 0);
    }

    function updatePlayButton() {
        if (playbackState === "playing") {
            playBtn.textContent = "Pause";
        } else if (playbackState === "paused") {
            playBtn.textContent = "Resume";
        } else {
            playBtn.textContent = hasPlayableMediaAudio()
                ? "Play Synced Audio + Tracker"
                : "No Synced Audio Available";
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
            if (playbackMode === "audio" && hasPlayableMediaAudio()) {
                seekMediaAudio(playbackPosition);
                keepPlaybackPositionInView(playbackPosition, true);
                renderPianoRollViewport();
            } else {
                startPlaybackFrom(playbackPosition, playbackDuration).catch((error) => {
                    console.warn("Could not seek synchronized audio playback:", error);
                    updateStatus(`Playback seek failed: ${error.message}`, "error");
                    updatePlayButton();
                });
            }
            return;
        }

        if (hasPlayableMediaAudio()) {
            seekMediaAudio(playbackPosition);
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
            const end = getPedalEnd(pedal);
            return Math.max(lastEnd, end);
        }, 0);
    }

    function getPedalStart(pedal) {
        return Math.max(0, Number(pedal && pedal.start) || 0);
    }

    function getPedalEnd(pedal) {
        const start = getPedalStart(pedal);
        const explicitEnd = Number(pedal && pedal.end);
        if (Number.isFinite(explicitEnd)) {
            return Math.max(start, explicitEnd);
        }
        return start + Math.max(0, Number(pedal && pedal.duration) || 0);
    }

    function getPedalDuration(pedal) {
        return Math.max(0, getPedalEnd(pedal) - getPedalStart(pedal));
    }

    function isFinitePedalInterval(pedal) {
        return Number.isFinite(Number(pedal && pedal.start)) &&
            (Number.isFinite(Number(pedal && pedal.end)) || Number.isFinite(Number(pedal && pedal.duration)));
    }

    function getVisualizationDuration(notes, pedals, duration) {
        const safeNotes = Array.isArray(notes) ? notes.filter(isFiniteNote) : [];
        const safePedals = Array.isArray(pedals) ? pedals.filter(isFinitePedalInterval) : [];
        return Math.max(
            Number(duration) || 0,
            getSourceAudioDuration(),
            getGeneratedAudioDuration(),
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
            pedals: Array.isArray(pedals) ? pedals.filter(isFinitePedalInterval) : [],
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

        // Clear canvas with the dark piano-roll surface.
        ctx.fillStyle = PIANO_ROLL_THEME.background;
        ctx.fillRect(0, 0, backingWidth, height);

        // Draw grid (octave lines and key labels)
        ctx.strokeStyle = PIANO_ROLL_THEME.octaveLine;
        ctx.lineWidth = 1;
        ctx.font = "600 12px Inter, sans-serif";

        for (let p = minPitch; p <= maxPitch; p++) {
            const y = noteAreaHeight - (p - minPitch) * keyHeight;
            if (p % 12 === 0) {
                // True C notes only. A0 is the lowest piano key, so no extra bottom C row is shown.
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(backingWidth, y);
                ctx.stroke();
                drawLabelWithHalo(
                    `C${Math.floor(p / 12) - 1}`,
                    5,
                    clamp(y - 5, 12, noteAreaHeight - 2),
                    PIANO_ROLL_THEME.text
                );
            }
        }

        drawTimeTicks(startTime, endTime, duration, virtualWidth, scrollLeft, backingWidth);

        drawPedalLane(pedals, startTime, endTime, duration, virtualWidth, scrollLeft, backingWidth, height, pedalHeight);

        // Draw notes. Durations may still be estimates because the model emits
        // note onsets/velocities rather than true note-off frames, but the
        // backend now supplies musically useful estimated extents for display.
        notes.forEach((note) => {
            const noteStart = Number(note.start) || 0;
            const noteDuration = Math.max(0, Number(note.duration) || 0);
            const hasEstimatedDuration = note.duration_estimated === true;
            const noteEnd = noteStart + Math.max(0.05, noteDuration);
            if (noteEnd < startTime || noteStart > endTime) {
                return;
            }
            if (note.pitch < minPitch || note.pitch > maxPitch) return;

            const y = noteAreaHeight - (note.pitch - minPitch) * keyHeight;
            const x = (noteStart / duration) * virtualWidth - scrollLeft;
            const w = Math.max(2, (Math.max(0.05, noteDuration) / duration) * virtualWidth);
            const velocity = clamp(Number(note.velocity) || 0.8, 0, 1);
            const noteColor = velocityToColor(velocity);

            const alpha = hasEstimatedDuration
                ? Math.max(0.32, noteColor.alpha * 0.72)
                : noteColor.alpha;
            ctx.fillStyle = `rgba(${noteColor.r}, ${noteColor.g}, ${noteColor.b}, ${alpha})`;
            ctx.fillRect(x, y - keyHeight, w, keyHeight);
            if (hasEstimatedDuration) {
                ctx.strokeStyle = `rgba(${noteColor.r}, ${noteColor.g}, ${noteColor.b}, 0.88)`;
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y - keyHeight + 0.5, w, Math.max(1, keyHeight - 1));
                const radius = Math.max(2, Math.min(6, keyHeight * (0.38 + velocity * 0.55)));
                ctx.beginPath();
                ctx.fillStyle = `rgba(${noteColor.r}, ${noteColor.g}, ${noteColor.b}, 0.72)`;
                ctx.arc(x + Math.min(w / 2, radius + 1), y - keyHeight / 2, radius, 0, Math.PI * 2);
                ctx.fill();
            }
        });

        drawPlaybackIndicator(duration, virtualWidth, scrollLeft, backingWidth, height);
    }

    function velocityToColor(velocity) {
        const v = clamp(Number(velocity) || 0, 0, 1);
        const low = { r: 34, g: 197, b: 94 };    // green: soft
        const mid = { r: 250, g: 204, b: 21 };   // yellow: medium
        const high = { r: 248, g: 113, b: 113 }; // red: loud
        const start = v < 0.5 ? low : mid;
        const end = v < 0.5 ? mid : high;
        const amount = v < 0.5 ? v * 2 : (v - 0.5) * 2;

        return {
            r: Math.round(start.r + (end.r - start.r) * amount),
            g: Math.round(start.g + (end.g - start.g) * amount),
            b: Math.round(start.b + (end.b - start.b) * amount),
            alpha: 0.45 + v * 0.55,
        };
    }

    function drawPedalLane(pedals, startTime, endTime, duration, virtualWidth, scrollLeft, backingWidth, height, pedalHeight) {
        const laneTop = height - pedalHeight;

        // Pedal lane background and label
        ctx.fillStyle = PIANO_ROLL_THEME.pedalLaneBackground;
        ctx.fillRect(0, laneTop, backingWidth, pedalHeight);
        ctx.font = "700 12px Inter, sans-serif";
        drawLabelWithHalo("Sustain pedal", 8, laneTop + 15, PIANO_ROLL_THEME.pedalLaneText);

        // Sustain-held intervals
        ctx.fillStyle = PIANO_ROLL_THEME.pedalFill;
        pedals.forEach((pedal) => {
            const pedalStart = getPedalStart(pedal);
            const pedalEnd = getPedalEnd(pedal);
            if (pedalEnd < startTime || pedalStart > endTime) return;

            const visibleStart = Math.max(pedalStart, startTime);
            const visibleEnd = Math.min(pedalEnd, endTime);
            const visibleDuration = Math.max(0, visibleEnd - visibleStart);
            if (visibleDuration <= 0) return;

            const x = (visibleStart / duration) * virtualWidth - scrollLeft;
            const w = Math.max(1, (visibleDuration / duration) * virtualWidth);
            ctx.fillRect(x, laneTop + 4, w, pedalHeight - 8);
        });

        ctx.strokeStyle = PIANO_ROLL_THEME.pedalBorder;
        ctx.lineWidth = 1;
        ctx.strokeRect(0, laneTop, backingWidth, pedalHeight);

        // Compact onset markers: a green arrow pointing down from the top of the pedal lane.
        pedals.forEach((pedal) => {
            const pedalStart = getPedalStart(pedal);
            if (pedalStart < startTime || pedalStart > endTime) return;

            const x = (pedalStart / duration) * virtualWidth - scrollLeft;
            if (x < -4 || x > backingWidth + 4) return;

            const arrowTipY = laneTop + 13;
            const arrowBaseY = laneTop + 5;
            ctx.fillStyle = PIANO_ROLL_THEME.pedalOnset;
            ctx.beginPath();
            ctx.moveTo(x, arrowTipY);
            ctx.lineTo(x - 4, arrowBaseY);
            ctx.lineTo(x + 4, arrowBaseY);
            ctx.closePath();
            ctx.fill();
        });

        // Compact offset markers: a red arrow pointing up from the bottom of the pedal lane.
        pedals.forEach((pedal) => {
            if (pedal.offset_estimated === true) return;

            const pedalEnd = getPedalEnd(pedal);
            if (pedalEnd < startTime || pedalEnd > endTime) return;

            const x = (pedalEnd / duration) * virtualWidth - scrollLeft;
            if (x < -4 || x > backingWidth + 4) return;

            const arrowTipY = height - 13;
            const arrowBaseY = height - 5;
            ctx.fillStyle = PIANO_ROLL_THEME.pedalOffset;
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
        ctx.shadowColor = PIANO_ROLL_THEME.playheadShadow;
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
        ctx.font = "700 11px Inter, sans-serif";
        const labelWidth = ctx.measureText(label).width + 10;
        const labelX = clamp(x + 6, 2, Math.max(2, backingWidth - labelWidth - 2));
        ctx.fillStyle = PIANO_ROLL_THEME.playheadLabel;
        ctx.fillRect(labelX, 20, labelWidth, 18);
        ctx.fillStyle = PIANO_ROLL_THEME.playheadText;
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

        ctx.strokeStyle = PIANO_ROLL_THEME.timeLine;
        ctx.font = "600 11px Inter, sans-serif";

        for (let t = firstTick; t <= endTime + tickStep; t += tickStep) {
            if (t < 0) continue;
            const x = (t / duration) * virtualWidth - scrollLeft;
            if (x < -1 || x > backingWidth + 1) continue;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, PIANO_ROLL_HEIGHT);
            ctx.stroke();
            drawLabelWithHalo(`${Math.round(t)}s`, x + 4, 14, PIANO_ROLL_THEME.mutedText);
        }
    }

    function drawLabelWithHalo(text, x, y, color = PIANO_ROLL_THEME.text) {
        ctx.save();
        ctx.lineJoin = "round";
        ctx.miterLimit = 2;
        ctx.strokeStyle = PIANO_ROLL_THEME.textHalo;
        ctx.lineWidth = 4;
        ctx.strokeText(text, x, y);
        ctx.fillStyle = color;
        ctx.fillText(text, x, y);
        ctx.restore();
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    // --- Start the app ---
    init();
});
