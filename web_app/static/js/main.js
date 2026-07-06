
document.addEventListener("DOMContentLoaded", () => {
    // --- Upload limits (kept in sync with web_app/app.py AppConfig) ---
    const MAX_AUDIO_FILE_SIZE_BYTES = 25 * 1024 * 1024;
    const MAX_AUDIO_FILE_SIZE_LABEL = "25MB";
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
    const notationPreviewContainer = document.getElementById("notation-preview-container");
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
    let visualObj;
    let isTranscribing = false;
    let currentRequest = null;
    let playbackSynth = null;
    let pianoRollSpacer = null;
    let pianoRollFrameRequest = null;
    let pianoRollState = null;
    let pianoRollResizeTimeout = null;

    const PIANO_ROLL_HEIGHT = 400;
    const PIANO_ROLL_MIN_WIDTH = 1000;
    const PIANO_ROLL_PX_PER_SECOND = 50;
    const PIANO_ROLL_MAX_BACKING_WIDTH = 2400;
    const PIANO_ROLL_RESIZE_DEBOUNCE_MS = 100;
    const APPROXIMATE_PREVIEW_MAX_NOTES = 512;

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
        playBtn.addEventListener("click", playTranscription);
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
        audioFileInput.value = "";
        fileNameSpan.textContent = " or drag and drop an audio file here";
        checkTranscribeButtonState();
    }

    function validateAudioFile(file) {
        if (file.size > MAX_AUDIO_FILE_SIZE_BYTES) {
            return `File is too large (${formatBytes(file.size)}). Maximum allowed size is ${MAX_AUDIO_FILE_SIZE_LABEL}.`;
        }

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
            drawPianoRoll(data.notes, data.pedals, data.duration);
            renderApproximatePitchPreview(data.notes);
        } catch (error) {
            console.error("Error displaying results:", error);
            updateStatus(`Error displaying results: ${error.message}`, "error");
        }
    }

    function renderApproximatePitchPreview(notes) {
        notationPreviewContainer.innerHTML = "";

        if (!Array.isArray(notes) || notes.length === 0) {
            notationPreviewContainer.textContent = "No detected notes to preview.";
            visualObj = null;
            return;
        }

        const previewableNotes = getPreviewableNotes(notes);

        if (previewableNotes.length === 0) {
            notationPreviewContainer.textContent = "No valid detected notes to preview.";
            visualObj = null;
            return;
        }

        if (!window.ABCJS || typeof ABCJS.renderAbc !== "function") {
            notationPreviewContainer.textContent = "Approximate pitch preview is unavailable because ABCJS did not load.";
            visualObj = null;
            return;
        }

        const renderedNotes = previewableNotes.slice(0, APPROXIMATE_PREVIEW_MAX_NOTES);
        const abcString = notesToApproximateAbc(renderedNotes);

        try {
            visualObj = ABCJS.renderAbc(notationPreviewContainer, abcString, {
                responsive: "resize",
            })[0] || null;
        } catch (error) {
            console.error("Approximate pitch preview render failed:", error);
            notationPreviewContainer.textContent = "Approximate pitch preview could not be rendered.";
            visualObj = null;
            return;
        }

        if (previewableNotes.length > renderedNotes.length) {
            const truncationNote = document.createElement("p");
            truncationNote.className = "preview-truncation-note";
            truncationNote.textContent = `Preview truncated: showing first ${renderedNotes.length} of ${previewableNotes.length} detected notes.`;
            notationPreviewContainer.appendChild(truncationNote);
        }
    }

    function getPreviewableNotes(notes) {
        return notes
            .filter((note) => isFiniteNote(note) && isValidMidiPitch(note.pitch))
            .sort((a, b) => Number(a.start) - Number(b.start) || Number(a.pitch) - Number(b.pitch));
    }

    function notesToApproximateAbc(notes) {
        let abc = "X:1\nT:Approximate Pitch Preview (Not Quantized Sheet Music)\nM:none\nL:1/8\nK:C\n";
        const tokens = [];

        notes
            .forEach((note) => {
                tokens.push(midiPitchToApproximateAbc(Number(note.pitch)));
            });

        if (tokens.length === 0) {
            return `${abc}z8`;
        }

        const lines = [];
        for (let i = 0; i < tokens.length; i += 32) {
            lines.push(tokens.slice(i, i + 32).join(" "));
        }

        return abc + lines.join("\n");
    }

    function midiPitchToApproximateAbc(midiPitch) {
        const pitchClasses = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"];
        const roundedPitch = Math.round(Number(midiPitch));
        const pitchClass = pitchClasses[((roundedPitch % 12) + 12) % 12];
        const octave = Math.floor(roundedPitch / 12) - 1;
        const abcBaseOctave = 4;
        const octaveDelta = octave - abcBaseOctave;

        if (octaveDelta > 0) {
            return pitchClass.toLowerCase() + "'".repeat(Math.max(0, octaveDelta - 1));
        }

        if (octaveDelta < 0) {
            return pitchClass + ",".repeat(Math.abs(octaveDelta));
        }

        return pitchClass;
    }

    function isValidMidiPitch(pitch) {
        const numericPitch = Number(pitch);
        return Number.isFinite(numericPitch) && numericPitch >= 0 && numericPitch <= 127;
    }

    async function playTranscription() {
        if (!transcriptionData || !transcriptionData.notes) return;

        playBtn.disabled = true;
        playBtn.textContent = "Starting audio...";

        try {
            // Browser autoplay policies require AudioContext startup from a user gesture.
            await Tone.start();
        } catch (error) {
            console.error("Tone.js audio start failed:", error);
            updateStatus("Audio playback could not start. Click Play again or check browser audio permissions.", "error");
            playBtn.disabled = false;
            playBtn.textContent = "Play Transcription";
            return;
        }

        stopPlayback();

        playbackSynth = new Tone.PolySynth(Tone.Synth, {
            oscillator: { type: "sine" },
            envelope: { attack: 0.01, decay: 0.1, sustain: 0.3, release: 1 },
        }).toDestination();

        const notes = transcriptionData.notes.filter((note) => isFiniteNote(note));
        const duration = Math.max(Number(transcriptionData.duration) || 0, getLastNoteEnd(notes));

        Tone.Transport.seconds = 0;
        Tone.Transport.bpm.value = 120;

        notes.forEach((note) => {
            Tone.Transport.schedule((time) => {
                playbackSynth.triggerAttackRelease(
                    Tone.Frequency(note.pitch, "midi"),
                    Math.max(0.01, Number(note.duration) || 0.01),
                    time,
                    clamp(Number(note.velocity) || 0.8, 0, 1)
                );
            }, Math.max(0, Number(note.start) || 0));
        });

        Tone.Transport.scheduleOnce(() => {
            stopPlayback(true);
        }, duration + 1);

        Tone.Transport.start("+0.05");
        playBtn.disabled = false;
        playBtn.textContent = "Restart Playback";
    }

    function stopPlayback(resetButton = true) {
        Tone.Transport.stop();
        Tone.Transport.cancel(0);

        if (playbackSynth) {
            playbackSynth.releaseAll(Tone.now());
            playbackSynth.dispose();
            playbackSynth = null;
        }

        if (resetButton) {
            playBtn.textContent = "Play Transcription";
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

    // --- Piano Roll Visualization ---
    function drawPianoRoll(notes, pedals, duration, resetScroll = true) {
        const minPitch = 21; // A0
        const maxPitch = 108; // C8
        const pitchRange = maxPitch - minPitch;
        const safeDuration = Math.max(Number(duration) || 0, getLastNoteEnd(notes || []), 1);
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
            pitchRange,
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
        pianoRollSpacer.style.height = "1px";
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
            pitchRange,
            virtualWidth,
            backingWidth,
            height,
        } = pianoRollState;

        const scrollLeft = pianoRollContainer.scrollLeft || 0;
        const visibleWidth = backingWidth;
        const startTime = (scrollLeft / virtualWidth) * duration;
        const endTime = ((scrollLeft + visibleWidth) / virtualWidth) * duration;

        const keyHeight = height / pitchRange;
        const pedalHeight = 20;

        // Clear canvas
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, backingWidth, height);

        // Draw grid (octave lines and key labels)
        ctx.strokeStyle = "#eee";
        ctx.lineWidth = 1;
        ctx.fillStyle = "#999";
        ctx.font = "12px sans-serif";

        for (let p = minPitch; p <= maxPitch; p++) {
            const y = height - (p - minPitch) * keyHeight;
            if ((p - 21) % 12 === 0) {
                // C notes
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(backingWidth, y);
                ctx.stroke();
                ctx.fillText(`C${Math.floor(p / 12) - 1}`, 5, y - 5);
            }
        }

        drawTimeTicks(startTime, endTime, duration, virtualWidth, scrollLeft, backingWidth);

        // Draw pedals
        ctx.fillStyle = "rgba(0, 123, 255, 0.2)";
        pedals.forEach((pedal) => {
            const pedalStart = Number(pedal.start) || 0;
            const pedalDuration = Math.max(0, Number(pedal.duration) || 0);
            const pedalEnd = pedalStart + pedalDuration;
            if (pedalEnd < startTime || pedalStart > endTime) return;

            const x = (pedalStart / duration) * virtualWidth - scrollLeft;
            const w = Math.max(1, (pedalDuration / duration) * virtualWidth);
            ctx.fillRect(x, height - pedalHeight, w, pedalHeight);
        });
        ctx.strokeStyle = "rgba(0, 123, 255, 0.5)";
        ctx.strokeRect(0, height - pedalHeight, backingWidth, pedalHeight);

        // Draw notes
        notes.forEach((note) => {
            const noteStart = Number(note.start) || 0;
            const noteDuration = Math.max(0, Number(note.duration) || 0);
            const noteEnd = noteStart + noteDuration;
            if (noteEnd < startTime || noteStart > endTime) return;

            const y = height - (note.pitch - minPitch) * keyHeight;
            const x = (noteStart / duration) * virtualWidth - scrollLeft;
            const w = Math.max(1, (noteDuration / duration) * virtualWidth);

            ctx.fillStyle = `rgba(0, 0, 0, ${clamp(Number(note.velocity) || 0.8, 0, 1) * 0.8 + 0.2})`;
            ctx.fillRect(x, y - keyHeight, w, keyHeight);
        });
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
