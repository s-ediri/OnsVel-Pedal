
document.addEventListener("DOMContentLoaded", () => {
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
    const sheetMusicContainer = document.getElementById("sheet-music-container");
    const dropZone = document.getElementById("drop-zone");

    // --- State ---
    let audioBlob = null;
    let modelFile = null;
    let mediaRecorder;
    let recordingChunks = [];
    let timerInterval;
    let transcriptionData = null;
    let visualObj;

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
            if (modelSelect.value === "browse") {
                modelFileInput.click();
            }
            checkTranscribeButtonState();
        });

        modelFileInput.addEventListener("change", (event) => {
            modelFile = event.target.files[0];
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
        playBtn.addEventListener("click", playTranscription);

        // Drag and drop events
        dropZone.addEventListener("dragenter", (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });

        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });

        dropZone.addEventListener("dragleave", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                audioFileInput.files = files;
                handleFileSelect({ target: { files: files } });
            }
        });
    }

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            audioBlob = file;
            fileNameSpan.textContent = file.name;
            checkTranscribeButtonState();
        }
    }

    // --- UI Updates ---
    function checkTranscribeButtonState() {
        transcribeBtn.disabled = !(modelSelect.value && audioBlob);
    }

    function updateStatus(message, type = "info") {
        statusMessage.textContent = message;
        statusDiv.className = `status-area ${type}`; // for potential styling
        if (type === "loading") {
            loadingSpinner.style.display = "block";
            transcribeBtn.disabled = true;
        } else {
            loadingSpinner.style.display = "none";
            checkTranscribeButtonState();
        }
    }

    // --- Recording Logic ---
    async function startRecording() {
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
        if (!modelSelect.value || !audioBlob) {
            updateStatus("Please select a model and provide audio.", "error");
            return;
        }

        updateStatus("Transcribing... this may take a moment.", "loading");
        resultsDiv.style.display = "none";

        const formData = new FormData();
        formData.append("audio", audioBlob);
        if (modelFile) {
            formData.append("model_file", modelFile);
        } else {
            formData.append("model", modelSelect.value);
        }

        try {
            const response = await fetch("/api/transcribe", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Transcription failed.");
            }

            transcriptionData = data;
            updateStatus("Transcription complete!", "success");
            displayResults(data);
        } catch (error) {
            console.error("Transcription error:", error);
            updateStatus(`Error: ${error.message}`, "error");
        }
    }

    // --- Results Display and Playback ---
    function displayResults(data) {
        try {
            resultsDiv.style.display = "block";
            drawPianoRoll(data.notes, data.pedals, data.duration);
            renderSheetMusic(data.notes);
        } catch (error) {
            console.error("Error displaying results:", error);
            updateStatus(`Error displaying results: ${error.message}`, "error");
        }
    }

    function renderSheetMusic(notes) {
        const abcString = notesToAbc(notes);
        visualObj = ABCJS.renderAbc(sheetMusicContainer, abcString, {
            responsive: "resize",
        })[0];
    }

    function notesToAbc(notes) {
        let abc = "X:1\nT:Piano Transcription\nM:4/4\nK:C\nL:1/8\n";
        let stream = "";

        notes.forEach((note) => {
            const pitch = Tone.Frequency(note.pitch, "midi").toNote();
            const abcPitch = pitch.replace("#", "^").replace("b", "_");
            stream += abcPitch + " ";
        });

        return abc + stream;
    }

    function playTranscription() {
        if (!transcriptionData || !transcriptionData.notes) return;

        Tone.Transport.cancel();
        Tone.Transport.stop();

        const synth = new Tone.PolySynth(Tone.Synth, {
            oscillator: { type: "sine" },
            envelope: { attack: 0.01, decay: 0.1, sustain: 0.3, release: 1 },
        }).toDestination();

        transcriptionData.notes.forEach((note) => {
            synth.triggerAttackRelease(
                Tone.Frequency(note.pitch, "midi"),
                note.duration,
                note.start,
                note.velocity
            );
        });

        Tone.Transport.start();
    }

    // --- Piano Roll Visualization ---
    function drawPianoRoll(notes, pedals, duration) {
        const minPitch = 21; // A0
        const maxPitch = 108; // C8
        const pitchRange = maxPitch - minPitch;

        const width = Math.max(1000, duration * 50); // 50 pixels per second
        const height = 400;
        canvas.width = width;
        canvas.height = height;

        const keyHeight = height / pitchRange;
        const pedalHeight = 20;

        // Clear canvas
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

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
                ctx.lineTo(width, y);
                ctx.stroke();
                ctx.fillText(`C${Math.floor(p / 12) - 1}`, 5, y - 5);
            }
        }

        // Draw pedals
        ctx.fillStyle = "rgba(0, 123, 255, 0.2)";
        pedals.forEach((pedal) => {
            const x = (pedal.start / duration) * width;
            const w = (pedal.duration / duration) * width;
            ctx.fillRect(x, height - pedalHeight, w, pedalHeight);
        });
        ctx.strokeStyle = "rgba(0, 123, 255, 0.5)";
        ctx.strokeRect(0, height - pedalHeight, width, pedalHeight);

        // Draw notes
        notes.forEach((note) => {
            const y = height - (note.pitch - minPitch) * keyHeight;
            const x = (note.start / duration) * width;
            const w = (note.duration / duration) * width;

            ctx.fillStyle = `rgba(0, 0, 0, ${note.velocity * 0.8 + 0.2})`;
            ctx.fillRect(x, y - keyHeight, w, keyHeight);
        });
    }

    // --- Start the app ---
    init();
});
