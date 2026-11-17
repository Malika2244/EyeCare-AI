"use strict";

// =====================================
// 1. PAGE NAVIGATION
// =====================================

function showPage(pageId) {
    document.querySelectorAll(".page").forEach(page => {
        page.classList.remove("active-page");
    });

    document.querySelectorAll(".nav-link").forEach(link => {
        link.classList.remove("active");
    });

    const page = document.getElementById(pageId);
    if (!page) return;

    page.classList.add("active-page");

    const link = document.querySelector(
        `.nav-link[data-page="${pageId}"]`
    );

    if (link) link.classList.add("active");

    const titles = {
        dashboard: "Dashboard",
        scanner: "Retinal Image Scanner",
        gradcam: "Grad-CAM Visualization",
        disease: "Disease Information",
        care: "Eye Care Recommendations",
        history: "Prediction History",
        surgery: "Eye Surgery Guide",
        chatbot: "AI EyeCare Chatbot"
    };

    const pageTitle = document.getElementById("page-title");

    if (pageTitle) {
        pageTitle.textContent = titles[pageId] || "EyeCare AI";
    }

    if (pageId === "history") updateHistory();
    if (pageId === "chatbot") scrollChatToBottom();

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}

document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", event => {
        event.preventDefault();
        showPage(link.dataset.page);
    });
});

document.querySelectorAll("[data-go]").forEach(button => {
    button.addEventListener("click", () => {
        showPage(button.dataset.go);
    });
});


// =====================================
// 2. IMAGE UPLOAD ELEMENTS
// =====================================

const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");

const previewPlaceholder = document.getElementById(
    "preview-placeholder"
);

const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const uploadArea = document.getElementById("upload-area");

const scannerMessage = document.getElementById(
    "scanner-message"
);

let selectedImage = null;
let imageObjectURL = null;
let latestHeatmapURL = "";


// =====================================
// 3. PREDICTION HISTORY STORAGE
// =====================================

let predictionHistory = [];

function loadHistory() {
    try {
        const savedHistory = localStorage.getItem(
            "eyeCareHistory"
        );

        predictionHistory = savedHistory
            ? JSON.parse(savedHistory)
            : [];

        if (!Array.isArray(predictionHistory)) {
            predictionHistory = [];
        }

    } catch (error) {
        console.error("History loading error:", error);
        predictionHistory = [];
    }

    updateHistory();
}

function saveHistory() {
    predictionHistory = predictionHistory.slice(0, 50);

    try {
        localStorage.setItem(
            "eyeCareHistory",
            JSON.stringify(predictionHistory)
        );
    } catch (error) {
        console.error("History saving error:", error);
    }
}

function updateHistory() {
    const historyList = document.getElementById(
        "history-list"
    );

    const emptyMessage = document.getElementById(
        "history-empty"
    );

    if (!historyList || !emptyMessage) return;

    historyList.replaceChildren();

    emptyMessage.style.display =
        predictionHistory.length ? "none" : "block";

    predictionHistory.forEach(item => {
        const card = document.createElement("div");
        card.className = "history-item";

        const title = document.createElement("h3");
        title.textContent = item.disease;

        const confidence = document.createElement("p");
        confidence.textContent =
            "Confidence: " + item.confidence + "%";

        const status = document.createElement("p");

        if (item.threshold != null) {
            const isUncertain =
                item.isUncertain === true ||
                Number(item.confidence) < Number(item.threshold);

            status.textContent = isUncertain
                ? "Status: Uncertain Prediction"
                : "Status: Accepted";

            status.style.color = isUncertain
                ? "#b7791f"
                : "#198754";

            status.style.fontWeight = "600";
        }

        const imageName = document.createElement("p");
        imageName.textContent =
            "Image: " + (item.imageName || "Uploaded image");

        const date = document.createElement("small");
        date.textContent = item.date;

        card.append(
            title,
            confidence,
            status,
            imageName,
            date
        );

        historyList.appendChild(card);
    });
}

const clearHistoryBtn = document.getElementById(
    "clear-history"
);

if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", () => {
        predictionHistory = [];
        localStorage.removeItem("eyeCareHistory");
        updateHistory();
    });
}


// =====================================
// 4. CONFIDENCE THRESHOLD SYSTEM
// =====================================

// Remove previous prediction status
function clearThresholdNotice() {
    const previousNotice = document.getElementById(
        "confidence-threshold-notice"
    );

    if (previousNotice) {
        previousNotice.remove();
    }
}


// Display accepted or uncertain status
function showThresholdNotice(result) {

    clearThresholdNotice();

    const resultBox = document.getElementById(
        "analysis-result"
    );

    if (!resultBox) return;

    const confidence = Number(result.confidence);
    const threshold = Number(result.threshold);

    if (
        !Number.isFinite(confidence) ||
        !Number.isFinite(threshold) ||
        result.threshold == null
    ) {
        return;
    }

    const isUncertain =
        result.is_uncertain === true ||
        confidence < threshold;

    const notice = document.createElement("div");

    notice.id = "confidence-threshold-notice";
    notice.setAttribute(
        "role",
        isUncertain ? "alert" : "status"
    );

    notice.style.cssText = `
        margin: 16px 0;
        padding: 18px;
        border-radius: 12px;
        line-height: 1.6;
        border: 1px solid;
        border-left: 5px solid;
    `;

    const heading = document.createElement("strong");
    heading.style.display = "block";
    heading.style.marginBottom = "8px";

    const details = document.createElement("p");
    details.style.margin = "0";

    if (isUncertain) {

        notice.style.background = "#fff8e8";
        notice.style.color = "#754b08";
        notice.style.borderColor = "#e6bd6a";
        notice.style.borderLeftColor = "#c68618";

        heading.textContent =
            "Uncertain Prediction - Low Confidence";

        details.textContent =
            `The model predicted ${result.disease} with ` +
            `${confidence.toFixed(2)}% confidence, ` +
            `which is below the ${threshold.toFixed(0)}% ` +
            `confidence threshold. This prediction is ` +
            `uncertain. Please consult an eye specialist.`;

    } else {

        notice.style.background = "#eaf8ef";
        notice.style.color = "#185b36";
        notice.style.borderColor = "#a7d9b7";
        notice.style.borderLeftColor = "#198754";

        heading.textContent =
            "Prediction Accepted - Confidence Threshold Met";

        details.textContent =
            `The model predicted ${result.disease} with ` +
            `${confidence.toFixed(2)}% confidence. ` +
            `This meets the ${threshold.toFixed(0)}% ` +
            `confidence threshold. This is an AI prediction, ` +
            `not a confirmed medical diagnosis.`;
    }

    notice.append(heading, details);

    resultBox.prepend(notice);
}


// =====================================
// 5. IMAGE PREVIEW
// =====================================

function previewImage(file) {
    if (!file) return false;

    const allowedTypes = [
        "image/jpeg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {
        alert("Please select a JPG, JPEG or PNG image.");
        return false;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert("Image size must be less than 10 MB.");
        return false;
    }

    selectedImage = file;

    if (imageObjectURL) {
        URL.revokeObjectURL(imageObjectURL);
    }

    imageObjectURL = URL.createObjectURL(file);

    imagePreview.src = imageObjectURL;
    imagePreview.style.display = "block";
    previewPlaceholder.style.display = "none";

    analyzeBtn.disabled = false;

    scannerMessage.textContent =
        "Image selected successfully.";

    clearThresholdNotice();

    document.getElementById(
        "analysis-result"
    ).hidden = true;

    return true;
}

if (imageInput) {
    imageInput.addEventListener("change", function () {
        if (!previewImage(this.files[0])) {
            this.value = "";
        }
    });
}


// =====================================
// 6. DRAG AND DROP
// =====================================

if (uploadArea) {

    uploadArea.addEventListener("dragover", event => {
        event.preventDefault();
        uploadArea.classList.add("dragging");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragging");
    });

    uploadArea.addEventListener("drop", event => {
        event.preventDefault();
        uploadArea.classList.remove("dragging");

        const file = event.dataTransfer.files[0];

        if (!file) return;
        if (!previewImage(file)) return;

        const transfer = new DataTransfer();
        transfer.items.add(file);

        imageInput.files = transfer.files;
    });
}


// =====================================
// 7. CLEAR SELECTED IMAGE
// =====================================

if (clearBtn) {

    clearBtn.addEventListener("click", () => {

        imageInput.value = "";
        selectedImage = null;

        if (imageObjectURL) {
            URL.revokeObjectURL(imageObjectURL);
            imageObjectURL = null;
        }

        imagePreview.src = "";
        imagePreview.style.display = "none";

        previewPlaceholder.style.display = "block";

        analyzeBtn.disabled = true;

        scannerMessage.textContent = "";

        clearThresholdNotice();

        document.getElementById(
            "analysis-result"
        ).hidden = true;
    });
}


// =====================================
// 8. DISPLAY LISTS
// =====================================

function displayList(elementId, items) {
    const list = document.getElementById(elementId);

    if (!list) return;

    list.replaceChildren();

    if (!Array.isArray(items) || items.length === 0) {
        const li = document.createElement("li");
        li.textContent = "No information available.";
        list.appendChild(li);
        return;
    }

    items.forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
    });
}


// =====================================
// 9. ANALYZE IMAGE WITH FLASK
// =====================================

if (analyzeBtn) {

    analyzeBtn.addEventListener("click", async () => {

        if (!selectedImage) {
            alert("Please upload an image first.");
            return;
        }

        const originalText = analyzeBtn.innerHTML;

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = "Analyzing Image...";

        scannerMessage.textContent =
            "Please wait. AI is analyzing your image.";

        const formData = new FormData();
        formData.append("image", selectedImage);

        try {

            const response = await fetch("/analyze", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (!response.ok || result.success === false) {
                throw new Error(
                    result.error || "Image analysis failed."
                );
            }

            // =====================================
            // DISEASE PREDICTION
            // =====================================

            document.getElementById(
                "prediction-name"
            ).textContent = result.disease;

            document.getElementById(
                "prediction-confidence"
            ).textContent = result.confidence + "%";


            // =====================================
            // UPLOADED IMAGE
            // =====================================

            document.getElementById(
                "uploaded-result-image"
            ).src = result.image_url;


            // =====================================
            // GRAD-CAM HEATMAP
            // =====================================

            document.getElementById(
                "heatmap-result-image"
            ).src = result.heatmap_url;


            // =====================================
            // DISEASE INFORMATION
            // =====================================

            const info = result.disease_info || {};

            document.getElementById(
                "disease-info-name"
            ).textContent = info.name || result.disease;

            document.getElementById(
                "disease-description"
            ).textContent =
                info.description || "No description available.";

            displayList(
                "disease-symptoms",
                info.symptoms
            );

            displayList(
                "disease-causes",
                info.causes
            );


            // =====================================
            // EYE CARE RECOMMENDATIONS
            // =====================================

            const care = result.eye_care || {};

            document.getElementById(
                "care-title"
            ).textContent = care.title || "General Eye Care";

            displayList(
                "care-recommendations",
                care.recommendations
            );


            // =====================================
            // DISPLAY RESULT + 80% THRESHOLD
            // =====================================

            document.getElementById(
                "analysis-result"
            ).hidden = false;

            // NEW: Display confidence status
            showThresholdNotice(result);


            // =====================================
            // UPDATE GRAD-CAM PAGE
            // =====================================

            latestHeatmapURL = result.heatmap_url;

            const gradcamImage = document.getElementById(
                "gradcam-page-image"
            );

            if (gradcamImage) {
                gradcamImage.src = latestHeatmapURL;
                gradcamImage.hidden = false;
            }


            // =====================================
            // SAVE PREDICTION HISTORY
            // =====================================

            predictionHistory.unshift({

                disease: result.disease,

                confidence: result.confidence,

                // New threshold information
                predictionStatus: result.prediction_status,

                isUncertain: result.is_uncertain,

                threshold: result.threshold,

                imageName: selectedImage.name,

                date: new Date().toLocaleString()
            });

            saveHistory();
            updateHistory();


            // =====================================
            // SUCCESS MESSAGE
            // =====================================

            const isUncertain = result.is_uncertain === true ||
                (
                    result.threshold != null &&
                    Number(result.confidence) <
                    Number(result.threshold)
                );

            scannerMessage.textContent = isUncertain
                ? "Analysis completed. Prediction is uncertain."
                : "Analysis completed successfully.";

            document.getElementById(
                "analysis-result"
            ).scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        } catch (error) {

            console.error("Analysis error:", error);

            scannerMessage.textContent = error.message;

            alert("Analysis Error: " + error.message);

        } finally {

            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = originalText;
        }
    });
}


// =====================================
// 10. EYE SURGERY GUIDE CHECKER
// =====================================

const surgeryForm = document.getElementById(
    "surgery-form"
);

if (surgeryForm) {

    surgeryForm.addEventListener("submit", function (event) {

        event.preventDefault();

        const ageInput = document.getElementById(
            "surgery-age"
        );

        const stability = document.getElementById(
            "prescription-stability"
        ).value;

        const dryEyes = document.getElementById(
            "dry-eyes"
        ).value;

        const eyeCondition = document.getElementById(
            "eye-condition"
        ).value;

        const corneaTest = document.getElementById(
            "cornea-test"
        ).value;

        const resultBox = document.getElementById(
            "surgery-result"
        );

        const resultTitle = document.getElementById(
            "surgery-result-title"
        );

        const resultMessage = document.getElementById(
            "surgery-result-message"
        );

        const resultList = document.getElementById(
            "surgery-result-list"
        );

        const age = Number(ageInput.value);

        if (
            ageInput.value.trim() === "" ||
            !Number.isFinite(age) ||
            age < 1 ||
            age > 120
        ) {
            alert(
                "Please enter a valid age between 1 and 120."
            );

            ageInput.focus();
            return;
        }

        const guidance = [];

        let needsSpecialistAssessment = false;
        let needsMoreInformation = false;


        // Age guidance
        if (age < 18) {

            guidance.push(
                "You are under 18. Elective laser vision correction is generally not offered to people under 18 because vision may still be changing."
            );

            needsSpecialistAssessment = true;

        } else if (age >= 40) {

            guidance.push(
                "At age 40 or above, near-vision changes and other age-related eye conditions should also be assessed."
            );

        } else {

            guidance.push(
                "Your age alone cannot determine whether laser vision correction is appropriate."
            );
        }


        // Prescription stability
        if (stability === "no") {

            guidance.push(
                "Your prescription is not stable. Changing vision needs professional assessment before considering surgery."
            );

            needsSpecialistAssessment = true;

        } else if (stability === "unknown") {

            guidance.push(
                "Your prescription stability is unknown. Previous eye prescriptions and a professional eye examination may help assess this."
            );

            needsMoreInformation = true;

        } else {

            guidance.push(
                "A stable prescription is a reassuring factor, but it does not establish surgical suitability."
            );
        }


        // Dry eyes
        if (dryEyes === "yes") {

            guidance.push(
                "Persistent dry eyes should be assessed and managed before surgery is considered."
            );

            needsSpecialistAssessment = true;

        } else if (dryEyes === "unknown") {

            guidance.push(
                "If you are unsure about dry eyes, ask an eye-care professional to evaluate your tear film and eye surface."
            );

            needsMoreInformation = true;

        } else {

            guidance.push(
                "No persistent dry-eye symptoms were reported. An eye examination is still needed."
            );
        }


        // Existing eye conditions
        if (eyeCondition === "yes") {

            guidance.push(
                "A reported cataract, glaucoma, or corneal condition requires specialist assessment. It may affect surgical planning."
            );

            needsSpecialistAssessment = true;

        } else if (eyeCondition === "unknown") {

            guidance.push(
                "Your eye-condition history is uncertain. A comprehensive eye examination is important."
            );

            needsMoreInformation = true;

        } else {

            guidance.push(
                "No listed eye condition was reported. This does not rule out other eye-health concerns."
            );
        }


        // Corneal measurements
        if (corneaTest === "no") {

            guidance.push(
                "Corneal thickness and shape measurements have not been completed. These tests are important for evaluating laser surgery options."
            );

            needsMoreInformation = true;

        } else if (corneaTest === "unknown") {

            guidance.push(
                "Confirm whether corneal thickness and shape tests have been completed and ask the surgeon to review the actual measurements."
            );

            needsMoreInformation = true;

        } else {

            guidance.push(
                "You report that corneal testing has been completed. The surgeon must review the actual measurements before making any decision."
            );
        }


        // Overall guidance
        if (needsSpecialistAssessment) {

            resultTitle.textContent =
                "Specialist Assessment Needed";

            resultMessage.textContent =
                "Your answers include factors that need professional review before considering elective vision correction.";

        } else if (needsMoreInformation) {

            resultTitle.textContent =
                "More Eye Information Needed";

            resultMessage.textContent =
                "Some important information or eye measurements are missing. A comprehensive eye examination is needed.";

        } else {

            resultTitle.textContent =
                "Some Initial Factors Are Reassuring";

            resultMessage.textContent =
                "Your answers do not identify the listed concerns, but only a qualified eye surgeon can assess whether any procedure is appropriate.";
        }


        resultList.replaceChildren();

        guidance.forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            resultList.appendChild(li);
        });

        resultBox.hidden = false;

        resultBox.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    });


    // Reset surgery questionnaire
    const resetSurgeryBtn = document.getElementById(
        "reset-surgery"
    );

    if (resetSurgeryBtn) {

        resetSurgeryBtn.addEventListener("click", function () {

            surgeryForm.reset();

            const resultBox = document.getElementById(
                "surgery-result"
            );

            resultBox.hidden = true;

            document.getElementById(
                "surgery-result-list"
            ).replaceChildren();
        });
    }
}


// =====================================
// 11. GEMINI AI CHATBOT - MODULE 8
// =====================================

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send-btn");

const chatMessages = document.getElementById(
    "chat-messages"
);

const chatClearBtn = document.getElementById(
    "chat-clear-btn"
);

const chatStatus = document.getElementById(
    "chat-status"
);

let chatHistory = [];
let chatRequestController = null;
let chatRequestNumber = 0;


// =====================================
// 12. CHATBOT INITIAL MESSAGE
// =====================================

const CHAT_WELCOME_MESSAGE =
    "Hello! I'm your EyeCare AI Assistant.\n\n" +
    "You can ask me questions about cataract, " +
    "glaucoma, diabetic retinopathy, eye " +
    "examinations and general eye-care habits.\n\n" +
    "How can I help you today?";


// =====================================
// 13. SCROLL CHAT TO BOTTOM
// =====================================

function scrollChatToBottom() {
    if (!chatMessages) return;

    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// =====================================
// 14. DISPLAY CHAT MESSAGE
// =====================================

function addChatMessage(role, message) {
    if (!chatMessages) return null;

    const messageRow = document.createElement("div");

    messageRow.className =
        "chat-message " +
        (role === "user" ? "user-message" : "bot-message");

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";

    // Prevent HTML injection
    bubble.textContent = message;

    messageRow.appendChild(bubble);
    chatMessages.appendChild(messageRow);

    scrollChatToBottom();

    return messageRow;
}


// =====================================
// 15. TYPING INDICATOR
// =====================================

function showChatLoading() {
    if (!chatMessages) return null;

    const messageRow = document.createElement("div");

    messageRow.className =
        "chat-message bot-message chat-loading";

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";

    bubble.setAttribute(
        "aria-label",
        "AI assistant is preparing a reply"
    );

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        dot.className = "typing-dot";
        bubble.appendChild(dot);
    }

    messageRow.appendChild(bubble);
    chatMessages.appendChild(messageRow);

    scrollChatToBottom();

    return messageRow;
}


// =====================================
// 16. CHATBOT BUTTON STATE
// =====================================

function setChatBusy(isBusy) {

    if (chatInput) {
        chatInput.disabled = isBusy;
    }

    if (chatSendBtn) {

        chatSendBtn.disabled = isBusy;

        if (isBusy) {
            chatSendBtn.textContent = "Thinking...";
        } else {
            chatSendBtn.innerHTML =
                '<i class="fa-solid fa-paper-plane"></i> Send';
        }
    }

    document.querySelectorAll(
        ".chat-suggestion"
    ).forEach(button => {
        button.disabled = isBusy;
    });
}


// =====================================
// 17. UPDATE CHAT STATUS
// =====================================

function setChatStatus(message) {
    if (!chatStatus) return;
    chatStatus.textContent = message;
}


// =====================================
// 18. SEND MESSAGE TO GEMINI AI
// =====================================

async function sendChatMessage(message) {

    if (
        !chatForm ||
        !chatInput ||
        !chatMessages
    ) {
        return;
    }

    if (chatRequestController) return;
    if (typeof message !== "string") return;

    const userMessage = message.trim();

    if (!userMessage) {
        setChatStatus("Please enter a message.");
        return;
    }

    if (userMessage.length > 2000) {
        setChatStatus(
            "Your message must not exceed 2000 characters."
        );
        return;
    }

    const previousHistory = chatHistory.slice(-10);

    addChatMessage("user", userMessage);

    chatInput.value = "";

    setChatBusy(true);

    setChatStatus(
        "Gemini AI is preparing your answer..."
    );

    const loadingElement = showChatLoading();

    const controller = new AbortController();

    chatRequestController = controller;

    const currentRequestNumber = ++chatRequestNumber;

    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: userMessage,
                history: previousHistory
            }),

            signal: controller.signal
        });

        let result;

        try {
            result = await response.json();
        } catch (parseError) {
            throw new Error(
                "Invalid response from the AI server."
            );
        }

        if (!response.ok || !result.success) {
            throw new Error(
                result.error ||
                "Unable to generate an AI response."
            );
        }

        if (
            typeof result.reply !== "string" ||
            !result.reply.trim()
        ) {
            throw new Error(
                "The AI assistant returned an empty response."
            );
        }

        if (
            currentRequestNumber !== chatRequestNumber
        ) {
            return;
        }

        if (loadingElement) {
            loadingElement.remove();
        }

        const assistantReply = result.reply.trim();

        addChatMessage("model", assistantReply);

        chatHistory.push({
            role: "user",
            text: userMessage
        });

        chatHistory.push({
            role: "model",
            text: assistantReply
        });

        chatHistory = chatHistory.slice(-10);

        setChatStatus("");

    } catch (error) {

        if (loadingElement) {
            loadingElement.remove();
        }

        if (
            currentRequestNumber !== chatRequestNumber
        ) {
            return;
        }

        if (error.name !== "AbortError") {

            console.error("Chatbot error:", error);

            addChatMessage(
                "model",
                "Sorry, I could not process your question. " +
                "Please try again."
            );

            setChatStatus(
                error.message ||
                "Chatbot connection failed."
            );
        }

    } finally {

        if (
            currentRequestNumber === chatRequestNumber
        ) {
            chatRequestController = null;

            setChatBusy(false);

            chatInput.focus();
        }
    }
}


// =====================================
// 19. CHAT FORM SUBMISSION
// =====================================

if (chatForm && chatInput) {

    chatForm.addEventListener(
        "submit",
        async function (event) {

            event.preventDefault();

            const message = chatInput.value;

            await sendChatMessage(message);
        }
    );
}


// =====================================
// 20. ENTER KEY TO SEND MESSAGE
// =====================================

if (chatInput) {

    chatInput.addEventListener("keydown", event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.isComposing
        ) {

            event.preventDefault();

            if (!chatRequestController) {
                sendChatMessage(chatInput.value);
            }
        }
    });
}


// =====================================
// 21. SUGGESTED CHAT QUESTIONS
// =====================================

document.querySelectorAll(
    ".chat-suggestion"
).forEach(button => {

    button.addEventListener("click", () => {

        const question = button.dataset.question;

        if (!question) return;

        sendChatMessage(question);
    });
});


// =====================================
// 22. CLEAR CHAT CONVERSATION
// =====================================

function clearChatConversation() {

    chatRequestNumber++;

    if (chatRequestController) {
        chatRequestController.abort();
        chatRequestController = null;
    }

    chatHistory = [];

    if (chatMessages) {
        chatMessages.replaceChildren();
    }

    addChatMessage(
        "model",
        CHAT_WELCOME_MESSAGE
    );

    if (chatInput) {
        chatInput.value = "";
    }

    setChatBusy(false);

    setChatStatus("Conversation cleared.");

    if (chatInput) {
        chatInput.focus();
    }
}

if (chatClearBtn) {

    chatClearBtn.addEventListener(
        "click",
        clearChatConversation
    );
}


// =====================================
// 23. INITIAL PAGE
// =====================================

showPage("dashboard");

loadHistory();


// =====================================
// 24. CHATBOT INITIALIZATION
// =====================================

if (chatMessages) {

    chatMessages.replaceChildren();

    addChatMessage(
        "model",
        CHAT_WELCOME_MESSAGE
    );
}

console.log(
    "EyeCare AI frontend initialized with 80% confidence threshold."
);