document.addEventListener("DOMContentLoaded", function () {
    // DOM Elements
    const textInput = document.getElementById("textInput");
    const fileInput = document.getElementById("fileInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultDiv = document.getElementById("result");
    const sentimentText = document.getElementById("sentimentText");
    const confidenceLevel = document.getElementById("confidenceLevel");

    if (!textInput || !fileInput || !analyzeBtn || !resultDiv || !sentimentText || !confidenceLevel) {
        console.error("❌ Missing HTML elements! Ensure the IDs match.");
        return;
    }

    // Function to analyze sentiment from text input
    async function analyzeTextSentiment(text) {
        try {
            const response = await fetch("https://sentiment-analysis-model-unqh.onrender.com/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP Error! Status: ${response.status}`);
            }

            const data = await response.json();
            updateResults(data);
        } catch (error) {
            console.error("Error:", error);
            alert(`⚠️ API Error: ${error.message}`);
        }
    }

    // Function to analyze sentiment from an uploaded file
    async function analyzeFileSentiment() {
        const file = fileInput.files[0];
        if (!file) {
            alert("⚠️ Please select a file to upload!");
            return;
        }

        const allowedFormats = ["application/pdf", "text/plain", "image/png", "image/jpeg", "image/jpg"];
        if (!allowedFormats.includes(file.type)) {
            alert("⚠️ Invalid file type! Only PDF, TXT, PNG, JPEG allowed.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("https://sentiment-analysis-model-unqh.onrender.com/upload", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP Error! Status: ${response.status}`);
            }

            const data = await response.json();
            updateResults(data);
        } catch (error) {
            console.error("Error:", error);
            alert(`⚠️ API Error: ${error.message}`);
        }
    }
    // Scroll-progress bar
    window.addEventListener('scroll', () => {
        const doc = document.documentElement;
        const scrollPercent = (doc.scrollTop) / (doc.scrollHeight - doc.clientHeight);
        document.getElementById('scrollProgress')
        .style.height = `${scrollPercent * 100}vh`;
    });
  

    // Function to update UI with Sentiment Results
    function updateResults(data) {
        resultDiv.classList.remove("hidden");

        if (!sentimentText || !confidenceLevel) {
            console.error("❌ Error: Missing elements for results update!");
            return;
        }

        sentimentText.textContent = data.sentiment.toUpperCase();
        confidenceLevel.textContent = Math.round(data.confidence * 100) + "%";

        // Convert confidence to percentages (Assuming confidence means probability)
        let pos = data.sentiment === "positive" ? data.confidence * 100 : 100 - (data.confidence * 100);
        let neg = 100 - pos;

        updateCharts(pos, neg);
    }

    // Pie Chart & Bar Graph for Sentiment Trends
    let ctxPie = document.getElementById("sentimentPieChart").getContext("2d");
    let pieChart = new Chart(ctxPie, {
        type: "pie",
        data: {
            labels: ["Positive 😊", "Negative 😠"],
            datasets: [{
                data: [50, 50], // Default values (gets updated later)
                backgroundColor: ["green", "red"]
            }]
        },
        options: { responsive: true }
    });

    let ctxBar = document.getElementById("sentimentBarChart").getContext("2d");
    let barChart = new Chart(ctxBar, {
        type: "bar",
        data: {
            labels: ["Positive 😊", "Negative 😠"],
            datasets: [{
                label: "Sentiment Analysis",
                backgroundColor: ["green", "red"],
                data: [50, 50]  // Default values (gets updated later)
            }]
        },
        options: { responsive: true }
    });

    // Function to update charts dynamically with real results
    function updateCharts(pos, neg) {
        pieChart.data.datasets[0].data = [pos, neg];
        pieChart.update();

        barChart.data.datasets[0].data = [pos, neg];
        barChart.update();
    }

    // Event Listener for the Analyze Button
    analyzeBtn.addEventListener("click", function () {
        const text = textInput.value.trim();
        if (text) {
            analyzeTextSentiment(text);
        } else {
            analyzeFileSentiment();
        }
    });
});
