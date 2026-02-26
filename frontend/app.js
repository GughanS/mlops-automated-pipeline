document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const sunIcon = document.querySelector('.sun-icon');
    const moonIcon = document.querySelector('.moon-icon');
    const html = document.documentElement;
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('span');
    const loader = document.querySelector('.loader');
    const resultContainer = document.getElementById('result-container');
    const resultBadge = document.getElementById('result-badge');
    const resultText = document.getElementById('result-text');
    const resultDetails = document.getElementById('result-details');

    // Theme toggling
    themeToggle.addEventListener('click', () => {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', newTheme);

        if (newTheme === 'dark') {
            sunIcon.style.display = 'block';
            moonIcon.style.display = 'none';
        } else {
            sunIcon.style.display = 'none';
            moonIcon.style.display = 'block';
        }
    });

    // Auto-fill mock transaction
    const autoFillBtn = document.getElementById('autofill-btn');
    if (autoFillBtn) {
        autoFillBtn.addEventListener('click', () => {
            document.getElementById('Time').value = 0;
            document.getElementById('Amount').value = 149.62;
            document.getElementById('pca_vector').value = "-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215";
        });
    }

    // Form Submissions
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI State update
        submitBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        loader.style.display = "block";
        resultContainer.classList.add('hidden');
        resultBadge.className = "result-badge";

        // Gather data
        const formData = new FormData(form);
        const timeVal = parseFloat(formData.get('Time'));
        const amountVal = parseFloat(formData.get('Amount'));
        const pcaString = formData.get('pca_vector').toString();

        const pcaValues = pcaString.split(',').map(v => parseFloat(v.trim()));

        if (pcaValues.length !== 28) {
            alert("Please ensure the PCA vector contains exactly 28 comma-separated numerical values.");
            submitBtn.disabled = false;
            btnText.textContent = "Analyze Transaction";
            loader.style.display = "none";
            return;
        }

        const data = {
            Time: timeVal,
            Amount: amountVal
        };

        for (let i = 0; i < 28; i++) {
            data[`V${i + 1}`] = pcaValues[i];
        }

        try {
            // Send to FastAPI
            // We use the relative URL here assuming the UI is served from the same domain
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const err = await response.json();
                let errMsg = 'Prediction failed';
                if (err.detail) {
                    if (Array.isArray(err.detail)) {
                        errMsg = err.detail.map(e => `${e.loc && e.loc.length > 1 ? e.loc[1] : 'Field'}: ${e.msg}`).join(' | ');
                    } else {
                        errMsg = err.detail;
                    }
                }
                throw new Error(errMsg);
            }

            const result = await response.json();

            // Show result
            resultContainer.classList.remove('hidden');

            if (result.is_fraud === 1) {
                resultBadge.classList.add('is-fraud');
                resultText.textContent = "Fraud Detected";
                resultDetails.textContent = "This transaction exhibits characteristics of fraudulent activity based on the current context models.";
            } else {
                resultBadge.classList.add('is-safe');
                resultText.textContent = "Transaction Safe";
                resultDetails.textContent = "This transaction appears legitimate according to the baseline distribution logs.";
            }

        } catch (error) {
            resultContainer.classList.remove('hidden');
            resultBadge.classList.add('is-fraud');
            resultText.textContent = "Error";
            resultDetails.textContent = error.message;
        } finally {
            submitBtn.disabled = false;
            btnText.textContent = "Analyze Transaction";
            loader.style.display = "none";
        }
    });
});
