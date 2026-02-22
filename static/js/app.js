/**
 * InsurAI - Client-side Application Logic
 * Handles prediction form, gauge animation, leaderboard, and toast notifications.
 */

// ─── DOM References ───
const predictForm = document.getElementById('predictForm');
const ageSlider = document.getElementById('ageSlider');
const ageDisplay = document.getElementById('ageDisplay');
const salaryInput = document.getElementById('salaryInput');
const predictBtn = document.getElementById('predictBtn');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent = document.getElementById('resultContent');
const resultBadge = document.getElementById('resultBadge');
const confidenceTag = document.getElementById('confidenceTag');
const gaugeArc = document.getElementById('gaugeArc');
const gaugeText = document.getElementById('gaugeText');
const detailAge = document.getElementById('detailAge');
const detailSalary = document.getElementById('detailSalary');
const detailRec = document.getElementById('detailRec');
const leaderboardBody = document.getElementById('leaderboardBody');
const retrainBtn = document.getElementById('retrainBtn');
const toast = document.getElementById('toast');

// ─── Constants ───
const GAUGE_MAX_DASH = 251.2; // Arc length of the semicircle

// ─── Age Slider Sync ───
ageSlider.addEventListener('input', () => {
    ageDisplay.textContent = ageSlider.value;
});

// ─── Salary Formatting ───
salaryInput.addEventListener('input', (e) => {
    let raw = e.target.value.replace(/[^0-9]/g, '');
    if (raw) {
        e.target.value = Number(raw).toLocaleString('en-IN');
    }
});

// ─── Salary Presets ───
document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const val = Number(btn.dataset.salary);
        salaryInput.value = val.toLocaleString('en-IN');
    });
});

// ─── Predict Form Submit ───
predictForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const age = parseInt(ageSlider.value, 10);
    const salaryRaw = salaryInput.value.replace(/[^0-9]/g, '');

    if (!salaryRaw) {
        showToast('Please enter a salary amount', 'error');
        return;
    }

    const salary = parseFloat(salaryRaw);
    if (salary < 0 || salary > 5000000) {
        showToast('Salary must be between \u20B90 and \u20B950,00,000', 'error');
        return;
    }

    // Show loading state
    predictBtn.disabled = true;
    predictBtn.querySelector('.btn-text').style.display = 'none';
    predictBtn.querySelector('.btn-loader').style.display = 'inline-flex';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ age, salary }),
        });

        const data = await res.json();

        if (!res.ok) {
            showToast(data.error || 'Prediction failed', 'error');
            return;
        }

        displayResult(data);
    } catch (err) {
        showToast('Network error. Is the server running?', 'error');
    } finally {
        predictBtn.disabled = false;
        predictBtn.querySelector('.btn-text').style.display = 'inline';
        predictBtn.querySelector('.btn-loader').style.display = 'none';
    }
});

// ─── Display Result ───
function displayResult(data) {
    resultPlaceholder.style.display = 'none';
    resultContent.style.display = 'block';

    // Badge
    const isYes = data.label === 'Yes';
    resultBadge.textContent = isYes ? 'WILL PURCHASE' : 'UNLIKELY';
    resultBadge.className = 'result-badge ' + (isYes ? 'yes' : 'no');

    // Confidence tag
    confidenceTag.textContent = data.confidence_level + ' Confidence';

    // Animate gauge
    animateGauge(data.probability);

    // Details
    detailAge.textContent = data.age + ' years';
    detailSalary.textContent = '\u20B9' + data.salary.toLocaleString('en-IN');

    // Recommendation
    if (data.probability > 0.75) {
        detailRec.textContent = 'High priority lead - immediate outreach';
        detailRec.style.color = '#22c55e';
    } else if (data.probability > 0.5) {
        detailRec.textContent = 'Moderate lead - nurture campaign';
        detailRec.style.color = '#f59e0b';
    } else if (data.probability > 0.3) {
        detailRec.textContent = 'Low priority - awareness stage';
        detailRec.style.color = '#fb923c';
    } else {
        detailRec.textContent = 'Not recommended for outreach';
        detailRec.style.color = '#ef4444';
    }

    showToast('Prediction complete!', 'success');
}

// ─── Gauge Animation ───
function animateGauge(probability) {
    const target = probability * GAUGE_MAX_DASH;
    const targetPct = Math.round(probability * 100);
    let current = 0;
    const duration = 800;
    const startTime = performance.now();

    function step(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);

        current = eased * target;
        const currentPct = Math.round(eased * targetPct);

        gaugeArc.setAttribute('stroke-dasharray', `${current} ${GAUGE_MAX_DASH}`);
        gaugeText.textContent = currentPct + '%';

        if (progress < 1) {
            requestAnimationFrame(step);
        }
    }

    // Reset first
    gaugeArc.setAttribute('stroke-dasharray', `0 ${GAUGE_MAX_DASH}`);
    gaugeText.textContent = '0%';

    requestAnimationFrame(step);
}

// ─── Leaderboard ───
async function loadLeaderboard() {
    try {
        // Try training to get leaderboard data (since we don't have a separate endpoint for it)
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();

        if (res.ok && data.leaderboard) {
            renderLeaderboard(data.leaderboard);
        } else {
            leaderboardBody.innerHTML = `
                <tr><td colspan="8" class="loading-cell">
                    Click "Retrain Model" to generate the leaderboard
                </td></tr>`;
        }
    } catch {
        leaderboardBody.innerHTML = `
            <tr><td colspan="8" class="loading-cell">
                Click "Retrain Model" to generate the leaderboard
            </td></tr>`;
    }
}

function renderLeaderboard(data) {
    leaderboardBody.innerHTML = '';

    data.forEach((row, i) => {
        const tr = document.createElement('tr');
        const isBest = i === 0;
        if (isBest) tr.classList.add('best-row');

        tr.innerHTML = `
            <td>${isBest ? '<span class="best-badge">&#9733; #1</span>' : '#' + (i + 1)}</td>
            <td style="font-weight:600">${row.Model}</td>
            <td>${(row.TestAcc * 100).toFixed(1)}%</td>
            <td>${(row.CVAcc * 100).toFixed(1)}%</td>
            <td>${(row.Precision * 100).toFixed(1)}%</td>
            <td>${(row.Recall * 100).toFixed(1)}%</td>
            <td>${(row.F1 * 100).toFixed(1)}%</td>
            <td>${row.TrainTime}</td>
        `;
        leaderboardBody.appendChild(tr);
    });
}

// ─── Retrain ───
retrainBtn.addEventListener('click', async () => {
    retrainBtn.disabled = true;
    retrainBtn.innerHTML = '<span class="spinner"></span> Training...';

    try {
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();

        if (res.ok) {
            renderLeaderboard(data.leaderboard);
            showToast('Model retrained successfully!', 'success');
        } else {
            showToast(data.error || 'Training failed', 'error');
        }
    } catch {
        showToast('Network error during training', 'error');
    } finally {
        retrainBtn.disabled = false;
        retrainBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 2v6h-6M3 12A9 9 0 0 1 18.36 5.64L21 8M3 22v-6h6M21 12A9 9 0 0 0 5.64 18.36L3 16"/>
            </svg>
            Retrain Model`;
    }
});

// ─── Toast ───
let toastTimeout;
function showToast(message, type = 'success') {
    clearTimeout(toastTimeout);
    toast.querySelector('.toast-icon').textContent = type === 'success' ? '✓' : '✕';
    toast.querySelector('.toast-message').textContent = message;
    toast.className = 'toast ' + type + ' show';
    toastTimeout = setTimeout(() => toast.classList.remove('show'), 3500);
}

// ─── Smooth scroll for nav links ───
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) target.scrollIntoView({ behavior: 'smooth' });
    });
});

// ─── Init ───
document.addEventListener('DOMContentLoaded', () => {
    loadLeaderboard();
});
