document.addEventListener('DOMContentLoaded', () => {

    // Global Stats
    let internalTotalTxn = 2143598;
    let internalFraudPrevented = 4092;

    // 1. Check AI Engine Status
    checkBackendStatus();

    // 2. Initialize Line Chart
    const lineCtx = document.getElementById('trafficCurveChart').getContext('2d');
    const primaryGradient = lineCtx.createLinearGradient(0, 0, 0, 300);
    primaryGradient.addColorStop(0, 'rgba(37, 99, 235, 0.25)'); // Primary Blue
    primaryGradient.addColorStop(1, 'rgba(37, 99, 235, 0)');

    const trafficChart = new Chart(lineCtx, {
        type: 'line',
        data: {
            labels: ['M', 'T', 'W', 'T', 'F', 'S', 'S', 'LIVE'],
            datasets: [{
                label: 'Processed Items',
                data: [120, 190, 150, 210, 240, 180, 290, 320],
                borderColor: '#2563EB',
                backgroundColor: primaryGradient,
                fill: true,
                tension: 0.4,
                borderWidth: 3,
                pointRadius: 4,
                pointBackgroundColor: '#FFF',
                pointBorderColor: '#2563EB',
                pointBorderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }, // Hide legend for cleaner look
            scales: {
                y: { beginAtZero: true, grid: { color: '#F1F5F9', borderDash: [5, 5] }, border: { display: false } },
                x: { grid: { display: false }, border: { display: false } }
            },
            interaction: { mode: 'index', intersect: false }
        }
    });

    // 3. Automated Live Traffic Injection
    setInterval(() => {
        // Organic random generation
        if (Math.random() > 0.4) {
            internalTotalTxn++;
            document.getElementById('totalTxn').textContent = internalTotalTxn.toLocaleString();

            const isHighRisk = Math.random() > 0.90; // 10% chance to simulate threat
            if (isHighRisk) {
                internalFraudPrevented++;
                document.getElementById('fraudTxn').textContent = internalFraudPrevented.toLocaleString();

                const amount = (Math.random() * 40000 + 5000).toFixed(2);
                const prob = (Math.random() * 0.25 + 0.70).toFixed(2); // 70-95%
                const confPct = Math.floor(prob * 100);
                
                injectTableThreat(`₹${amount}`, "Dist_Anomaly, High_Ratio", confPct, true);

                // Peak the chart
                const currentData = trafficChart.data.datasets[0].data;
                currentData[currentData.length - 1] = parseInt(currentData[currentData.length - 1]) + Math.floor(Math.random() * 50);
                trafficChart.update();
            }
        }
    }, 3500);

    // Initial table populates
    injectTableThreat("₹45,890.50", "Odd_Hour, No_PIN", 94, true);
    injectTableThreat("₹850.00", "None", 12, false, "RX-992");
    injectTableThreat("₹81,204.00", "Dist_Home_Max", 82, true);

    // 4. Form Submission & API Connection
    const predictForm = document.getElementById('aiPredictorForm');
    
    predictForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading State
        const submitBtn = document.getElementById('runAnalysisBtn');
        const btnText = submitBtn.querySelector('.btn-text');
        const spinner = document.getElementById('analyzeSpinner');
        
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        submitBtn.disabled = true;

        // Gather advanced inputs
        const payload = {
            amount: parseFloat(document.getElementById('in_amount').value),
            time_hour: parseInt(document.getElementById('in_time_hour').value),
            dist_from_home: parseFloat(document.getElementById('in_dist_home').value),
            dist_from_last_tx: parseFloat(document.getElementById('in_dist_last').value),
            ratio_to_median: parseFloat(document.getElementById('in_ratio').value),
            online_order: document.getElementById('in_online').checked ? 1 : 0,
            used_chip: document.getElementById('in_chip').checked ? 1 : 0,
            used_pin: document.getElementById('in_pin').checked ? 1 : 0,
        };

        try {
            const resp = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await resp.json();

            if (resp.ok) {
                renderPredictionResults(data, payload);
            } else {
                showToast("System Error", data.error || "Execution halted.", "danger");
            }

        } catch (error) {
            console.error(error);
            showToast("Network Failure", "Cannot reach the AI endpoint.", "danger");
        } finally {
            btnText.style.display = 'block';
            spinner.style.display = 'none';
            submitBtn.disabled = false;
        }
    });

});

// --- UI Helper Functions ---

async function checkBackendStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const statLabel = document.getElementById('modelStatus');
        const statDot = document.querySelector('.status-indicator');

        if (data.status === 'online') {
            statLabel.textContent = `AI Engine Online (${data.model})`;
            statDot.className = 'status-indicator active';
            statDot.style.backgroundColor = 'var(--success)';
        } else {
            statLabel.textContent = "AI Engine Offline";
            statDot.className = 'status-indicator';
            statDot.style.backgroundColor = 'var(--danger)';
        }
    } catch {
        document.getElementById('modelStatus').textContent = "API Unreachable";
        document.querySelector('.status-indicator').style.backgroundColor = 'var(--warning)';
        document.querySelector('.status-indicator').classList.remove('active');
    }
}

function injectTableThreat(amount, factors, confidence, isThreat, prefix="TX") {
    const tbody = document.querySelector('#liveThreatsTable tbody');
    if (!tbody) return;

    const tr = document.createElement('tr');
    tr.className = 'slide-in';
    const txId = `${prefix}-${Math.floor(10000 + Math.random() * 90000)}`;
    
    const badgeHtml = isThreat 
        ? `<span class="badge badge-high">High Risk</span>` 
        : (confidence > 40 ? `<span class="badge badge-medium">Review</span>` : `<span class="badge badge-low">Safe</span>`);
    
    const actionHtml = isThreat
        ? `<button class="btn btn-outline btn-sm" style="color:var(--danger); border-color:var(--danger-light)">Block</button>`
        : `<button class="btn btn-outline btn-sm">Allow</button>`;

    tr.innerHTML = `
        <td><strong>${txId}</strong></td>
        <td>${amount}</td>
        <td style="color: var(--text-light); font-size: 0.8rem;">${factors}</td>
        <td><strong>${confidence}%</strong> ${badgeHtml}</td>
        <td>${actionHtml}</td>
    `;

    tbody.insertBefore(tr, tbody.firstChild);
    if(tbody.children.length > 5) tbody.removeChild(tbody.lastChild);
}

function renderPredictionResults(res, input) {
    const box = document.getElementById('analysisResult');
    box.classList.remove('hidden');

    const probPct = Math.round(res.probability * 100);
    const circlePath = document.getElementById('riskCirclePath');
    const pctText = document.getElementById('riskPercentage');
    
    // Animate SVG Circle
    pctText.textContent = `${probPct}%`;
    circlePath.setAttribute('stroke-dasharray', `${probPct}, 100`);
    
    // Reset classes
    circlePath.classList.remove('low', 'med', 'high');
    const badge = document.getElementById('verdictBadge');
    badge.className = 'badge';

    const title = document.getElementById('verdictTitle');

    if (res.risk_level === 'High') {
        circlePath.classList.add('high');
        badge.classList.add('badge-high');
        badge.textContent = "CRITICAL RISK";
        title.textContent = "Fraudulent Pattern Detected";
        title.style.color = "var(--danger)";
        
        showToast("Threat Identified", "Machine learning flagged this payload.", "danger");
        injectTableThreat(`₹${input.amount.toFixed(2)}`, "Manual Submission", probPct, true, "MANL");
    } else if (res.risk_level === 'Medium') {
        circlePath.classList.add('med');
        badge.classList.add('badge-medium');
        badge.textContent = "ELEVATED RISK";
        title.textContent = "Requires Manual Review";
        title.style.color = "var(--warning)";
        injectTableThreat(`₹${input.amount.toFixed(2)}`, "Manual Submission", probPct, false, "MANL");
    } else {
        circlePath.classList.add('low');
        badge.classList.add('badge-low');
        badge.textContent = "SAFE TRANSACTION";
        title.textContent = "Transaction Verified Safe";
        title.style.color = "var(--success)";
        showToast("Scan Complete", "No anomalies detected.", "success");
    }

    // Explanations
    const list = document.getElementById('explainList');
    list.innerHTML = "";
    res.explanations.forEach(exp => {
        const li = document.createElement('li');
        li.textContent = exp;
        list.appendChild(li);
    });

}

function showToast(title, body, type="info") {
    const zone = document.getElementById('toastZone');
    
    const div = document.createElement('div');
    div.className = `toast-msg ${type}`;
    
    let icon = "info";
    if (type === "danger") icon = "alert-octagon";
    else if (type === "success") icon = "check-circle-2";

    div.innerHTML = `
        <div class="toast-icon"><i data-lucide="${icon}"></i></div>
        <div>
            <h5>${title}</h5>
            <p>${body}</p>
        </div>
    `;

    zone.appendChild(div);
    lucide.createIcons();

    setTimeout(() => {
        div.style.opacity = "0";
        setTimeout(() => div.remove(), 300);
    }, 4000);
}
