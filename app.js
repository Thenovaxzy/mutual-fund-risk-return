const API_URL = 'http://localhost:8000/api/analyze';

// Set Chart.js Defaults for Dark Theme
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = 'rgba(255, 255, 255, 0.7)';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
}

// UI Elements
const form = document.getElementById('analysis-form');
const analyzeBtn = document.getElementById('analyze-btn');
const btnText = document.querySelector('.btn-text');
const spinner = document.querySelector('.spinner');
const yearsSlider = document.getElementById('years');
const yearsVal = document.getElementById('years-val');
const resultsContainer = document.getElementById('results-container');
const emptyState = document.getElementById('empty-state');

// Custom Prediction Elements
const customBetaSlider = document.getElementById('custom-beta');
const customBetaVal = document.getElementById('custom-beta-val');
const predictCustomBtn = document.getElementById('predict-custom-btn');
const predictionResultArea = document.getElementById('prediction-result-area');
const customPredictionText = document.getElementById('custom-prediction-text');

// Variables
let currentSml = null;
let smlChartInstance = null;
let customChartInstance = null;

// Event Listeners
yearsSlider.addEventListener('input', (e) => {
    yearsVal.textContent = e.target.value;
});

if (customBetaSlider && customBetaVal) {
    customBetaSlider.addEventListener('input', (e) => {
        customBetaVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
}

if (predictCustomBtn) {
    predictCustomBtn.addEventListener('click', () => {
        if (!currentSml) {
            alert("Please run the main analysis first to establish the prediction model.");
            return;
        }

        const beta = parseFloat(customBetaSlider.value);
        const predictedReturn = (beta * currentSml.slope) + currentSml.intercept;

        renderCustomPredictionChart(beta, predictedReturn);
    });
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    await fetchAnalysis();
});

async function fetchAnalysis() {
    // 1. Get input values
    const payload = {
        tickers: document.getElementById('tickers').value,
        benchmark: document.getElementById('benchmark').value,
        years: parseInt(document.getElementById('years').value),
        risk_free_rate: parseFloat(document.getElementById('risk_free_rate').value) / 100
    };

    // 2. Set loading state
    setLoadingState(true);

    try {
        // 3. Call API
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to fetch data');
        }

        const data = await response.json();

        // 4. Update UI
        updateUI(data);

    } catch (error) {
        console.error('API Error:', error);
        alert(`Error: ${error.message}`);
    } finally {
        setLoadingState(false);
    }
}

function setLoadingState(isLoading) {
    analyzeBtn.disabled = isLoading;
    if (isLoading) {
        btnText.textContent = 'Analyzing...';
        spinner.classList.remove('hidden');
    } else {
        btnText.textContent = 'Run Analysis';
        spinner.classList.add('hidden');
    }
}

function updateUI(data) {
    currentSml = data.sml;

    // Hide empty state, show results
    emptyState.classList.add('hidden');
    resultsContainer.classList.remove('hidden');

    // Render Chart
    renderChart(data.metrics, data.trendline);

    // Update Stats
    updateStats(data.sml);

    // Populate Table
    populateTable(data.metrics, data.sml);
}

function renderChart(metrics, trendline) {
    const ctx = document.getElementById('smlChart').getContext('2d');

    // Parse data for Chart.js
    const scatterData = metrics.map(m => ({
        x: m.Beta,
        y: m['Return (Annualized)'],
        label: m.Fund,
        sharpe: m['Sharpe Ratio']
    }));

    if (smlChartInstance) {
        smlChartInstance.destroy();
    }

    smlChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Funds',
                    data: scatterData,
                    backgroundColor: '#38bdf8',
                    pointRadius: 6,
                    pointHoverRadius: 8,
                },
                {
                    label: 'Linear Regression Prediction Line',
                    data: trendline,
                    type: 'line',
                    borderColor: '#10b981',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const point = context.raw;
                            if (context.datasetIndex === 1) return `Predicted Return (Regression): ${(point.y * 100).toFixed(2)}%`;
                            return `${point.label} | Beta: ${point.x.toFixed(2)} | Actual Return: ${(point.y * 100).toFixed(2)}%`;
                        }
                    }
                },
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Systematic Risk (Beta)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Annualized Return'
                    },
                    ticks: {
                        callback: function (value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

function updateStats(sml) {
    const slope = sml.slope * 100;
    const intercept = sml.intercept * 100;

    document.getElementById('stat-equation').textContent = `E(R) = (${sml.slope.toFixed(4)} * Beta) + ${sml.intercept.toFixed(4)}`;
    document.getElementById('stat-slope').textContent = `${slope.toFixed(2)}%`;
    document.getElementById('stat-intercept').textContent = `${intercept.toFixed(2)}%`;
    document.getElementById('stat-rsquared').textContent = sml.r_squared.toFixed(4);
}

function populateTable(metrics, sml) {
    const tbody = document.querySelector('#metrics-table tbody');
    tbody.innerHTML = '';

    metrics.forEach(m => {
        const tr = document.createElement('tr');

        // Formatting helper
        const formatPct = (val) => val != null ? (val * 100).toFixed(2) + '%' : 'N/A';
        const formatDec = (val) => val != null ? val.toFixed(4) : 'N/A';
        const formatAlpha = (val) => {
            if (val == null) return 'N/A';
            const str = (val * 100).toFixed(2) + '%';
            return val > 0 ? `<span class="positive-val">+${str}</span>` : (val < 0 ? `<span class="negative-val">${str}</span>` : str);
        };

        // Calculate explicit predicted return
        const predictedReturn = (m.Beta * sml.slope) + sml.intercept;

        tr.innerHTML = `
            <td><strong>${m.Fund}</strong></td>
            <td>${formatPct(m['Return (Annualized)'])}</td>
            <td>${formatPct(m['Volatility (Annualized)'])}</td>
            <td>${formatDec(m.Beta)}</td>
            <td><strong>${formatPct(predictedReturn)}</strong></td>
            <td>${formatAlpha(m['Alpha (Annualized)'])}</td>
            <td>${formatDec(m['Sharpe Ratio'])}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderCustomPredictionChart(beta, predictedReturn) {
    // Show the result area
    predictionResultArea.classList.remove('hidden');

    const ctx = document.getElementById('customPredictionChart').getContext('2d');
    const returnPct = (predictedReturn * 100).toFixed(2);
    const isProfit = predictedReturn >= 0;

    if (customChartInstance) {
        customChartInstance.destroy();
    }

    // Set professional text message
    if (isProfit) {
        customPredictionText.textContent = `Predicted Profit: +${returnPct}%`;
        customPredictionText.style.color = '#38bdf8'; // Sky Blue
    } else {
        customPredictionText.textContent = `Predicted Risk/Loss: ${returnPct}%`;
        customPredictionText.style.color = '#ef4444'; // Red
    }

    const barColor = isProfit ? 'rgba(56, 189, 248, 0.8)' : 'rgba(239, 68, 68, 0.8)';
    const borderColor = isProfit ? 'rgba(56, 189, 248, 1)' : 'rgba(239, 68, 68, 1)';

    customChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [`Beta Risk: ${beta.toFixed(2)}`],
            datasets: [{
                label: 'Expected Return (%)',
                data: [parseFloat(returnPct)],
                backgroundColor: barColor,
                borderColor: borderColor,
                borderWidth: 1,
                borderRadius: 8,
                barPercentage: 0.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y', // Horizontal bar
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Expected Return: ${context.raw}%`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Expected Return (%)',
                        font: { weight: 'bold' }
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.05)',
                        drawBorder: false
                    }
                },
                y: {
                    grid: {
                        display: false,
                        drawBorder: false
                    },
                    ticks: {
                        font: { weight: 'bold' }
                    }
                }
            }
        }
    });
}
