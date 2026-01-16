/**
 * Neuro-Swarm Dashboard Frontend
 *
 * Real-time monitoring for distributed evolution.
 */

// State
let fitnessChart = null;
let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const reconnectDelay = 2000;

// DOM Elements
const elements = {
    statusBadge: document.getElementById('status-badge'),
    generation: document.getElementById('generation'),
    evaluations: document.getElementById('evaluations'),
    elapsed: document.getElementById('elapsed'),
    algorithm: document.getElementById('algorithm'),
    archiveCoverage: document.getElementById('archive-coverage'),
    archiveCells: document.getElementById('archive-cells'),
    workersTbody: document.getElementById('workers-tbody'),
    taskQueueBar: document.getElementById('task-queue-bar'),
    taskQueueValue: document.getElementById('task-queue-value'),
    taskQueueAlert: document.getElementById('task-queue-alert'),
    resultQueueBar: document.getElementById('result-queue-bar'),
    resultQueueValue: document.getElementById('result-queue-value'),
    throughputValue: document.getElementById('throughput-value'),
    bestFitness: document.getElementById('best-fitness'),
    bestGenome: document.getElementById('best-genome'),
    wsStatus: document.getElementById('ws-status'),
    wsStatusText: document.getElementById('ws-status-text')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initFitnessChart();
    initHeatmap();
    connectWebSocket();
    fetchInitialData();

    // Periodic refresh of data that doesn't come via WebSocket
    setInterval(fetchWorkers, 5000);
    setInterval(fetchHistory, 10000);
    setInterval(fetchArchive, 10000);
    setInterval(fetchBest, 5000);
});

// Fitness Chart
function initFitnessChart() {
    const ctx = document.getElementById('fitness-chart').getContext('2d');

    fitnessChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Best',
                    data: [],
                    borderColor: '#3fb950',
                    backgroundColor: 'rgba(63, 185, 80, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: 'Mean',
                    data: [],
                    borderColor: '#58a6ff',
                    borderWidth: 2,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: 'Min',
                    data: [],
                    borderColor: '#d29922',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    tension: 0.1,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#21262d',
                    titleColor: '#c9d1d9',
                    bodyColor: '#8b949e',
                    borderColor: '#30363d',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: '#30363d',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8b949e',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: '#30363d',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8b949e'
                    }
                }
            }
        }
    });
}

function updateFitnessChart(history) {
    if (!fitnessChart || !history.length) return;

    const labels = history.map(h => h.generation);
    const best = history.map(h => h.best_fitness || 0);
    const mean = history.map(h => h.mean_fitness || 0);
    const min = history.map(h => h.min_fitness || 0);

    fitnessChart.data.labels = labels;
    fitnessChart.data.datasets[0].data = best;
    fitnessChart.data.datasets[1].data = mean;
    fitnessChart.data.datasets[2].data = min;
    fitnessChart.update('none');
}

// MAP-Elites Heatmap
let heatmapCanvas = null;
let heatmapCtx = null;

function initHeatmap() {
    heatmapCanvas = document.getElementById('archive-heatmap');
    heatmapCtx = heatmapCanvas.getContext('2d');

    // Clear to default state
    heatmapCtx.fillStyle = '#21262d';
    heatmapCtx.fillRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);

    // Tooltip handling
    const tooltip = document.getElementById('heatmap-tooltip');

    heatmapCanvas.addEventListener('mousemove', (e) => {
        const rect = heatmapCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Calculate cell coordinates
        const cellWidth = heatmapCanvas.width / 20;
        const cellHeight = heatmapCanvas.height / 20;
        const cellX = Math.floor(x / cellWidth);
        const cellY = Math.floor(y / cellHeight);

        // Get cell data if available
        if (window.archiveData && window.archiveData.grid) {
            const cell = window.archiveData.grid.find(c =>
                c.coords && c.coords[0] === cellX && c.coords[1] === cellY
            );

            if (cell) {
                tooltip.innerHTML = `
                    <div>Cell: (${cellX}, ${cellY})</div>
                    <div>Fitness: ${cell.fitness.toFixed(4)}</div>
                `;
                tooltip.style.left = `${e.clientX - rect.left + 10}px`;
                tooltip.style.top = `${e.clientY - rect.top + 10}px`;
                tooltip.classList.add('visible');
            } else {
                tooltip.classList.remove('visible');
            }
        }
    });

    heatmapCanvas.addEventListener('mouseleave', () => {
        tooltip.classList.remove('visible');
    });
}

function updateHeatmap(archiveData) {
    if (!heatmapCtx || !archiveData) return;

    window.archiveData = archiveData;

    const width = heatmapCanvas.width;
    const height = heatmapCanvas.height;
    const gridRes = archiveData.grid_resolution || [20, 20];
    const cellWidth = width / gridRes[0];
    const cellHeight = height / gridRes[1];

    // Clear canvas
    heatmapCtx.fillStyle = '#21262d';
    heatmapCtx.fillRect(0, 0, width, height);

    // Draw grid lines
    heatmapCtx.strokeStyle = '#30363d';
    heatmapCtx.lineWidth = 0.5;

    for (let i = 0; i <= gridRes[0]; i++) {
        heatmapCtx.beginPath();
        heatmapCtx.moveTo(i * cellWidth, 0);
        heatmapCtx.lineTo(i * cellWidth, height);
        heatmapCtx.stroke();
    }

    for (let j = 0; j <= gridRes[1]; j++) {
        heatmapCtx.beginPath();
        heatmapCtx.moveTo(0, j * cellHeight);
        heatmapCtx.lineTo(width, j * cellHeight);
        heatmapCtx.stroke();
    }

    // Draw cells
    if (archiveData.grid && archiveData.grid.length > 0) {
        // Find fitness range
        const fitnesses = archiveData.grid.map(c => c.fitness);
        const minFit = Math.min(...fitnesses);
        const maxFit = Math.max(...fitnesses);
        const range = maxFit - minFit || 1;

        archiveData.grid.forEach(cell => {
            if (cell.coords) {
                const [x, y] = cell.coords;
                const normalized = (cell.fitness - minFit) / range;
                const color = fitnessToColor(normalized);

                heatmapCtx.fillStyle = color;
                heatmapCtx.fillRect(
                    x * cellWidth + 1,
                    y * cellHeight + 1,
                    cellWidth - 2,
                    cellHeight - 2
                );
            }
        });
    }

    // Update stats
    const totalCells = gridRes[0] * gridRes[1];
    const filledCells = archiveData.grid ? archiveData.grid.length : 0;
    const coverage = (filledCells / totalCells) * 100;

    elements.archiveCoverage.textContent = `${coverage.toFixed(1)}%`;
    elements.archiveCells.textContent = `${filledCells} / ${totalCells}`;
}

function fitnessToColor(normalized) {
    // Color gradient: dark purple -> blue -> green -> yellow
    const r = Math.round(normalized < 0.5 ? 30 + normalized * 100 : 30 + (1 - normalized) * 100 + 155);
    const g = Math.round(normalized * 200 + 50);
    const b = Math.round(normalized < 0.5 ? 100 + normalized * 150 : 250 - normalized * 200);
    return `rgb(${r}, ${g}, ${b})`;
}

// WebSocket Connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/updates`;

    setConnectionStatus('connecting');

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        setConnectionStatus('connected');
        reconnectAttempts = 0;
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    };

    ws.onclose = () => {
        setConnectionStatus('disconnected');
        scheduleReconnect();
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
    };
}

function scheduleReconnect() {
    if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        setTimeout(connectWebSocket, reconnectDelay);
    }
}

function setConnectionStatus(status) {
    elements.wsStatus.className = `ws-indicator ${status}`;
    elements.wsStatusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

function handleWebSocketMessage(data) {
    if (data.type === 'update') {
        updateStatus(data.status);
        updateQueueStats(data.queue);
    }
}

// Update Functions
function updateStatus(status) {
    if (!status) return;

    // Status badge
    const isRunning = status.running;
    elements.statusBadge.className = `status-badge ${isRunning ? 'running' : 'stopped'}`;
    elements.statusBadge.textContent = isRunning ? 'Running' : 'Stopped';

    // Stats
    elements.generation.textContent = status.generation || 0;
    elements.evaluations.textContent = formatNumber(status.total_evaluations || 0);
    elements.elapsed.textContent = formatDuration(status.elapsed_time || 0);
    elements.algorithm.textContent = (status.algorithm || '-').toUpperCase();
}

function updateQueueStats(queue) {
    if (!queue) return;

    const taskLength = queue.task_queue_length || 0;
    const resultLength = queue.result_queue_length || 0;
    const throughput = queue.throughput || 0;
    const maxQueue = 500; // For progress bar scaling

    // Task queue
    const taskPercent = Math.min(100, (taskLength / maxQueue) * 100);
    elements.taskQueueBar.style.width = `${taskPercent}%`;
    elements.taskQueueValue.textContent = taskLength;

    // Color coding
    elements.taskQueueBar.className = 'progress-fill';
    if (taskLength > 300) {
        elements.taskQueueBar.classList.add('critical');
        elements.taskQueueAlert.classList.remove('hidden');
    } else if (taskLength > 100) {
        elements.taskQueueBar.classList.add('warning');
        elements.taskQueueAlert.classList.add('hidden');
    } else {
        elements.taskQueueAlert.classList.add('hidden');
    }

    // Result queue
    const resultPercent = Math.min(100, (resultLength / maxQueue) * 100);
    elements.resultQueueBar.style.width = `${resultPercent}%`;
    elements.resultQueueValue.textContent = resultLength;

    // Throughput
    elements.throughputValue.textContent = throughput.toFixed(1);
}

function updateWorkers(workers) {
    if (!workers || !workers.length) {
        elements.workersTbody.innerHTML = `
            <tr class="empty-row">
                <td colspan="5">No workers connected</td>
            </tr>
        `;
        return;
    }

    elements.workersTbody.innerHTML = workers.map(w => {
        const secondsSince = w.seconds_since_heartbeat || 999;
        let statusClass = 'active';
        let statusText = 'Active';

        if (secondsSince > 60) {
            statusClass = 'offline';
            statusText = 'Offline';
        } else if (secondsSince > 30) {
            statusClass = 'stale';
            statusText = 'Stale';
        }

        return `
            <tr>
                <td><code>${w.worker_id || 'unknown'}</code></td>
                <td><span class="worker-status ${statusClass}">${statusText}</span></td>
                <td>${formatNumber(w.tasks_completed || 0)}</td>
                <td>${w.tasks_failed || 0}</td>
                <td>${formatTimeSince(secondsSince)}</td>
            </tr>
        `;
    }).join('');
}

function updateBest(best) {
    if (!best) return;

    if (best.fitness !== null && best.fitness !== undefined) {
        elements.bestFitness.textContent = best.fitness.toFixed(4);
    } else {
        elements.bestFitness.textContent = '-';
    }

    if (best.genome) {
        elements.bestGenome.textContent = JSON.stringify(best.genome, null, 2);
    } else {
        elements.bestGenome.textContent = 'No solution yet';
    }
}

// API Fetches
async function fetchInitialData() {
    await Promise.all([
        fetchStatus(),
        fetchHistory(),
        fetchArchive(),
        fetchWorkers(),
        fetchBest()
    ]);
}

async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        updateStatus(status);
    } catch (e) {
        console.error('Failed to fetch status:', e);
    }
}

async function fetchHistory() {
    try {
        const response = await fetch('/api/history?limit=500');
        const history = await response.json();
        updateFitnessChart(history);
    } catch (e) {
        console.error('Failed to fetch history:', e);
    }
}

async function fetchArchive() {
    try {
        const response = await fetch('/api/archive');
        const archive = await response.json();
        updateHeatmap(archive);
    } catch (e) {
        console.error('Failed to fetch archive:', e);
    }
}

async function fetchWorkers() {
    try {
        const response = await fetch('/api/workers');
        const workers = await response.json();
        updateWorkers(workers);
    } catch (e) {
        console.error('Failed to fetch workers:', e);
    }
}

async function fetchBest() {
    try {
        const response = await fetch('/api/best');
        const best = await response.json();
        updateBest(best);
    } catch (e) {
        console.error('Failed to fetch best:', e);
    }
}

// Utility Functions
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function formatTimeSince(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)}s ago`;
    } else if (seconds < 3600) {
        return `${Math.floor(seconds / 60)}m ago`;
    } else {
        return `${Math.floor(seconds / 3600)}h ago`;
    }
}
