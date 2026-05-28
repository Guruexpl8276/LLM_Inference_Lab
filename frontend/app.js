/* ============================================================
   LLM GPU Inference Lab — Dashboard JavaScript
   Handles: SSE chat streaming, benchmark polling, GPU live metrics,
            Chart.js visualizations, model management.
   ============================================================ */

const API = 'http://localhost:8000';

// ── State ─────────────────────────────────────────────────
let isGenerating = false;
let tpsHistory   = [];
let gpuHistory   = [];
let tpsChartInst = null;
let gpuChartInst = null;
let benchChartInst  = null;
let compareChartInst = null;
let benchPollTimer = null;

// ── MODEL INFO MAP ────────────────────────────────────────
const MODEL_INFO = {
  'llama3.2:3b'        : '~2.0 GB VRAM · Q4_K_M · 3B params · Meta',
  'llama3.2:1b'        : '~1.3 GB VRAM · Q4_K_M · 1B params · Meta (fast)',
  'phi3:mini'          : '~2.3 GB VRAM · Q4_K_M · 3.8B params · Microsoft',
  'phi3.5:mini'        : '~2.3 GB VRAM · Q4_K_M · 3.8B params · Microsoft',
  'qwen2.5:3b'         : '~2.0 GB VRAM · Q4_K_M · 3B params · Alibaba (multilingual)',
  'qwen2.5:1.5b'       : '~1.0 GB VRAM · Q4_K_M · 1.5B params · Alibaba',
  'qwen2.5-coder:1.5b' : '~1.0 GB VRAM · Q4_K_M · 1.5B params · Code-focused',
  'gemma2:2b'          : '~1.6 GB VRAM · Q4_K_M · 2B params · Google',
  'gemma3:1b'          : '~1.0 GB VRAM · Q4_K_M · 1B params · Google',
  'tinyllama'          : '~0.7 GB VRAM · Q4_K_M · 1.1B params · Speed baseline',
  'smollm2:1.7b'       : '~1.2 GB VRAM · Q4_K_M · 1.7B params · HF',
  'deepseek-r1:1.5b'   : '~1.1 GB VRAM · Q4_K_M · 1.5B reasoning (<think> tokens)',
};

// ═══════════════════════════════════════════════════════════
// TAB NAVIGATION
// ═══════════════════════════════════════════════════════════
function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
  document.getElementById(`nav-${name}`).classList.add('active');
}

// ═══════════════════════════════════════════════════════════
// MODEL SELECTOR
// ═══════════════════════════════════════════════════════════
function onModelChange() {
  const model = document.getElementById('model-select').value;
  const info  = MODEL_INFO[model] || '';
  document.getElementById('model-info').textContent = info;
}

// ═══════════════════════════════════════════════════════════
// CHAT — SEND + SSE STREAMING
// ═══════════════════════════════════════════════════════════
function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

async function sendMessage() {
  if (isGenerating) return;

  const input = document.getElementById('chat-input');
  const prompt = input.value.trim();
  if (!prompt) return;

  const model       = document.getElementById('model-select').value;
  const maxTokens   = parseInt(document.getElementById('max-tokens').value);
  const temperature = parseFloat(document.getElementById('temperature').value);

  // Clear placeholder
  document.getElementById('welcome-msg')?.remove();

  input.value = '';
  input.style.height = 'auto';
  setGenerating(true);

  // Add user bubble
  appendMessage('user', prompt);

  // Add assistant bubble with cursor
  const assistantMsgId = 'msg-' + Date.now();
  const bubble = appendMessage('assistant', '', assistantMsgId);
  bubble.innerHTML = '<span class="cursor-blink"></span>';

  let fullText = '';
  let firstToken = true;

  try {
    const response = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, model, max_tokens: maxTokens, temperature }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));

        if (data.error) {
          bubble.textContent = '❌ ' + data.error;
          break;
        }

        if (data.token) {
          fullText += data.token;
          bubble.textContent = fullText;
        }

        // First token
        if (firstToken && data.ttft_ms !== null && data.ttft_ms !== undefined) {
          firstToken = false;
          bubble.innerHTML = fullText; // remove cursor flash on first token
        }

        // Done
        if (data.done && data.stats) {
          const s = data.stats;
          updateInferenceStats(s);
          tpsHistory.push(s.tokens_per_second);
          if (tpsHistory.length > 20) tpsHistory.shift();
          updateTpsChart();

          // Append inline metrics
          const metricBar = document.createElement('div');
          metricBar.className = 'msg-metrics';
          metricBar.innerHTML = `
            TTFT: <span>${s.ttft_ms.toFixed(0)}ms</span>
            TPOT: <span>${s.tpot_ms.toFixed(1)}ms</span>
            TPS: <span>${s.tokens_per_second.toFixed(1)} tok/s</span>
            Tokens: <span>${s.output_tokens}</span>
          `;
          bubble.parentElement.appendChild(metricBar);
        }
      }
    }
  } catch (err) {
    bubble.textContent = '❌ ' + err.message;
  }

  setGenerating(false);
  scrollToBottom();
}

function appendMessage(role, text, id) {
  const msgs   = document.getElementById('messages');
  const div    = document.createElement('div');
  div.className = `message ${role}`;
  if (id) div.id = id;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'user' ? '👤' : '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;

  div.appendChild(avatar);
  div.appendChild(bubble);
  msgs.appendChild(div);
  scrollToBottom();
  return bubble;
}

function scrollToBottom() {
  const msgs = document.getElementById('messages');
  msgs.scrollTop = msgs.scrollHeight;
}

function setGenerating(val) {
  isGenerating = val;
  document.getElementById('send-btn').disabled = val;
  document.getElementById('chat-input').disabled = val;
}

function updateInferenceStats(s) {
  document.getElementById('stat-ttft').textContent   = s.ttft_ms.toFixed(0) + ' ms';
  document.getElementById('stat-tpot').textContent   = s.tpot_ms.toFixed(1) + ' ms';
  document.getElementById('stat-tps').textContent    = s.tokens_per_second.toFixed(1) + ' tok/s';
  document.getElementById('stat-tokens').textContent = s.output_tokens + ' tokens';
  document.getElementById('stat-time').textContent   = s.total_time_s.toFixed(1) + 's';
}

// ═══════════════════════════════════════════════════════════
// BENCHMARK
// ═══════════════════════════════════════════════════════════
async function startBenchmark() {
  const model      = document.getElementById('bench-model').value;
  const promptType = document.getElementById('bench-prompt').value;
  const runs       = document.getElementById('bench-runs').value;
  const btn        = document.getElementById('bench-start-btn');
  const log        = document.getElementById('bench-log');

  btn.disabled = true;
  btn.textContent = '⏳ Running...';
  log.innerHTML = '<span class="log-line">Starting benchmark...</span>\n';
  document.getElementById('bench-results').style.display = 'none';
  document.getElementById('bench-chart-wrap').style.display = 'none';

  try {
    const startResp = await fetch(`${API}/benchmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt_type: promptType, runs: parseInt(runs) }),
    });
    if (!startResp.ok) {
      const err = await startResp.json();
      log.innerHTML += `<span class="log-error">Error: ${err.detail}</span>\n`;
      resetBenchBtn();
      return;
    }

    // Poll for results
    clearInterval(benchPollTimer);
    benchPollTimer = setInterval(async () => {
      try {
        const status = await fetch(`${API}/benchmark/status`).then(r => r.json());

        // Update log
        log.innerHTML = status.progress.map(p => `<span class="log-line">✓ ${p}</span>`).join('\n');

        if (!status.running && status.result) {
          clearInterval(benchPollTimer);
          if (status.result.error) {
            log.innerHTML += `<span class="log-error">❌ ${status.result.error}</span>`;
          } else {
            renderBenchResults(status.result, model);
          }
          resetBenchBtn();
        }
      } catch (e) {
        clearInterval(benchPollTimer);
        resetBenchBtn();
      }
    }, 1000);

  } catch (err) {
    log.innerHTML += `<span class="log-error">❌ ${err.message}</span>`;
    resetBenchBtn();
  }
}

function resetBenchBtn() {
  const btn = document.getElementById('bench-start-btn');
  btn.disabled = false;
  btn.textContent = '▶ Run Benchmark';
}

function renderBenchResults(result, model) {
  const tbody = document.getElementById('bench-tbody');
  const best  = (val, threshold, lowGood) => {
    if (lowGood) return val < threshold ? '🟢 Good' : val < threshold * 2 ? '🟡 Moderate' : '🔴 Slow';
    return val > threshold ? '🟢 Good' : val > threshold / 2 ? '🟡 Moderate' : '🔴 Slow';
  };

  tbody.innerHTML = `
    <tr><td>Avg TTFT</td><td class="best">${result.avg_ttft_ms} ms</td><td>${best(result.avg_ttft_ms, 500, true)}</td></tr>
    <tr><td>Avg TPOT</td><td>${result.avg_tpot_ms} ms</td><td>${best(result.avg_tpot_ms, 50, true)}</td></tr>
    <tr><td>Avg Tok/s</td><td class="best">${result.avg_tps}</td><td>${best(result.avg_tps, 15, false)}</td></tr>
    <tr><td>Min Tok/s</td><td>${result.min_tps}</td><td>—</td></tr>
    <tr><td>Max Tok/s</td><td>${result.max_tps}</td><td>—</td></tr>
    <tr><td>Avg Tokens</td><td>${result.avg_output_tokens}</td><td>—</td></tr>
    <tr><td>Runs</td><td>${result.runs}</td><td>—</td></tr>
  `;
  document.getElementById('bench-results').style.display = 'block';

  // Draw chart
  const tpsVals = result.raw_results.map(r => r.tps);
  drawBenchChart(tpsVals);
}

function drawBenchChart(tpsVals) {
  const wrap = document.getElementById('bench-chart-wrap');
  wrap.style.display = 'block';

  const ctx = document.getElementById('bench-chart').getContext('2d');
  if (benchChartInst) benchChartInst.destroy();

  benchChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: tpsVals.map((_, i) => `Run ${i + 1}`),
      datasets: [{
        label: 'Tokens/sec',
        data: tpsVals,
        backgroundColor: 'rgba(59,130,246,0.4)',
        borderColor: '#3b82f6',
        borderWidth: 2,
        borderRadius: 4,
      }]
    },
    options: chartDefaults('Tokens/sec'),
  });
}

// ═══════════════════════════════════════════════════════════
// COMPARE
// ═══════════════════════════════════════════════════════════
async function startCompare() {
  const promptType = document.getElementById('compare-prompt').value;
  const runs       = document.getElementById('compare-runs').value;
  const btn        = document.getElementById('compare-start-btn');
  const log        = document.getElementById('compare-log');

  btn.disabled = true;
  btn.textContent = '⏳ Comparing...';
  log.innerHTML = '<span class="log-line">Auto-detecting installed models...</span>';
  document.getElementById('compare-results').style.display = 'none';
  document.getElementById('compare-chart-wrap').style.display = 'none';

  try {
    const startResp = await fetch(`${API}/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ models: null, prompt_type: promptType, runs: parseInt(runs) }),
    });

    clearInterval(benchPollTimer);
    benchPollTimer = setInterval(async () => {
      const status = await fetch(`${API}/benchmark/status`).then(r => r.json());
      log.innerHTML = status.progress.map(p => `<span class="log-line">✓ ${p}</span>`).join('\n');

      if (!status.running && status.result) {
        clearInterval(benchPollTimer);
        if (status.result.error) {
          log.innerHTML += `<span class="log-error">❌ ${status.result.error}</span>`;
        } else {
          renderCompareResults(status.result);
        }
        btn.disabled = false;
        btn.textContent = '⚖ Compare Models';
      }
    }, 1000);

  } catch (err) {
    log.innerHTML += `<span class="log-error">❌ ${err.message}</span>`;
    btn.disabled = false;
    btn.textContent = '⚖ Compare Models';
  }
}

function renderCompareResults(report) {
  const tbody = document.getElementById('compare-tbody');
  const best  = report.fastest_model;
  const bestT = report.lowest_ttft_model;

  tbody.innerHTML = report.results.map(r => `
    <tr>
      <td>${r.model}</td>
      <td class="${r.model === best ? 'best' : ''}">${r.avg_tps}</td>
      <td class="${r.model === bestT ? 'best' : ''}">${r.avg_ttft_ms} ms</td>
      <td>${r.avg_tpot_ms} ms</td>
      <td>${r.model === best ? '<span class="badge badge-green">⚡ Fastest</span>' : ''} ${r.model === bestT ? '<span class="badge badge-blue">⏱ Lowest TTFT</span>' : ''}</td>
    </tr>
  `).join('');
  document.getElementById('compare-results').style.display = 'block';

  // Chart
  const labels = report.results.map(r => r.model.split(':')[0]);
  const data   = report.results.map(r => r.avg_tps);
  drawCompareChart(labels, data);
}

function drawCompareChart(labels, data) {
  const wrap = document.getElementById('compare-chart-wrap');
  wrap.style.display = 'block';

  const ctx = document.getElementById('compare-chart').getContext('2d');
  if (compareChartInst) compareChartInst.destroy();

  const colors = ['rgba(59,130,246,0.5)', 'rgba(168,85,247,0.5)', 'rgba(34,197,94,0.5)'];
  const borders = ['#3b82f6', '#a855f7', '#22c55e'];

  compareChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Avg Tokens/sec',
        data,
        backgroundColor: colors.slice(0, data.length),
        borderColor: borders.slice(0, data.length),
        borderWidth: 2,
        borderRadius: 6,
      }]
    },
    options: chartDefaults('Tokens/sec'),
  });
}

// ═══════════════════════════════════════════════════════════
// CHARTS — TPS + GPU history
// ═══════════════════════════════════════════════════════════
function chartDefaults(label) {
  return {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0d1117',
        borderColor: '#1e2a3e',
        borderWidth: 1,
        titleColor: '#94a3b8',
        bodyColor: '#f1f5f9',
      }
    },
    scales: {
      x: {
        ticks: { color: '#475569', font: { size: 10 } },
        grid:  { color: '#1e2a3e' },
      },
      y: {
        ticks: { color: '#475569', font: { size: 10 } },
        grid:  { color: '#1e2a3e' },
        beginAtZero: true,
      }
    }
  };
}

function initTpsChart() {
  const ctx = document.getElementById('tps-chart').getContext('2d');
  tpsChartInst = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,0.1)',
        tension: 0.4,
        fill: true,
        pointRadius: 3,
        pointBackgroundColor: '#3b82f6',
      }]
    },
    options: {
      ...chartDefaults('tok/s'),
      plugins: { legend: { display: false } },
      animation: { duration: 300 },
    }
  });
}

function updateTpsChart() {
  if (!tpsChartInst) return;
  tpsChartInst.data.labels   = tpsHistory.map((_, i) => i + 1);
  tpsChartInst.data.datasets[0].data = tpsHistory;
  tpsChartInst.update('none');
}

function initGpuChart() {
  const ctx = document.getElementById('gpu-chart').getContext('2d');
  gpuChartInst = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245,158,11,0.1)',
        tension: 0.4,
        fill: true,
        pointRadius: 0,
      }]
    },
    options: {
      ...chartDefaults('%'),
      animation: { duration: 0 },
      scales: {
        x: { display: false },
        y: { ticks: { color: '#475569', font: { size: 10 } }, grid: { color: '#1e2a3e' }, min: 0, max: 100 }
      }
    }
  });
}

function updateGpuChart(util) {
  if (!gpuChartInst) return;
  const now = new Date().toLocaleTimeString();
  gpuHistory.push(util);
  if (gpuHistory.length > 30) gpuHistory.shift();
  gpuChartInst.data.labels   = gpuHistory.map((_, i) => i);
  gpuChartInst.data.datasets[0].data = gpuHistory;
  gpuChartInst.update('none');
}

// ═══════════════════════════════════════════════════════════
// QUEUE STATUS POLLING
// ═══════════════════════════════════════════════════════════
async function pollQueueStatus() {
  try {
    const data = await fetch(`${API}/queue/status`).then(r => r.json());
    document.getElementById('queue-active').textContent =
      `${data.active_inference} / ${data.max_concurrent}`;
    document.getElementById('queue-pct').textContent =
      `${data.queue_capacity_pct}%`;

    const stateEl = document.getElementById('queue-state');
    stateEl.textContent = data.backpressure_state;
    stateEl.style.color = {
      NORMAL: 'var(--green)',
      THROTTLING: 'var(--amber)',
      SHEDDING: 'var(--red)',
    }[data.backpressure_state] || 'var(--text-primary)';
  } catch (_) {
    const stateEl = document.getElementById('queue-state');
    if (stateEl) stateEl.textContent = 'API Offline';
  }
}

// ═══════════════════════════════════════════════════════════
// VRAM SIMULATOR
// ═══════════════════════════════════════════════════════════
async function runVRAMSim() {
  const params   = parseFloat(document.getElementById('sim-params').value) || 3;
  const bits     = parseInt(document.getElementById('sim-bits').value);
  const overhead = parseFloat(document.getElementById('sim-overhead').value);

  try {
    const data = await fetch(`${API}/simulate/vram`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params_billions: params, bits, overhead }),
    }).then(r => r.json());

    document.getElementById('sim-result').style.display = 'block';
    document.getElementById('sim-vram').textContent =
      `${data.estimated_vram_gb} GB (${data.estimated_vram_mb} MB)`;

    const fitsEl = document.getElementById('sim-fits');
    if (data.fits_in_free_vram) {
      fitsEl.textContent = '✅ Yes (free VRAM)';
      fitsEl.style.color = 'var(--green)';
    } else if (data.fits_in_gpu) {
      fitsEl.textContent = '⚠ Maybe (total VRAM)';
      fitsEl.style.color = 'var(--amber)';
    } else {
      fitsEl.textContent = '❌ No — OOM risk';
      fitsEl.style.color = 'var(--red)';
    }

    document.getElementById('sim-formula').textContent = data.formula;
  } catch (err) {
    document.getElementById('sim-result').style.display = 'block';
    document.getElementById('sim-vram').textContent = 'Error — API offline?';
    document.getElementById('sim-fits').textContent = '—';
  }
}

// ═══════════════════════════════════════════════════════════
// GPU POLLING
// ═══════════════════════════════════════════════════════════
async function pollGpu() {
  try {
    const data = await fetch(`${API}/gpu`).then(r => r.json());
    const pct  = data.vram_used_pct;
    const util = data.gpu_utilization_pct;
    const temp = data.temperature_c;

    // Header quick stats
    document.getElementById('header-vram').textContent = `${pct}%`;
    document.getElementById('header-gpu').textContent  = `${util}%`;

    // Right sidebar
    document.getElementById('gpu-util').textContent  = `${util}%`;
    document.getElementById('gpu-temp').textContent  = `${temp}°`;
    document.getElementById('gpu-vram-used').textContent = data.vram_used_mb;
    document.getElementById('gpu-power').textContent = data.power_draw_w ? `${data.power_draw_w}W` : 'N/A';
    document.getElementById('gpu-name-label').textContent = data.name;

    // Color coding
    const utilEl = document.getElementById('gpu-util');
    utilEl.className = 'gpu-metric-value ' + (util > 80 ? 'danger' : util > 50 ? 'hot' : 'cool');
    const tempEl = document.getElementById('gpu-temp');
    tempEl.className = 'gpu-metric-value ' + (temp > 80 ? 'danger' : temp > 65 ? 'hot' : 'cool');

    // Progress bars
    document.getElementById('vram-bar').style.width = `${pct}%`;
    document.getElementById('vram-pct-label').textContent = `${pct}%`;
    const vramFill = document.getElementById('vram-bar');
    vramFill.className = 'progress-fill' + (pct > 85 ? ' red' : pct > 65 ? ' amber' : '');

    document.getElementById('gpu-util-bar').style.width = `${util}%`;
    document.getElementById('gpu-pct-label').textContent = `${util}%`;

    updateGpuChart(util);
  } catch (_) {
    document.getElementById('header-vram').textContent = 'ERR';
  }
}

// ═══════════════════════════════════════════════════════════
// HEALTH CHECK + MODEL LIST
// ═══════════════════════════════════════════════════════════
async function checkHealth() {
  try {
    const data = await fetch(`${API}/health`).then(r => r.json());
    const dot  = document.getElementById('ollama-dot');
    const txt  = document.getElementById('ollama-status-text');

    if (data.ollama.status === 'online') {
      dot.className = 'status-dot online';
      txt.textContent = 'Ollama Online';
    } else {
      dot.className = 'status-dot offline';
      txt.textContent = 'Ollama Offline';
    }
  } catch (_) {
    document.getElementById('ollama-dot').className = 'status-dot offline';
    document.getElementById('ollama-status-text').textContent = 'API Offline';
  }
}

const FALLBACK_MODELS = [
  { name: 'llama3.2:3b',  size_gb: 2.0 },
  { name: 'phi3:mini',    size_gb: 2.3 },
  { name: 'tinyllama',    size_gb: 0.7 },
];

async function loadModels() {
  let models = FALLBACK_MODELS;
  const el = document.getElementById('models-list');

  try {
    const data = await fetch(`${API}/models`).then(r => r.json());
    if (data.models && data.models.length > 0) {
      models = data.models;
      el.innerHTML = models.map(m =>
        `<div class="stat-row">
          <span class="stat-label" style="font-family:'JetBrains Mono',monospace">${m.name}</span>
          <span class="stat-value">${m.size_gb} GB</span>
        </div>`
      ).join('');
    } else {
      el.innerHTML = '<span style="color:var(--text-muted)">No models installed yet.<br>Run: ollama pull llama3.2:3b</span>';
    }
  } catch (_) {
    el.innerHTML = '<span style="color:var(--amber)">⚠ Ollama offline — start Ollama first.</span>';
  }

  // Populate model dropdowns with available (or fallback) models
  const opts = models.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
  ['model-select', 'bench-model'].forEach(id => {
    const sel = document.getElementById(id);
    if (sel) {
      sel.innerHTML = opts;
    }
  });
  onModelChange();
}

// ═══════════════════════════════════════════════════════════
// RESEARCH LOG — /history/*
// ═══════════════════════════════════════════════════════════
let histTimelineChartInst = null;

async function loadHistory() {
  const btn = document.getElementById('hist-refresh-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Loading...';

  const modelFilter = document.getElementById('hist-model-filter').value;
  const limit       = parseInt(document.getElementById('hist-limit').value);

  try {
    const [agg, runs, timeline] = await Promise.all([
      fetch(`${API}/history/aggregate`).then(r => r.json()),
      fetch(`${API}/history/runs?limit=${limit}${modelFilter ? `&model=${encodeURIComponent(modelFilter)}` : ''}`).then(r => r.json()),
      fetch(`${API}/history/timeline?limit=${limit * 2}${modelFilter ? `&model=${encodeURIComponent(modelFilter)}` : ''}`).then(r => r.json()),
    ]);

    const hasData = (agg.aggregates && agg.aggregates.length > 0) || (runs.runs && runs.runs.length > 0);
    document.getElementById('hist-empty').style.display = hasData ? 'none' : 'block';
    document.getElementById('hist-aggregate-card').style.display = (agg.aggregates && agg.aggregates.length > 0) ? 'block' : 'none';
    document.getElementById('hist-recent-card').style.display    = (runs.runs && runs.runs.length > 0) ? 'block' : 'none';
    document.getElementById('hist-timeline-wrap').style.display  = (timeline.points && timeline.points.length > 1) ? 'block' : 'none';

    renderHistAggregate(agg.aggregates || []);
    renderHistRuns(runs.runs || []);
    renderHistTimeline(timeline.points || []);
    populateHistModelFilter(agg.aggregates || []);
  } catch (err) {
    document.getElementById('hist-empty').style.display = 'block';
    document.getElementById('hist-empty').innerHTML =
      `<span style="color:var(--red);">❌ Failed to load history: ${err.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🔄 Refresh';
  }
}

function populateHistModelFilter(aggregates) {
  const sel = document.getElementById('hist-model-filter');
  const current = sel.value;
  const names = aggregates.map(a => a.model);
  sel.innerHTML = '<option value="">All models</option>' +
    names.map(n => `<option value="${n}">${n}</option>`).join('');
  if (names.includes(current)) sel.value = current;
}

function renderHistAggregate(rows) {
  const tbody = document.getElementById('hist-agg-tbody');
  if (rows.length === 0) { tbody.innerHTML = ''; return; }
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td style="font-family:'JetBrains Mono',monospace;">${r.model}</td>
      <td>${r.runs}</td>
      <td class="best">${r.avg_tps ?? '—'}</td>
      <td>${r.avg_ttft_ms ?? '—'}</td>
      <td>${r.avg_tpot_ms ?? '—'}</td>
      <td>${r.avg_queue_wait_ms ?? 0} ms</td>
      <td>${r.avg_vram_mb ?? '—'} MB</td>
    </tr>
  `).join('');
}

function renderHistRuns(rows) {
  const tbody = document.getElementById('hist-runs-tbody');
  if (rows.length === 0) { tbody.innerHTML = ''; return; }
  tbody.innerHTML = rows.map(r => {
    const t = new Date(r.ts * 1000).toLocaleTimeString();
    const state = r.backpressure_state || '—';
    const stateColor = { NORMAL: 'var(--green)', THROTTLING: 'var(--amber)', SHEDDING: 'var(--red)' }[state] || 'var(--text-muted)';
    return `
      <tr>
        <td>${t}</td>
        <td>${r.endpoint}</td>
        <td style="font-family:'JetBrains Mono',monospace;">${r.model}</td>
        <td class="best">${r.tokens_per_second != null ? r.tokens_per_second.toFixed(1) : (r.error ? '<span style="color:var(--red)">err</span>' : '—')}</td>
        <td>${r.ttft_ms != null ? r.ttft_ms.toFixed(0) : '—'}</td>
        <td>${r.tpot_ms != null ? r.tpot_ms.toFixed(1) : '—'}</td>
        <td>${r.output_tokens ?? '—'}</td>
        <td>${r.queue_wait_ms != null ? r.queue_wait_ms.toFixed(0) : '0'}</td>
        <td><span style="color:${stateColor};font-weight:600;">${state}</span></td>
      </tr>
    `;
  }).join('');
}

function renderHistTimeline(points) {
  if (points.length < 2) return;
  // Group by model — one line per model
  const byModel = {};
  points.forEach(p => {
    if (!byModel[p.model]) byModel[p.model] = [];
    byModel[p.model].push({ x: p.ts * 1000, y: p.tps });
  });

  const palette = ['#3b82f6', '#a855f7', '#22c55e', '#f59e0b', '#ec4899', '#06b6d4'];
  const datasets = Object.entries(byModel).map(([model, data], i) => ({
    label: model,
    data,
    borderColor: palette[i % palette.length],
    backgroundColor: palette[i % palette.length] + '22',
    tension: 0.3,
    fill: false,
    pointRadius: 2,
  }));

  const ctx = document.getElementById('hist-timeline-chart').getContext('2d');
  if (histTimelineChartInst) histTimelineChartInst.destroy();
  histTimelineChartInst = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true, labels: { color: '#94a3b8', font: { size: 11 } } },
        tooltip: {
          backgroundColor: '#0d1117', borderColor: '#1e2a3e', borderWidth: 1,
          titleColor: '#94a3b8', bodyColor: '#f1f5f9',
        }
      },
      scales: {
        x: {
          type: 'linear',
          ticks: {
            color: '#475569', font: { size: 10 },
            callback: (v) => new Date(v).toLocaleTimeString(),
          },
          grid: { color: '#1e2a3e' },
        },
        y: {
          ticks: { color: '#475569', font: { size: 10 } },
          grid: { color: '#1e2a3e' },
          title: { display: true, text: 'Tokens/sec', color: '#94a3b8', font: { size: 11 } },
          beginAtZero: true,
        }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════
// STRESS TEST — /loadtest
// ═══════════════════════════════════════════════════════════
let ltTimelineChartInst = null;
let ltStatusPoller = null;

async function runLoadTest() {
  const btn   = document.getElementById('lt-start-btn');
  const log   = document.getElementById('lt-log');
  const model = document.getElementById('lt-model').value;
  const conc  = parseInt(document.getElementById('lt-concurrency').value);
  const mxt   = parseInt(document.getElementById('lt-maxtok').value);

  btn.disabled = true;
  btn.textContent = '🔥 Firing...';
  log.innerHTML = `<span class="log-line">Launching ${conc} parallel requests against ${model}...</span>`;
  document.getElementById('lt-summary-card').style.display    = 'none';
  document.getElementById('lt-timeline-wrap').style.display   = 'none';

  // Watch queue state visibly during the test
  let watchTicks = 0;
  const stateBuf = [];
  ltStatusPoller = setInterval(async () => {
    try {
      const q = await fetch(`${API}/queue/status`).then(r => r.json());
      stateBuf.push(`<span class="log-line">t+${watchTicks * 0.5}s — active=${q.active_inference} wait=${q.waiting ?? 0} cap=${q.queue_capacity_pct}% — <span style="color:${q.backpressure_state === 'SHEDDING' ? 'var(--red)' : q.backpressure_state === 'THROTTLING' ? 'var(--amber)' : 'var(--green)'};">${q.backpressure_state}</span></span>`);
      if (stateBuf.length > 12) stateBuf.shift();
      log.innerHTML = stateBuf.join('');
      watchTicks++;
    } catch (_) { /* ignore — test in flight */ }
  }, 500);

  try {
    const resp = await fetch(`${API}/loadtest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, concurrency: conc, max_tokens: mxt }),
    });
    clearInterval(ltStatusPoller);

    if (!resp.ok) {
      const err = await resp.text();
      log.innerHTML += `<span class="log-error">❌ ${err}</span>`;
      return;
    }
    const result = await resp.json();
    log.innerHTML += `<span class="log-line" style="color:var(--green);">✅ Test complete in ${result.duration_s}s</span>`;

    renderLoadTestSummary(result);
    renderLoadTestTimeline(result.timeline);
    loadPastLoadTests();
  } catch (err) {
    clearInterval(ltStatusPoller);
    log.innerHTML += `<span class="log-error">❌ ${err.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Fire Load Test';
  }
}

function renderLoadTestSummary(r) {
  const tbody = document.getElementById('lt-summary-tbody');
  const stateColor = { NORMAL: 'var(--green)', THROTTLING: 'var(--amber)', SHEDDING: 'var(--red)' }[r.final_state] || 'var(--text-muted)';
  tbody.innerHTML = `
    <tr><td>Concurrency</td><td class="best">${r.concurrency}</td></tr>
    <tr><td>Total duration</td><td>${r.duration_s} s</td></tr>
    <tr><td>✅ Successful</td><td class="best" style="color:var(--green);">${r.successful}</td></tr>
    <tr><td>⚠ Rejected (429)</td><td style="color:var(--amber);">${r.rejected_429}</td></tr>
    <tr><td>❌ Errored</td><td style="color:var(--red);">${r.errored}</td></tr>
    <tr><td>Avg latency</td><td>${r.latency.avg_s ?? '—'} s</td></tr>
    <tr><td>p50 latency</td><td>${r.latency.p50_s ?? '—'} s</td></tr>
    <tr><td>p95 latency</td><td class="best">${r.latency.p95_s ?? '—'} s</td></tr>
    <tr><td>Min / Max latency</td><td>${r.latency.min_s ?? '—'} / ${r.latency.max_s ?? '—'} s</td></tr>
    <tr><td>Max queue depth seen</td><td>${r.max_queue_depth_seen}</td></tr>
    <tr><td>Final state</td><td><span style="color:${stateColor};font-weight:700;">${r.final_state}</span></td></tr>
  `;
  document.getElementById('lt-summary-card').style.display = 'block';
}

function renderLoadTestTimeline(timeline) {
  if (!timeline || timeline.length < 2) return;
  const labels = timeline.map(p => p.t.toFixed(1) + 's');
  const queuePct = timeline.map(p => p.queue_pct);
  const vramPct  = timeline.map(p => p.vram_pct);
  const gpuUtil  = timeline.map(p => p.gpu_util_pct);

  const ctx = document.getElementById('lt-timeline-chart').getContext('2d');
  if (ltTimelineChartInst) ltTimelineChartInst.destroy();
  ltTimelineChartInst = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Queue capacity %',
          data: queuePct,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.15)',
          tension: 0.3, fill: true, pointRadius: 1,
        },
        {
          label: 'VRAM %',
          data: vramPct,
          borderColor: '#a855f7',
          backgroundColor: 'rgba(168,85,247,0.10)',
          tension: 0.3, fill: false, pointRadius: 1,
        },
        {
          label: 'GPU Util %',
          data: gpuUtil,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245,158,11,0.10)',
          tension: 0.3, fill: false, pointRadius: 1,
        },
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true, labels: { color: '#94a3b8', font: { size: 11 } } },
      },
      scales: {
        x: { ticks: { color: '#475569', font: { size: 9 } }, grid: { color: '#1e2a3e' } },
        y: { ticks: { color: '#475569', font: { size: 10 } }, grid: { color: '#1e2a3e' }, min: 0, max: 100 },
      }
    }
  });
  document.getElementById('lt-timeline-wrap').style.display = 'block';
}

async function loadPastLoadTests() {
  try {
    const data = await fetch(`${API}/history/loadtests?limit=15`).then(r => r.json());
    const tests = data.loadtests || [];
    if (tests.length === 0) {
      document.getElementById('lt-past-card').style.display = 'none';
      return;
    }
    const tbody = document.getElementById('lt-past-tbody');
    tbody.innerHTML = tests.map(t => {
      const when = new Date(t.ts_start * 1000).toLocaleTimeString();
      const stateColor = { NORMAL: 'var(--green)', THROTTLING: 'var(--amber)', SHEDDING: 'var(--red)' }[t.final_state] || 'var(--text-muted)';
      return `
        <tr>
          <td>${when}</td>
          <td style="font-family:'JetBrains Mono',monospace;">${t.model}</td>
          <td>${t.concurrency}</td>
          <td style="color:var(--green);">${t.successful}</td>
          <td style="color:var(--amber);">${t.rejected_429}</td>
          <td style="color:var(--red);">${t.errored}</td>
          <td>${t.p50_latency_s ?? '—'}s</td>
          <td>${t.p95_latency_s ?? '—'}s</td>
          <td><span style="color:${stateColor};font-weight:600;">${t.final_state}</span></td>
        </tr>
      `;
    }).join('');
    document.getElementById('lt-past-card').style.display = 'block';
  } catch (_) { /* ignore */ }
}

// Auto-populate stress-test model dropdown from /models
async function syncLoadTestModels() {
  try {
    const data = await fetch(`${API}/models`).then(r => r.json());
    const sel = document.getElementById('lt-model');
    if (sel && data.models && data.models.length > 0) {
      const cur = sel.value;
      sel.innerHTML = data.models.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
      if (data.models.find(m => m.name === cur)) sel.value = cur;
    }
  } catch (_) { /* ignore */ }
}

// Lazy-load history/loadtest data the first time their tab is opened
const _origShowTab = showTab;
showTab = function(name) {
  _origShowTab(name);
  if (name === 'history') loadHistory();
  if (name === 'loadtest') { syncLoadTestModels(); loadPastLoadTests(); }
};

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initTpsChart();
  initGpuChart();
  checkHealth();
  loadModels();
  pollGpu();
  pollQueueStatus();

  // Poll GPU every 2.5s
  setInterval(pollGpu, 2500);
  // Queue status every 3s
  setInterval(pollQueueStatus, 3000);
  // Health check every 10s
  setInterval(checkHealth, 10000);
});
