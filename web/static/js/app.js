/**
 * M250 Weld Repair Property Predictor — Dynamic Frontend Engine
 * Handles REST API calls, real-time Chart.js rendering, and interactive UI state.
 */

let appMetadata = null;
let currentPrediction = null;
let stressStrainChart = null;
let leaderboardChart = null;
let shapChart = null;

// Initialize on DOM loaded
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupR1Toggle();
  setupEventListeners();
  loadMetadata();
});

/* ==========================================================================
   Navigation Tabs
   ========================================================================== */
function setupTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabViews = document.querySelectorAll('.tab-view');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');

      tabBtns.forEach(b => b.classList.remove('active'));
      tabViews.forEach(v => {
        v.classList.remove('active');
        v.style.display = 'none';
      });

      btn.classList.add('active');
      const activeView = document.getElementById(targetTab);
      if (activeView) {
        activeView.classList.add('active');
        activeView.style.display = 'block';
      }

      // Trigger lazy charts and data initialization in isolated try-catch
      try {
        if (targetTab === 'tab-curves') {
          renderStressStrainCurves();
        } else if (targetTab === 'tab-leaderboard') {
          loadLeaderboard();
        } else if (targetTab === 'tab-xai') {
          loadShapImportance();
        } else if (targetTab === 'tab-training') {
          loadTrainingStatus();
          loadTrainingComparison();
        } else if (targetTab === 'tab-metrics') {
          loadDetailedMetrics();
        } else if (targetTab === 'tab-explorer') {
          loadDataExplorer();
        }
      } catch (err) {
        console.error('Error loading tab content:', err);
      }
    });
  });
}

function setupR1Toggle() {
  const toggle = document.getElementById('r1-toggle');
  const fields = document.getElementById('r1-fields');
  if (toggle && fields) {
    toggle.addEventListener('change', () => {
      if (toggle.checked) {
        fields.classList.add('open');
      } else {
        fields.classList.remove('open');
      }
    });
  }
}

function setupEventListeners() {
  // Predict button
  const predictBtn = document.getElementById('predict-btn');
  if (predictBtn) {
    predictBtn.addEventListener('click', executePrediction);
  }

  // Export PDF button
  const exportPdfBtn = document.getElementById('export-pdf-btn');
  if (exportPdfBtn) {
    exportPdfBtn.addEventListener('click', exportPdfReport);
  }

  // Export Curve PNG
  const exportCurvePngBtn = document.getElementById('export-curve-png');
  if (exportCurvePngBtn) {
    exportCurvePngBtn.addEventListener('click', () => {
      if (stressStrainChart) {
        const link = document.createElement('a');
        link.download = `M250_Stress_Strain_Curve_${Date.now()}.png`;
        link.href = stressStrainChart.toBase64Image();
        link.click();
        showToast('Stress-Strain chart exported as PNG');
      }
    });
  }

  // Refresh Curves button
  const refreshCurvesBtn = document.getElementById('refresh-curves-btn');
  if (refreshCurvesBtn) {
    refreshCurvesBtn.addEventListener('click', renderStressStrainCurves);
  }

  // SHAP controls
  const shapTargetSelect = document.getElementById('shap-target-select');
  if (shapTargetSelect) {
    shapTargetSelect.addEventListener('change', loadShapImportance);
  }
  const shapModelSelect = document.getElementById('shap-model-select');
  if (shapModelSelect) {
    shapModelSelect.addEventListener('change', loadShapImportance);
  }
  const btnComputeShap = document.getElementById('btn-compute-shap');
  if (btnComputeShap) {
    btnComputeShap.addEventListener('click', loadShapImportance);
  }

  // Training controls
  const startTrainBtn = document.getElementById('start-train-btn');
  if (startTrainBtn) {
    startTrainBtn.addEventListener('click', startTraining);
  }
  const clearLogsBtn = document.getElementById('clear-logs-btn');
  if (clearLogsBtn) {
    clearLogsBtn.addEventListener('click', () => {
      const consoleEl = document.getElementById('training-console-logs');
      if (consoleEl) consoleEl.textContent = '--- View Cleared ---';
    });
  }
  const trainViewSelect = document.getElementById('train-view-select');
  if (trainViewSelect) {
    trainViewSelect.addEventListener('change', () => {
      loadTrainingComparison(trainViewSelect.value);
    });
  }

  // Detailed metrics model selector
  const detailedModelSelect = document.getElementById('detailed-model-select');
  if (detailedModelSelect) {
    detailedModelSelect.addEventListener('change', () => {
      loadDetailedMetrics(detailedModelSelect.value);
    });
  }

  // Scatter plot selectors
  const scatterX = document.getElementById('scatter-x-select');
  const scatterY = document.getElementById('scatter-y-select');
  if (scatterX) scatterX.addEventListener('change', updateScatterPlot);
  if (scatterY) scatterY.addEventListener('change', updateScatterPlot);
}

/* ==========================================================================
   Load Metadata & Presets
   ========================================================================== */
async function loadMetadata() {
  try {
    const res = await fetch('/api/metadata');
    const data = await res.json();
    if (data.status === 'success') {
      appMetadata = data;
      populateModelDropdown(data.models, data.best_model);
      populatePresets(data.presets);
      // Auto run prediction on nominal preset once loaded
      executePrediction();
    }
  } catch (err) {
    console.error('Error fetching metadata:', err);
    showToast('Failed to load system metadata from server', true);
  }
}

function populateModelDropdown(models, bestModel) {
  const select = document.getElementById('model-select');
  if (!select) return;
  select.innerHTML = '';

  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m === 'Auto-Best' ? `Auto-Best (${bestModel})` : m;
    select.appendChild(opt);
  });
}

function populatePresets(presets) {
  const container = document.getElementById('preset-pills');
  if (!container || !presets) return;
  container.innerHTML = '';

  Object.keys(presets).forEach(name => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset-btn';
    btn.textContent = name;
    btn.addEventListener('click', () => applyPreset(presets[name]));
    container.appendChild(btn);
  });
}

function applyPreset(preset) {
  if (preset.specimen) {
    const specElem = document.getElementById('specimen-select');
    if (specElem) specElem.value = preset.specimen;
  }
  if (preset.location) {
    const locElem = document.getElementById('location-select');
    if (locElem) locElem.value = preset.location;
  }

  // Set R0
  if (preset.r0) {
    Object.keys(preset.r0).forEach(key => {
      const input = document.getElementById(`in_${key}`);
      if (input) input.value = preset.r0[key];
    });
  }

  // Set R1
  const toggle = document.getElementById('r1-toggle');
  const fields = document.getElementById('r1-fields');
  if (preset.include_r1 && preset.r1) {
    if (toggle) toggle.checked = true;
    if (fields) fields.classList.add('open');
    Object.keys(preset.r1).forEach(key => {
      const input = document.getElementById(`in_${key}`);
      if (input) input.value = preset.r1[key];
    });
  } else {
    if (toggle) toggle.checked = false;
    if (fields) fields.classList.remove('open');
  }

  showToast(`Applied preset parameters`);
}

/* ==========================================================================
   Prediction API Call & UI Update
   ========================================================================== */
async function executePrediction() {
  const predictBtn = document.getElementById('predict-btn');
  if (predictBtn) {
    predictBtn.disabled = true;
    predictBtn.innerHTML = `<span>⏳</span> Computing MC Dropout (50 passes)...`;
  }

  // Gather values
  const r0 = {};
  const r0Keys = ['R0_UTS', 'R0_YS', 'R0_K1c', 'R0_Elongation', 'R0_HAZ_Width', 'R0_Gauge_Depth', 'R0_Austenite_Content'];
  r0Keys.forEach(k => {
    const elem = document.getElementById(`in_${k}`);
    r0[k] = elem ? parseFloat(elem.value) || 0 : 0;
  });

  const includeR1 = document.getElementById('r1-toggle')?.checked || false;
  const r1 = {};
  if (includeR1) {
    const r1Keys = ['R1_UTS', 'R1_YS', 'R1_K1c', 'R1_Elongation', 'R1_HAZ_Width', 'R1_Gauge_Depth', 'R1_Austenite_Content'];
    r1Keys.forEach(k => {
      const elem = document.getElementById(`in_${k}`);
      r1[k] = elem ? parseFloat(elem.value) || 0 : 0;
    });
  }

  const specimen = document.getElementById('specimen-select')?.value || 'S139';
  const location = document.getElementById('location-select')?.value || 'Center';
  const model = document.getElementById('model-select')?.value || 'Auto-Best';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        r0, r1, include_r1: includeR1, specimen, location, model
      })
    });

    const data = await res.json();
    if (data.status === 'success') {
      currentPrediction = data;
      renderPredictionResults(data);
      showToast(`Prediction complete using ${data.model_used}`);
    } else {
      showToast(data.message || 'Prediction failed', true);
    }
  } catch (err) {
    console.error('Prediction request error:', err);
    showToast('Failed to connect to backend prediction server', true);
  } finally {
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.innerHTML = `<span>🔬</span> RUN R2 PREDICTION`;
    }
  }
}

function renderPredictionResults(data) {
  const preds = data.predictions;

  // 1. Update 6 Property Result Cards
  const cardMap = {
    'R2_UTS': 'res-uts',
    'R2_YS': 'res-ys',
    'R2_Elongation': 'res-elon',
    'R2_HAZ_Width': 'res-haz',
    'R2_Gauge_Depth': 'res-gauge',
    'R2_Austenite_Content': 'res-aust'
  };

  Object.keys(cardMap).forEach(key => {
    const p = preds[key];
    const prefix = cardMap[key];
    if (!p) return;

    const valElem = document.getElementById(`${prefix}-val`);
    const ciElem = document.getElementById(`${prefix}-ci`);
    const minElem = document.getElementById(`${prefix}-min`);

    if (valElem) animateValue(valElem, p.predicted, 1);
    if (ciElem) ciElem.textContent = `95% CI: ±${p.ci_95} ${p.unit}`;
    if (minElem) minElem.textContent = `Min Guaranteed: ${p.min_guaranteed} ${p.unit}`;
  });

  // 2. Comparison Table
  const tbody = document.getElementById('comparison-tbody');
  if (tbody) {
    tbody.innerHTML = '';
    Object.keys(preds).forEach(key => {
      const p = preds[key];
      const tr = document.createElement('tr');
      const r0Val = p.r0_value !== null ? `${p.r0_value} ${p.unit}` : '—';
      const r2Val = `${p.predicted} ${p.unit}`;
      const degHtml = p.degradation_pct !== null
        ? `<span class="badge-degrade">-${p.degradation_pct}%</span>`
        : '—';

      tr.innerHTML = `
        <td><strong>${p.name}</strong></td>
        <td>${r0Val}</td>
        <td style="color: var(--primary); font-weight: 700;">${r2Val}</td>
        <td>${degHtml}</td>
        <td style="color: #34d399; font-weight: 600;">${p.min_guaranteed} ${p.unit}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // If curve tab is open, update curve too
  if (document.getElementById('tab-curves')?.classList.contains('active')) {
    renderStressStrainCurves();
  }
}

function animateValue(element, target, decimals = 1) {
  const start = 0;
  const duration = 500;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const current = start + (target - start) * easeOutQuad(progress);
    element.textContent = current.toFixed(decimals);
    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}

function easeOutQuad(x) {
  return 1 - (1 - x) * (1 - x);
}

/* ==========================================================================
   Stress-Strain Curve Visualization (Chart.js)
   ========================================================================== */
async function renderStressStrainCurves() {
  const canvas = document.getElementById('stressStrainCanvas');
  if (!canvas) return;

  const r0_uts = parseFloat(document.getElementById('in_R0_UTS')?.value) || 1601.0;
  const r0_ys = parseFloat(document.getElementById('in_R0_YS')?.value) || 1570.0;
  const pred_uts = currentPrediction?.summary?.r2_uts || 1448.0;
  const pred_ys = currentPrediction?.summary?.r2_ys || 1366.0;

  try {
    const res = await fetch('/api/curve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ r0_uts, r0_ys, pred_uts, pred_ys })
    });
    const data = await res.json();
    if (data.status !== 'success') return;

    const curves = data.curves;
    const datasets = [];

    // Colors matching research aesthetics
    const styles = {
      'Parent Material': { color: '#38bdf8', dash: [], width: 2.5 },
      'R0': { color: '#f59e0b', dash: [], width: 2.5 },
      'R1': { color: '#94a3b8', dash: [], width: 2 },
      'R2 Reference': { color: '#fbbf24', dash: [4, 4], width: 2 },
      'R2 Predicted': { color: '#10b981', dash: [], width: 3 }
    };

    // Shaded 95% Confidence Band for Predicted Curve
    if (curves['R2 Predicted']?.upper_ci && curves['R2 Predicted']?.lower_ci) {
      const pred = curves['R2 Predicted'];
      const upperPoints = pred.strain.map((s, i) => ({ x: s, y: pred.upper_ci[i] }));
      const lowerPoints = pred.strain.map((s, i) => ({ x: s, y: pred.lower_ci[i] }));

      datasets.push({
        label: '95% CI (Upper)',
        data: upperPoints,
        borderColor: 'transparent',
        backgroundColor: 'rgba(16, 185, 129, 0.12)',
        fill: '+1',
        pointRadius: 0,
        tension: 0.3
      });

      datasets.push({
        label: '95% CI Band',
        data: lowerPoints,
        borderColor: 'transparent',
        backgroundColor: 'transparent',
        pointRadius: 0,
        tension: 0.3
      });
    }

    // Add 5 curves
    Object.keys(curves).forEach(name => {
      const c = curves[name];
      const style = styles[name] || { color: '#ffffff', dash: [], width: 2 };
      const points = c.strain.map((s, i) => ({ x: s, y: c.stress[i] }));

      datasets.push({
        label: name,
        data: points,
        borderColor: style.color,
        borderWidth: style.width,
        borderDash: style.dash,
        backgroundColor: 'transparent',
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.25
      });
    });

    if (stressStrainChart) stressStrainChart.destroy();

    const ctx = canvas.getContext('2d');
    stressStrainChart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600 },
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 12 },
              filter: item => !item.text.includes('CI')
            }
          },
          tooltip: {
            backgroundColor: '#0f172a',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            titleColor: '#38bdf8',
            bodyColor: '#f8fafc',
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} MPa at ${ctx.parsed.x.toFixed(0)} με`
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Strain (Microstrain με)', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748b' }
          },
          y: {
            title: { display: true, text: 'Engineering Stress (MPa)', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748b' }
          }
        }
      }
    });

  } catch (err) {
    console.error('Error rendering curve:', err);
  }
}

/* ==========================================================================
   Model Leaderboard & Comparison
   ========================================================================== */
async function loadLeaderboard() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    if (data.status !== 'success') return;

    const tbody = document.getElementById('leaderboard-tbody');
    if (tbody) {
      tbody.innerHTML = '';
      data.leaderboard.forEach((m, idx) => {
        const isBest = m.model === data.best_model;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>#${idx + 1}</strong></td>
          <td>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <strong>${m.model}</strong>
              ${isBest ? '<span class="status-pill" style="font-size: 0.65rem; padding: 0.1rem 0.4rem;">BEST</span>' : ''}
            </div>
          </td>
          <td style="color: var(--primary); font-weight: 700;">${m.r2.toFixed(4)}</td>
          <td>${m.mae.toFixed(2)} MPa</td>
          <td>${m.rmse.toFixed(2)} MPa</td>
          <td style="color: #34d399; font-weight: 600;">${m.accuracy.toFixed(2)}%</td>
          <td>${m.uts_rmse.toFixed(1)} MPa</td>
          <td>${m.ys_rmse.toFixed(1)} MPa</td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Chart
    const canvas = document.getElementById('leaderboardChartCanvas');
    if (canvas) {
      const labels = data.leaderboard.map(m => m.model);
      const r2Values = data.leaderboard.map(m => m.r2 * 100);
      const accValues = data.leaderboard.map(m => m.accuracy);

      if (leaderboardChart) leaderboardChart.destroy();
      leaderboardChart = new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'R² Score (%)',
              data: r2Values,
              backgroundColor: 'rgba(56, 189, 248, 0.75)',
              borderRadius: 6
            },
            {
              label: 'Overall Accuracy (%)',
              data: accValues,
              backgroundColor: 'rgba(16, 185, 129, 0.75)',
              borderRadius: 6
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#94a3b8' } }
          },
          scales: {
            x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
            y: { min: 70, max: 100, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } }
          }
        }
      });
    }

  } catch (err) {
    console.error('Error loading leaderboard:', err);
  }
}

/* ==========================================================================
   Explainable AI & SHAP Feature Importance
   ========================================================================== */
async function loadShapImportance() {
  const targetSelect = document.getElementById('shap-target-select');
  const modelSelect = document.getElementById('shap-model-select');
  const target = targetSelect?.value || 'Austenite Content';
  const model = modelSelect?.value || 'Extratrees';

  const statusText = document.getElementById('shap-status-desc');
  const progressFill = document.getElementById('shap-progress-fill');
  const chartTitle = document.getElementById('shap-chart-title');
  const rankingTitle = document.getElementById('shap-ranking-title');
  const rankingList = document.getElementById('shap-ranking-list');
  const insightText = document.getElementById('shap-insight-text');
  const computeBtn = document.getElementById('btn-compute-shap');

  if (statusText) statusText.textContent = `Computing SHAP for ${model} — ${target}…`;
  if (progressFill) progressFill.style.width = '45%';
  if (computeBtn) {
    computeBtn.disabled = true;
    computeBtn.innerHTML = `<span>⏳</span> Computing…`;
  }

  try {
    const res = await fetch(`/api/shap?target=${encodeURIComponent(target)}&model=${encodeURIComponent(model)}`);
    const data = await res.json();

    if (progressFill) progressFill.style.width = '100%';
    if (computeBtn) {
      computeBtn.disabled = false;
      computeBtn.innerHTML = `<span>🔍</span> Compute SHAP`;
    }

    if (data.status !== 'success') {
      if (statusText) statusText.textContent = `Error computing SHAP: ${data.message || 'Unknown error'}`;
      return;
    }

    if (statusText) statusText.textContent = `SHAP computed for ${data.model} — ${data.target}`;

    const canvas = document.getElementById('shapChartCanvas');
    if (!canvas) return;

    if (shapChart) {
      shapChart.destroy();
      shapChart = null;
    }

    if (data.mode === 'comparison') {
      // Grouped Vertical Bar Chart
      if (chartTitle) chartTitle.textContent = `Feature Importance Across All Targets — ${data.model}`;
      if (rankingTitle) rankingTitle.innerHTML = `<span>📋</span> Average Across Targets`;

      const ctx = canvas.getContext('2d');
      shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: data.features_comparison.labels,
          datasets: data.features_comparison.datasets.map(ds => ({
            label: ds.label,
            data: ds.data,
            backgroundColor: ds.backgroundColor,
            borderRadius: 2,
            maxBarThickness: 14
          }))
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              position: 'right',
              labels: {
                color: '#e2e8f0',
                boxWidth: 12,
                boxHeight: 12,
                font: { size: 10, family: 'Inter' }
              }
            },
            tooltip: {
              callbacks: {
                title: items => {
                  const idx = items[0]?.dataIndex;
                  return data.features_comparison.full_labels[idx] || items[0]?.label;
                },
                label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(6)}`
              }
            }
          },
          scales: {
            x: {
              ticks: { color: '#e2e8f0', font: { size: 9 }, maxRotation: 25, minRotation: 25 },
              grid: { display: false }
            },
            y: {
              title: { display: true, text: 'Mean |SHAP Value|', color: '#94a3b8', font: { size: 10 } },
              ticks: { color: '#94a3b8', font: { size: 9 } },
              grid: { color: 'rgba(255, 255, 255, 0.08)' }
            }
          }
        }
      });

    } else {
      // Single Target Horizontal Bar Chart
      if (chartTitle) chartTitle.textContent = `SHAP Feature Importance — ${data.target} (${data.model})`;
      if (rankingTitle) rankingTitle.innerHTML = `<span>📋</span> Feature Ranking`;

      const features = data.features.slice(0, 15);
      const labels = features.map(f => f.feature);
      const values = features.map(f => f.importance);

      // Magma color gradient: from dark purple through magenta to coral/orange
      const colors = features.map((_, i) => {
        const ratio = i / Math.max(features.length - 1, 1);
        if (ratio < 0.15) return '#581c87';
        if (ratio < 0.3) return '#7e22ce';
        if (ratio < 0.45) return '#a21caf';
        if (ratio < 0.6) return '#be185d';
        if (ratio < 0.75) return '#e11d48';
        if (ratio < 0.9) return '#ea580c';
        return '#f59e0b';
      });

      const ctx = canvas.getContext('2d');
      shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Mean |SHAP Value|',
            data: values,
            backgroundColor: colors,
            borderRadius: 3,
            barThickness: 12
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: ctx => `Mean |SHAP Value|: ${ctx.parsed.x.toFixed(6)}`
              }
            }
          },
          scales: {
            x: {
              title: { display: true, text: 'Mean |SHAP Value|', color: '#94a3b8', font: { size: 10 } },
              ticks: { color: '#94a3b8', font: { size: 9 } },
              grid: { color: 'rgba(255, 255, 255, 0.08)' }
            },
            y: {
              ticks: { color: '#ffffff', font: { size: 9, weight: '500' } },
              grid: { display: false }
            }
          }
        }
      });
    }

    // Populate Ranking Panel (Left)
    if (rankingList && data.features) {
      rankingList.innerHTML = '';
      data.features.forEach((item, index) => {
        const rank = index + 1;
        let badge, color;
        if (rank === 1) { badge = '🥇'; color = '#e91e63'; }
        else if (rank === 2) { badge = '🥈'; color = '#e91e63'; }
        else if (rank === 3) { badge = '🥉'; color = '#e91e63'; }
        else if (rank <= 6) { badge = `#${rank}`; color = '#ff9800'; }
        else { badge = `#${rank}`; color = '#78909c'; }

        const row = document.createElement('div');
        row.style.cssText = `
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 3px 6px;
          border-radius: 4px;
          font-size: 0.78rem;
          transition: background 0.15s ease;
          cursor: pointer;
        `;
        row.title = item.tooltip || item.feature;
        row.onmouseenter = () => { row.style.background = 'rgba(255,255,255,0.06)'; };
        row.onmouseleave = () => { row.style.background = 'transparent'; };

        const leftSpan = document.createElement('div');
        leftSpan.style.cssText = `display: flex; align-items: center; gap: 0.35rem; color: ${color}; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 190px;`;
        leftSpan.innerHTML = `<span style="font-weight: 700; width: 20px; flex-shrink: 0;">${badge}</span> <span style="font-weight: 500;">${escapeHtml(item.feature)}</span>`;

        const valSpan = document.createElement('span');
        valSpan.style.cssText = `font-family: 'Consolas', 'Courier New', monospace; font-size: 0.76rem; color: #b0bec5; font-weight: 600;`;
        valSpan.textContent = item.formatted || (typeof item.importance === 'number' ? item.importance.toFixed(6) : item.importance);

        row.appendChild(leftSpan);
        row.appendChild(valSpan);
        rankingList.appendChild(row);
      });
    }

    // Populate Detailed Insight Panel (Right)
    if (insightText && data.explanation && data.explanation.text) {
      const paras = data.explanation.text.split('\n\n');
      insightText.innerHTML = paras.map(p => `<p style="margin: 0; line-height: 1.6; color: #b0bec5; font-size: 0.82rem;">${escapeHtml(p)}</p>`).join('');
    }

  } catch (err) {
    console.error('Error loading SHAP:', err);
    if (statusText) statusText.textContent = `Failed to compute SHAP: ${err.message}`;
    if (computeBtn) {
      computeBtn.disabled = false;
      computeBtn.innerHTML = `<span>🔍</span> Compute SHAP`;
    }
  }
}

/* ==========================================================================
   PDF Export
   ========================================================================== */
async function exportPdfReport() {
  if (!currentPrediction) {
    showToast('Please execute a prediction first before exporting report', true);
    return;
  }

  showToast('Generating official PDF report...');

  const r0 = {};
  ['R0_UTS', 'R0_YS', 'R0_K1c', 'R0_Elongation', 'R0_HAZ_Width', 'R0_Gauge_Depth', 'R0_Austenite_Content'].forEach(k => {
    r0[k] = parseFloat(document.getElementById(`in_${k}`)?.value) || 0;
  });

  try {
    const res = await fetch('/api/export-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        r0,
        predictions: currentPrediction.predictions,
        model: currentPrediction.model_used
      })
    });

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `M250_Weld_Repair_Report_${Date.now()}.pdf`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    showToast('PDF report downloaded successfully');
  } catch (err) {
    console.error('Error generating PDF:', err);
    showToast('Failed to export PDF report', true);
  }
}

/* ==========================================================================
   Model Training & Live Telemetry Logs
   ========================================================================== */
let trainPollInterval = null;

async function loadTrainingStatus() {
  try {
    const res = await fetch('/api/train/status');
    const data = await res.json();
    if (data.status !== 'success') return;

    // Update status & progress bar
    const statusText = document.getElementById('train-status-text');
    const pctText = document.getElementById('train-pct-text');
    const progressFill = document.getElementById('train-progress-fill');
    const startBtn = document.getElementById('start-train-btn');

    if (statusText) statusText.textContent = `Status: ${data.status_text}`;
    if (pctText) pctText.textContent = `${data.progress}%`;
    if (progressFill) progressFill.style.width = `${data.progress}%`;

    // Console logs
    const consoleEl = document.getElementById('training-console-logs');
    if (consoleEl && data.logs) {
      consoleEl.innerHTML = data.logs.map(line => {
        let cls = 'log-info';
        if (line.includes('[ERROR]') || line.includes('Error')) cls = 'log-err';
        else if (line.includes('[OK]') || line.includes('[SUCCESS]') || line.includes('Accuracy=')) cls = 'log-ok';
        return `<div class="${cls}">${escapeHtml(line)}</div>`;
      }).join('');
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }

    if (data.is_training) {
      if (startBtn) {
        startBtn.disabled = true;
        startBtn.innerHTML = `<span>⏳</span> Training in Progress (${data.progress}%)`;
      }
      if (!trainPollInterval) {
        trainPollInterval = setInterval(loadTrainingStatus, 1500);
      }
    } else {
      if (startBtn) {
        startBtn.disabled = false;
        startBtn.innerHTML = `<span>🚀</span> Train All Models`;
      }
      if (trainPollInterval) {
        clearInterval(trainPollInterval);
        trainPollInterval = null;
      }
    }
  } catch (err) {
    console.error('Error loading training status:', err);
  }
}

async function startTraining() {
  const startBtn = document.getElementById('start-train-btn');
  if (startBtn) {
    startBtn.disabled = true;
    startBtn.innerHTML = `<span>⏳</span> Launching Training...`;
  }
  showToast('Starting ML + DL automated training pipeline...');

  try {
    const res = await fetch('/api/train/start', { method: 'POST' });
    const data = await res.json();
    showToast(data.message || 'Training started');
    loadTrainingStatus();
  } catch (err) {
    console.error('Error starting training:', err);
    showToast('Failed to trigger training pipeline', true);
  }
}

async function loadTrainingComparison(view = 'Overall') {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    if (data.status !== 'success') return;

    const tbody = document.getElementById('training-comparison-tbody');
    const thead = document.getElementById('training-table-head');
    if (!tbody || !thead) return;

    tbody.innerHTML = '';

    if (view === 'Overall') {
      thead.innerHTML = `
        <th>Model</th>
        <th>R² Score</th>
        <th>MAE</th>
        <th>RMSE</th>
        <th>Accuracy %</th>
      `;
      data.leaderboard.forEach(m => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${m.model}</strong></td>
          <td style="color: var(--primary); font-weight: 700;">${m.r2.toFixed(4)}</td>
          <td>${m.mae.toFixed(2)} MPa</td>
          <td>${m.rmse.toFixed(2)} MPa</td>
          <td style="color: #34d399; font-weight: 600;">${m.accuracy.toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
      });
    } else {
      const cleanTarget = view.replace('R2_', '');
      thead.innerHTML = `
        <th>Model</th>
        <th>Target</th>
        <th>R² Score</th>
        <th>MAE</th>
        <th>RMSE</th>
        <th>Accuracy %</th>
      `;
      data.leaderboard.forEach(m => {
        const tm = m.target_metrics?.[view] || {};
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${m.model}</strong></td>
          <td style="color: var(--accent-amber);">${cleanTarget}</td>
          <td style="color: var(--primary); font-weight: 700;">${(tm.R2 !== undefined ? tm.R2.toFixed(4) : '—')}</td>
          <td>${(tm.MAE !== undefined ? tm.MAE.toFixed(3) : '—')}</td>
          <td>${(tm.RMSE !== undefined ? tm.RMSE.toFixed(3) : '—')}</td>
          <td style="color: #34d399; font-weight: 600;">${(tm['Accuracy%'] !== undefined ? tm['Accuracy%'].toFixed(2) + '%' : '—')}</td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (err) {
    console.error('Error loading training comparison:', err);
  }
}

/* ==========================================================================
   Detailed Metrics & Errors Calculations (Tab: Detailed Metrics)
   ========================================================================== */
async function loadDetailedMetrics(selectedModel = null) {
  try {
    const url = selectedModel ? `/api/metrics/detailed?model=${encodeURIComponent(selectedModel)}` : '/api/metrics/detailed';
    const res = await fetch(url);
    const data = await res.json();
    if (data.status !== 'success') return;

    // Populate model dropdown
    const select = document.getElementById('detailed-model-select');
    if (select && data.models_available) {
      select.innerHTML = '';
      data.models_available.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        if (m === data.model) opt.selected = true;
        select.appendChild(opt);
      });
    }

    // Overall summary banner
    const banner = document.getElementById('model-overall-banner');
    if (banner) {
      const ov = data.overall;
      banner.innerHTML = `
        <div class="result-card" style="padding: 1rem;">
          <div class="result-label">Overall R² Score</div>
          <div class="result-val" style="font-size: 1.6rem; color: var(--primary);">${ov.r2}</div>
        </div>
        <div class="result-card" style="padding: 1rem;">
          <div class="result-label">Overall Accuracy</div>
          <div class="result-val" style="font-size: 1.6rem; color: #34d399;">${ov.accuracy}%</div>
        </div>
        <div class="result-card" style="padding: 1rem;">
          <div class="result-label">Mean Abs Error (MAE)</div>
          <div class="result-val" style="font-size: 1.6rem;">${ov.mae} MPa</div>
        </div>
        <div class="result-card" style="padding: 1rem;">
          <div class="result-label">Root Mean Sq Error (RMSE)</div>
          <div class="result-val" style="font-size: 1.6rem;">${ov.rmse} MPa</div>
        </div>
      `;
    }

    // 6 Target Cards (Error Calculations)
    const grid = document.getElementById('detailed-targets-grid');
    if (grid && data.targets_breakdown) {
      grid.innerHTML = '';
      const colors = ['#38bdf8', '#10b981', '#f59e0b', '#8b5cf6', '#f43f5e', '#06b6d4'];
      let idx = 0;

      Object.keys(data.targets_breakdown).forEach(key => {
        const t = data.targets_breakdown[key];
        const color = colors[idx % colors.length];
        idx++;

        const card = document.createElement('div');
        card.className = 'result-card';
        card.style.borderColor = `${color}40`;
        card.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div style="font-family: var(--font-heading); font-size: 1.1rem; font-weight: 700; color: ${color};">
              ${t.property_name}
            </div>
            <span class="badge-tag" style="color: ${color}; border-color: ${color}40;">${key}</span>
          </div>

          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1rem;">
            <div style="background: rgba(255,255,255,0.03); padding: 0.6rem; border-radius: 8px;">
              <div style="font-size: 0.72rem; color: var(--text-muted);">MAE Error</div>
              <div style="font-size: 1.15rem; font-weight: 700;">${t.mae} <span style="font-size: 0.75rem; color: var(--text-dim);">${t.unit}</span></div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 0.6rem; border-radius: 8px;">
              <div style="font-size: 0.72rem; color: var(--text-muted);">RMSE Error</div>
              <div style="font-size: 1.15rem; font-weight: 700;">${t.rmse} <span style="font-size: 0.75rem; color: var(--text-dim);">${t.unit}</span></div>
            </div>
          </div>

          <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.35rem;">
            <span>R² Fit: <strong>${t.r2}</strong></span>
            <span style="color: #34d399; font-weight: 600;">Acc: ${t.accuracy}%</span>
          </div>
          <div class="progress-bar-container" style="height: 6px;">
            <div class="progress-bar-fill" style="width: ${Math.min(Math.max(t.accuracy, 0), 100)}%; background: ${color};"></div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

  } catch (err) {
    console.error('Error loading detailed metrics:', err);
  }
}

/* ==========================================================================
   Data Explorer & Scatter Plots (Tab: Data Explorer)
   ========================================================================== */
let scatterChart = null;
let explorerDataCache = null;

async function loadDataExplorer() {
  try {
    const res = await fetch('/api/data/explorer');
    const data = await res.json();
    if (data.status !== 'success') return;
    explorerDataCache = data;

    // File badge
    const badge = document.getElementById('explorer-file-badge');
    if (badge) badge.innerHTML = `Loaded: <strong>${data.filename}</strong> (${data.total_rows} rows)`;

    // Populate Statistics table
    const statsTbody = document.getElementById('explorer-stats-tbody');
    if (statsTbody && data.statistics) {
      statsTbody.innerHTML = '';
      data.statistics.forEach(s => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${s.feature}</strong></td>
          <td>${s.count}</td>
          <td>${s.mean}</td>
          <td>${s.std}</td>
          <td>${s.min}</td>
          <td>${s.p50}</td>
          <td>${s.max}</td>
        `;
        statsTbody.appendChild(tr);
      });
    }

    // Populate Samples table
    const thead = document.getElementById('explorer-samples-thead');
    const tbody = document.getElementById('explorer-samples-tbody');
    if (thead && tbody && data.columns && data.samples) {
      thead.innerHTML = `<tr>${data.columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
      tbody.innerHTML = '';
      data.samples.slice(0, 30).forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = data.columns.map(c => `<td>${row[c] !== undefined ? row[c] : ''}</td>`).join('');
        tbody.appendChild(tr);
      });
    }

    // Populate scatter selectors
    const selX = document.getElementById('scatter-x-select');
    const selY = document.getElementById('scatter-y-select');
    if (selX && selY && data.columns) {
      selX.innerHTML = data.columns.map(c => `<option value="${c}" ${c.includes('R0_UTS') ? 'selected' : ''}>${c}</option>`).join('');
      selY.innerHTML = data.columns.map(c => `<option value="${c}" ${c.includes('Target_R2_UTS') ? 'selected' : ''}>${c}</option>`).join('');
    }

    // Load initial scatter
    updateScatterPlot();

  } catch (err) {
    console.error('Error loading data explorer:', err);
  }
}

async function updateScatterPlot() {
  const x = document.getElementById('scatter-x-select')?.value || 'R0_UTS_S139_MPa';
  const y = document.getElementById('scatter-y-select')?.value || 'Target_R2_UTS_MPa';

  try {
    const res = await fetch(`/api/data/scatter?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}`);
    const data = await res.json();
    if (data.status !== 'success') return;

    const canvas = document.getElementById('explorerScatterCanvas');
    if (!canvas) return;

    if (scatterChart) scatterChart.destroy();
    scatterChart = new Chart(canvas.getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: [{
          label: `${data.y_col} vs ${data.x_col}`,
          data: data.points,
          backgroundColor: 'rgba(56, 189, 248, 0.65)',
          borderColor: '#38bdf8',
          pointRadius: 4,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8' } },
          tooltip: {
            callbacks: {
              label: ctx => `(${ctx.parsed.x}, ${ctx.parsed.y})`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: data.x_col, color: '#94a3b8' },
            ticks: { color: '#64748b' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            title: { display: true, text: data.y_col, color: '#94a3b8' },
            ticks: { color: '#64748b' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          }
        }
      }
    });

  } catch (err) {
    console.error('Error loading scatter plot:', err);
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function showToast(message, isError = false) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText = `
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 9999;
      display: flex;
      flex-direction: column;
      gap: 10px;
      pointer-events: none;
    `;
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.style.cssText = `
    background: ${isError ? 'rgba(239, 68, 68, 0.92)' : 'rgba(15, 23, 42, 0.92)'};
    color: #fff;
    padding: 12px 20px;
    border-radius: 10px;
    font-family: var(--font-body, 'Inter', sans-serif);
    font-size: 0.88rem;
    font-weight: 500;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    border: 1px solid ${isError ? 'rgba(239, 68, 68, 0.4)' : 'rgba(56, 189, 248, 0.3)'};
    backdrop-filter: blur(12px);
    display: flex;
    align-items: center;
    gap: 10px;
    pointer-events: auto;
    transition: all 0.3s ease;
    opacity: 0;
    transform: translateY(12px);
  `;
  toast.innerHTML = `<span>${isError ? '⚠️' : '✨'}</span> <span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.style.opacity = '1';
    toast.style.transform = 'translateY(0)';
  });

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(-8px)';
    setTimeout(() => {
      if (toast.parentElement) toast.parentElement.removeChild(toast);
    }, 300);
  }, 3500);
}

