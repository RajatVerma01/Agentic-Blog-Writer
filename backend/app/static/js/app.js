/**
 * app.js — Agentic Blog Writer Frontend
 *
 * Architecture: one state object drives all UI updates.
 * - api.*      → all fetch calls (one place, no duplication)
 * - ui.*       → all DOM mutations (one place, no duplication)
 * - app.*      → orchestration logic (polling loop, event handlers)
 */

'use strict';

// ── DOM refs (cached once, never re-queried) ─────────────────────────────
const DOM = {
  topicInput:   document.getElementById('topic-input'),
  charCounter:  document.getElementById('char-counter'),
  btnGenerate:  document.getElementById('btn-generate'),
  progressCard: document.getElementById('progress-card'),
  progressTitle:document.getElementById('progress-title'),
  statusBadge:  document.getElementById('status-badge'),
  agentsTrack:  document.getElementById('agents-track'),
  errorMsg:     document.getElementById('error-msg'),
  resultCard:   document.getElementById('result-card'),
  totalScore:   document.getElementById('total-score'),
  dimGrid:      document.getElementById('dim-grid'),
  blogContent:  document.getElementById('blog-content'),
  btnCopy:      document.getElementById('btn-copy'),
};

// Agent order used to advance the progress stepper
const AGENT_ORDER = ['researcher', 'planner', 'writer', 'evaluator'];

const AGENT_LABELS = {
  researcher: 'Researching the web…',
  planner:    'Planning the outline…',
  writer:     'Writing the draft…',
  evaluator:  'Evaluating quality…',
};

// Dimension display labels
const DIM_LABELS = {
  grammar_clarity:  'Grammar',
  factual_accuracy: 'Accuracy',
  citation_quality: 'Citations',
  structure_flow:   'Structure',
  seo_optimization: 'SEO',
};

// ── API layer ────────────────────────────────────────────────────────────
const api = {
  async generate(topic) {
    const res = await fetch('/api/v1/blog/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.message || `HTTP ${res.status}`);
    }
    return res.json();
  },

  async status(jobId) {
    const res = await fetch(`/api/v1/blog/status/${jobId}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.message || `HTTP ${res.status}`);
    }
    return res.json();
  },

  async result(jobId) {
    const res = await fetch(`/api/v1/blog/result/${jobId}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.message || `HTTP ${res.status}`);
    }
    return res.json();
  },
};

// ── UI layer ─────────────────────────────────────────────────────────────
const ui = {
  setGenerating(isGenerating) {
    DOM.btnGenerate.disabled = isGenerating;
    DOM.btnGenerate.innerHTML = isGenerating
      ? '<div class="spinner"></div> Generating…'
      : '✦ Generate Blog';
    DOM.progressCard.classList.toggle('visible', isGenerating);
    if (!isGenerating) DOM.resultCard.classList.remove('visible');
    DOM.errorMsg.style.display = 'none';
  },

  setStatusBadge(status) {
    const labels = { queued: 'Queued', running: 'Running', completed: 'Done ✓', failed: 'Failed' };
    DOM.statusBadge.textContent = labels[status] || status;
    DOM.statusBadge.className = `status-badge ${status}`;
  },

  setProgressTitle(agent) {
    if (DOM.progressTitle) {
      DOM.progressTitle.textContent = AGENT_LABELS[agent] || 'Generating your blog…';
    }
  },

  // Advance the stepper: agents before `agentName` are DONE, `agentName` is ACTIVE
  setActiveAgent(agentName) {
    const idx = AGENT_ORDER.indexOf(agentName);
    if (idx === -1) return;
    AGENT_ORDER.forEach((name, i) => {
      const el = document.getElementById(`step-${name}`);
      if (!el) return;
      el.classList.remove('done', 'active');
      if (i < idx)  el.classList.add('done');
      if (i === idx) el.classList.add('active');
    });
    ui.setProgressTitle(agentName);
  },

  markAllDone() {
    AGENT_ORDER.forEach(name => {
      const el = document.getElementById(`step-${name}`);
      if (el) { el.classList.remove('active'); el.classList.add('done'); }
    });
    if (DOM.progressTitle) DOM.progressTitle.textContent = 'All agents complete!';
  },

  showError(message) {
    DOM.progressCard.classList.add('visible');
    DOM.errorMsg.textContent = `⚠ ${message}`;
    DOM.errorMsg.style.display = 'block';
    DOM.btnGenerate.disabled = false;
    DOM.btnGenerate.innerHTML = '✦ Generate Blog';
  },

  renderResult(data) {
    try {
      const eval_ = data.evaluation_summary || {};
      const score  = eval_.score ?? '—';
      DOM.totalScore.innerHTML = `${typeof score === 'number' ? score.toFixed(1) : score} <span>/ 10</span>`;

      const dims = eval_.scores_by_dimension || {};
      DOM.dimGrid.innerHTML = Object.entries(dims).map(([key, val]) => `
        <div class="dim-item">
          <div class="dim-name">${DIM_LABELS[key] || key}</div>
          <div class="dim-bar-wrap">
            <div class="dim-bar" style="width:${(typeof val === 'number' ? val : 0) / 10 * 100}%"></div>
          </div>
          <div class="dim-score">${typeof val === 'number' ? val.toFixed(1) : '0.0'}</div>
        </div>
      `).join('');

      const rawBlog = data.final_blog || '';
      DOM.blogContent.innerHTML = typeof marked !== 'undefined' ? marked.parse(rawBlog) : rawBlog;
      DOM.btnCopy.dataset.raw = rawBlog;

      // Hide progress card, show result card
      DOM.progressCard.classList.remove('visible');
      DOM.resultCard.classList.add('visible');
    } catch (err) {
      console.error("renderResult error:", err);
      ui.showError(`Failed to render result: ${err.message}`);
    }
  },
};

// ── App orchestration ────────────────────────────────────────────────────
const app = {
  pollInterval:   null,
  lastAgentIndex: -1,   // tracks the last agent we've visually shown
  jobId:          null,

  async generate() {
    const topic = DOM.topicInput.value.trim();
    if (!topic || topic.length < 10) return;

    // Reset state
    app.lastAgentIndex = -1;
    app.jobId = null;
    AGENT_ORDER.forEach(name => {
      const el = document.getElementById(`step-${name}`);
      if (el) el.classList.remove('done', 'active');
    });

    ui.setGenerating(true);
    ui.setStatusBadge('queued');
    ui.setActiveAgent('researcher');
    app.lastAgentIndex = 0;

    try {
      const { job_id } = await api.generate(topic);
      app.jobId = job_id;
      app.startPolling(job_id);
    } catch (err) {
      ui.setGenerating(false);
      ui.showError(`Failed to start: ${err.message}`);
    }
  },

  startPolling(jobId) {
    // Poll every 1.2 seconds — fast enough to catch each ~2s agent step
    app.pollInterval = setInterval(() => app.poll(jobId), 1200);
  },

  async poll(jobId) {
    try {
      const status = await api.status(jobId);
      ui.setStatusBadge(status.status);

      // Advance stepper to the current agent if it's ahead of what we're showing
      const agent = status.current_agent;
      if (agent && agent !== 'none' && agent !== 'done') {
        const idx = AGENT_ORDER.indexOf(agent);
        if (idx > app.lastAgentIndex) {
          ui.setActiveAgent(agent);
          app.lastAgentIndex = idx;
        }
      }

      if (status.status === 'completed') {
        app.stopPolling();
        await app.animateRemainingSteps();   // smoothly light up any skipped steps
        const result = await api.result(jobId);
        ui.setGenerating(false);
        ui.renderResult(result);

      } else if (status.status === 'failed') {
        app.stopPolling();
        ui.setGenerating(false);
        ui.showError(status.error || 'Blog generation failed. Please try again.');
      }

    } catch (err) {
      app.stopPolling();
      ui.setGenerating(false);
      ui.showError(`Connection error: ${err.message}`);
    }
  },

  // Animate through any agents that weren't seen during polling
  // (pipeline can run faster than polling interval)
  async animateRemainingSteps() {
    const startFrom = app.lastAgentIndex + 1;
    for (let i = startFrom; i < AGENT_ORDER.length; i++) {
      ui.setActiveAgent(AGENT_ORDER[i]);
      await new Promise(r => setTimeout(r, 700));
    }
    ui.markAllDone();
    await new Promise(r => setTimeout(r, 500));
  },

  stopPolling() {
    if (app.pollInterval) {
      clearInterval(app.pollInterval);
      app.pollInterval = null;
    }
  },

  async copyMarkdown() {
    const raw = DOM.btnCopy.dataset.raw || '';
    try {
      await navigator.clipboard.writeText(raw);
      DOM.btnCopy.innerHTML = '✓ Copied!';
      setTimeout(() => { DOM.btnCopy.innerHTML = '📋 Copy Markdown'; }, 2000);
    } catch {
      DOM.btnCopy.innerHTML = 'Copy failed';
    }
  },
};

// ── Event listeners ───────────────────────────────────────────────────────
DOM.topicInput.addEventListener('input', () => {
  const len = DOM.topicInput.value.length;
  DOM.charCounter.textContent = `${len} / 200`;
  DOM.charCounter.className = 'char-counter' + (len > 180 ? ' warning' : '') + (len >= 200 ? ' over' : '');
  DOM.btnGenerate.disabled = len < 10;
});

DOM.btnGenerate.addEventListener('click', () => app.generate());
DOM.topicInput.addEventListener('keydown', e => { if (e.key === 'Enter' && e.ctrlKey) app.generate(); });
DOM.btnCopy.addEventListener('click', () => app.copyMarkdown());
