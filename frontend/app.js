// RecoverAI Interactive Dashboard Application Logic

const PRESETS = {
  transient: {
    recovery_case_id: "CASE-DEMO-TRANSIENT-01",
    amount_at_risk: 2500,
    failure_reason: "BANK_TIMEOUT",
    payment_method: "UPI",
    customer_segment: "NORMAL",
    retry_count: 0,
    cooldown_eligible: true,
    human_review_required: false,
    simulate_failure: false,
  },
  funds: {
    recovery_case_id: "CASE-DEMO-FUNDS-02",
    amount_at_risk: 1800,
    failure_reason: "INSUFFICIENT_FUNDS",
    payment_method: "CARD",
    customer_segment: "NORMAL",
    retry_count: 0,
    cooldown_eligible: true,
    human_review_required: false,
    simulate_failure: false,
  },
  highvalue: {
    recovery_case_id: "CASE-DEMO-HIGHVAL-03",
    amount_at_risk: 75000,
    failure_reason: "PAYMENT_METHOD_DECLINED",
    payment_method: "CARD",
    customer_segment: "VIP",
    retry_count: 0,
    cooldown_eligible: true,
    human_review_required: false,
    simulate_failure: false,
  },
  retrylimit: {
    recovery_case_id: "CASE-DEMO-RETRYMAX-04",
    amount_at_risk: 3200,
    failure_reason: "GATEWAY_TIMEOUT",
    payment_method: "NETBANKING",
    customer_segment: "NORMAL",
    retry_count: 2,
    cooldown_eligible: true,
    human_review_required: false,
    simulate_failure: false,
  },
};

document.addEventListener("DOMContentLoaded", () => {
  initPresets();
  initFormActions();
  fetchKpis();
});

function initPresets() {
  const buttons = document.querySelectorAll(".btn-preset");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const presetKey = btn.dataset.preset;
      const data = PRESETS[presetKey];
      if (data) {
        document.getElementById("recovery_case_id").value = data.recovery_case_id;
        document.getElementById("amount_at_risk").value = data.amount_at_risk;
        document.getElementById("failure_reason").value = data.failure_reason;
        document.getElementById("payment_method").value = data.payment_method;
        document.getElementById("customer_segment").value = data.customer_segment;
        document.getElementById("retry_count").value = data.retry_count;
        document.getElementById("cooldown_eligible").checked = data.cooldown_eligible;
        document.getElementById("human_review_required").checked = data.human_review_required;
        document.getElementById("simulate_failure").checked = data.simulate_failure;
      }
    });
  });
}

function getFormData() {
  return {
    recovery_case_id: document.getElementById("recovery_case_id").value.trim(),
    amount_at_risk: parseFloat(document.getElementById("amount_at_risk").value) || 1000,
    failure_reason: document.getElementById("failure_reason").value,
    payment_method: document.getElementById("payment_method").value,
    customer_segment: document.getElementById("customer_segment").value,
    retry_count: parseInt(document.getElementById("retry_count").value, 10) || 0,
    cooldown_eligible: document.getElementById("cooldown_eligible").checked,
    human_review_required: document.getElementById("human_review_required").checked,
    simulate_failure: document.getElementById("simulate_failure").checked,
  };
}

function initFormActions() {
  const btnRecommend = document.getElementById("btn-recommend");
  const btnExecute = document.getElementById("btn-execute");

  btnRecommend.addEventListener("click", async () => {
    const formData = getFormData();
    await handleRecoveryAction(formData, false);
  });

  btnExecute.addEventListener("click", async () => {
    const formData = getFormData();
    await handleRecoveryAction(formData, true);
  });
}

async function fetchKpis() {
  try {
    const res = await fetch("/api/v1/metrics/summary");
    if (!res.ok) return;
    const data = await res.json();
    const kpis = data.kpis;

    if (kpis) {
      document.getElementById("kpi-amount-risk").textContent = `₹${kpis.total_amount_at_risk.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
      document.getElementById("kpi-recovered-revenue").textContent = `₹${kpis.ml_recovered_revenue.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
      document.getElementById("kpi-recovery-rate").textContent = `${(kpis.ml_recovery_rate * 100).toFixed(2)}%`;
      document.getElementById("kpi-revenue-lift").textContent = `+₹${kpis.absolute_revenue_lift.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
      document.getElementById("kpi-baseline-revenue").textContent = `₹${kpis.baseline_recovered_revenue.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
    }
  } catch (err) {
    console.warn("Could not fetch KPI summary:", err);
  }
}

async function handleRecoveryAction(formData, execute = false) {
  const badge = document.getElementById("decision-status-badge");
  badge.textContent = execute ? "Executing..." : "Evaluating...";
  badge.className = "badge badge-testmode";

  try {
    const payload = {
      case_data: {
        recovery_case_id: formData.recovery_case_id,
        amount_at_risk: formData.amount_at_risk,
        failure_reason: formData.failure_reason,
        payment_method: formData.payment_method,
        customer_segment: formData.customer_segment,
        retry_count: formData.retry_count,
        cooldown_eligible: formData.cooldown_eligible,
        human_review_required: formData.human_review_required,
      },
      execute: execute,
      idempotency_key: `idem_${formData.recovery_case_id}_${Date.now()}`,
      simulate_failure: formData.simulate_failure,
    };

    const res = await fetch("/api/v1/recovery/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`API Error: ${err.detail || "Unknown error"}`);
      badge.textContent = "Error";
      badge.className = "badge badge-idle";
      return;
    }

    const data = await res.json();
    renderResults(data);

    badge.textContent = execute ? "Executed" : "Recommended";
    badge.className = "badge badge-synthetic";
  } catch (err) {
    console.error("Request failed:", err);
    alert(`Request failed: ${err.message}`);
    badge.textContent = "Error";
    badge.className = "badge badge-idle";
  }
}

function renderResults(data) {
  document.getElementById("empty-state").style.display = "none";
  document.getElementById("results-view").style.display = "flex";

  const selectedAction = data.selected_action;
  const policy = data.policy_decision;
  const modelRec = data.model_recommendation;
  const explanation = data.explanation;
  const execution = data.execution;

  // Selected Action Banner
  document.getElementById("res-selected-action").textContent = selectedAction;
  const policyTag = document.getElementById("res-policy-status");
  if (policy.fallback_occurred) {
    policyTag.textContent = "Policy Rerouted";
    policyTag.className = "policy-tag rerouted";
  } else {
    policyTag.textContent = "Policy Approved";
    policyTag.className = "policy-tag approved";
  }

  const selectedScore = modelRec.candidate_rankings.find((r) => r.action === selectedAction);
  const expectedVal = selectedScore ? selectedScore.expected_recovery_value : 0;
  document.getElementById("res-expected-val").textContent = `E[Val]: ₹${expectedVal.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;

  // Policy Engine Guardrails
  const reasonsList = document.getElementById("policy-reasons-list");
  reasonsList.innerHTML = policy.policy_reasons.map((r) => `<p>${r}</p>`).join("");

  const blockedList = document.getElementById("blocked-actions-list");
  if (policy.blocked_actions && Object.keys(policy.blocked_actions).length > 0) {
    blockedList.innerHTML = Object.entries(policy.blocked_actions)
      .map(([act, reason]) => `<div class="blocked-tag">🚫 ${act}: ${reason}</div>`)
      .join("");
  } else {
    blockedList.innerHTML = "<div class='policy-reasons'>No actions blocked.</div>";
  }

  // 5-Action Candidate Ranking
  const rankingBars = document.getElementById("ranking-bars");
  rankingBars.innerHTML = modelRec.candidate_rankings
    .map((cand) => {
      const isSelected = cand.action === selectedAction;
      const isBlocked = policy.blocked_actions && policy.blocked_actions[cand.action];
      const fillClass = isSelected ? "selected" : isBlocked ? "blocked" : "";
      const percent = (cand.predicted_probability * 100).toFixed(1);

      return `
        <div class="ranking-row">
          <div class="ranking-info">
            <span>${cand.rank}. ${cand.action} ${isSelected ? "✓" : isBlocked ? "🚫" : ""}</span>
            <span class="mono">${percent}% (₹${cand.expected_recovery_value.toLocaleString('en-IN', {minimumFractionDigits: 2})})</span>
          </div>
          <div class="ranking-bar-bg">
            <div class="ranking-bar-fill ${fillClass}" style="width: ${percent}%;"></div>
          </div>
        </div>
      `;
    })
    .join("");

  // Decision Explainability
  const explainList = document.getElementById("explain-reasons");
  explainList.innerHTML = explanation.decision_reasons.map((r) => `<li>${r}</li>`).join("");

  // Execution Result
  const statusBadge = document.getElementById("exec-status");
  statusBadge.textContent = execution.status;
  statusBadge.className = `exec-badge ${execution.status.toLowerCase()}`;
  document.getElementById("exec-ref").textContent = execution.reference_id || "N/A";
  document.getElementById("exec-msg").textContent = execution.message;

  const linkContainer = document.getElementById("exec-link-container");
  const linkUrl = document.getElementById("exec-link-url");
  if (execution.link_url) {
    linkContainer.style.display = "block";
    linkUrl.href = execution.link_url;
  } else {
    linkContainer.style.display = "none";
  }

  // Fetch Live Audit Trail
  fetchAuditTrail(data.recovery_case_id);
}

async function fetchAuditTrail(caseId) {
  try {
    const res = await fetch(`/api/v1/recovery/${caseId}/audit`);
    if (!res.ok) return;
    const auditData = await res.json();
    const timeline = document.getElementById("audit-timeline");

    timeline.innerHTML = auditData.audit_trail
      .map((ev) => {
        const time = ev.timestamp ? ev.timestamp.split(" ")[1] || ev.timestamp : "Just now";
        return `
          <div class="timeline-item">
            <span class="timeline-time">${time}</span>
            <span class="timeline-event"><strong>${ev.event_type}</strong></span>
          </div>
        `;
      })
      .join("");
  } catch (err) {
    console.warn("Could not fetch audit trail:", err);
  }
}
