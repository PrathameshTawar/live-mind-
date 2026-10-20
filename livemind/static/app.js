/**
 * LiveMind — Apple HIG Frontend Controller
 * Connects to LiveMind FastAPI Endpoints:
 * - /research
 * - /connectors/jira/sync
 * - /connectors/notion/sync
 */

(function () {
  "use strict";

  // 1. Theme Controller
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeToggleIcon = document.getElementById("themeToggleIcon");
  const htmlRoot = document.documentElement;

  function applyTheme(theme) {
    htmlRoot.setAttribute("data-theme", theme);
    localStorage.setItem("livemind-theme", theme);
    if (themeToggleIcon) {
      themeToggleIcon.textContent = theme === "dark" ? "☀️" : "🌙";
    }
  }

  const savedTheme = localStorage.getItem("livemind-theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  applyTheme(savedTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const current = htmlRoot.getAttribute("data-theme") || "light";
      applyTheme(current === "dark" ? "light" : "dark");
    });
  }

  // 2. Subtle Matrix Canvas
  const canvas = document.getElementById("matrixCanvas");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    let width = 0, height = 0;
    const gridSize = 42;
    let particles = [];

    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      width = canvas.width = rect.width;
      height = canvas.height = rect.height;
      particles = Array.from({ length: 24 }, () => ({
        x: Math.floor(Math.random() * (width / gridSize)) * gridSize,
        y: Math.floor(Math.random() * (height / gridSize)) * gridSize,
        speed: 1.5 + Math.random() * 2,
        radius: 2 + Math.random() * 2,
        opacity: 0.3 + Math.random() * 0.5,
      }));
    }

    function render() {
      const isDark = htmlRoot.getAttribute("data-theme") === "dark";
      ctx.clearRect(0, 0, width, height);

      // Grid
      ctx.lineWidth = 1;
      ctx.strokeStyle = isDark ? "rgba(196, 181, 253, 0.08)" : "rgba(139, 92, 246, 0.08)";
      ctx.beginPath();
      for (let x = 0; x <= width; x += gridSize) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
      for (let y = 0; y <= height; y += gridSize) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
      ctx.stroke();

      // Pulses
      particles.forEach((p) => {
        p.y += p.speed;
        if (p.y > height) { p.y = 0; p.x = Math.floor(Math.random() * (width / gridSize)) * gridSize; }
        ctx.fillStyle = isDark ? "rgba(196, 181, 253, 0.85)" : "rgba(139, 92, 246, 0.75)";
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fill();
      });
      requestAnimationFrame(render);
    }

    window.addEventListener("resize", resize);
    resize();
    render();
  }

  // 3. Multi-Hop Research Execution
  const researchBtn = document.getElementById("researchBtn");
  const queryInput = document.getElementById("queryInput");
  const terminalLogs = document.getElementById("terminalLogs");
  const liveStatusText = document.getElementById("liveStatusText");
  const statusBeacon = document.getElementById("statusBeacon");

  function appendLog(text, type = "info") {
    const p = document.createElement("p");
    p.className = `log-line log-${type}`;
    p.textContent = text;
    terminalLogs.appendChild(p);
    terminalLogs.scrollTop = terminalLogs.scrollHeight;
  }

  async function executeResearch(question) {
    terminalLogs.innerHTML = "";
    researchBtn.disabled = true;
    researchBtn.textContent = "Executing Multi-Hop Research...";
    liveStatusText.textContent = "RESEARCHING...";
    statusBeacon.style.background = "#ffbd2e";
    statusBeacon.style.boxShadow = "0 0 10px #ffbd2e";

    appendLog(`[LiveMind Agent] Inquiry Dispatched: "${question}"`, "step");

    try {
      // First ensure sample ticket is indexed if asking about PAY-1842
      if (question.includes("PAY-1842")) {
        await fetch("/connectors/jira/sync", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ticket_key: "PAY-1842",
            summary: "Payment webhook failure in auth service",
            status: "Resolved",
            priority: "High",
            description: "Root cause was NullPointerException in deserializer. Fixed in v2.4.1.",
          }),
        });
      }

      const response = await fetch("/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: "apple-ui-session" }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const result = await response.json();

      appendLog(`✓ Research Task Completed: ${result.task_id.substring(0, 8)}...`, "success");

      // Print Plan
      if (result.plan && result.plan.length > 0) {
        appendLog(`\n[Multi-Hop Trajectory]:`, "step");
        result.plan.forEach((step) => appendLog(`  • ${step}`, "info"));
      }

      // Print Synthesized Findings
      appendLog(`\n[Synthesized Answer]:`, "step");
      appendLog(result.final_answer, "success");

      // Print Citations
      if (result.citations && result.citations.length > 0) {
        appendLog(`\n[Verified Citations (${result.citations.length})]:`, "step");
        result.citations.forEach((c, idx) => {
          appendLog(`  [${idx + 1}] ${c.source_title} (${c.source_type.toUpperCase()})`, "warn");
          appendLog(`      URL: ${c.source_url}`, "info");
        });
      }

      liveStatusText.textContent = "SYNTHESIS VERIFIED";
      statusBeacon.style.background = "#34D399";
      statusBeacon.style.boxShadow = "0 0 10px #34D399";
    } catch (err) {
      appendLog(`\n✗ Research Error: ${err.message}`, "warn");
      statusBeacon.style.background = "#ef4444";
    } finally {
      researchBtn.disabled = false;
      researchBtn.textContent = "Execute Multi-Hop Research 🧠";
    }
  }

  if (researchBtn) {
    researchBtn.addEventListener("click", () => {
      const val = queryInput.value.trim() || "What was the resolution of PAY-1842?";
      executeResearch(val);
    });
  }

  // Presets
  document.querySelectorAll(".preset-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      queryInput.value = pill.getAttribute("data-preset");
    });
  });

  // 4. Modals
  const privacyModal = document.getElementById("privacyModal");
  const termsModal = document.getElementById("termsModal");
  const openPrivacyBtn = document.getElementById("openPrivacyBtn");
  const openTermsBtn = document.getElementById("openTermsBtn");

  function openM(m) { if (m) m.classList.add("active"); }
  function closeM(m) { if (m) m.classList.remove("active"); }

  if (openPrivacyBtn) openPrivacyBtn.addEventListener("click", () => openM(privacyModal));
  if (openTermsBtn) openTermsBtn.addEventListener("click", () => openM(termsModal));

  document.querySelectorAll(".modal-close-btn, #acceptPrivacyBtn, #acceptTermsBtn").forEach((btn) => {
    btn.addEventListener("click", () => {
      closeM(privacyModal);
      closeM(termsModal);
    });
  });

  [privacyModal, termsModal].forEach((m) => {
    if (m) m.addEventListener("click", (e) => { if (e.target === m) closeM(m); });
  });

  console.log("LiveMind Apple HIG Frontend Connected.");
})();
