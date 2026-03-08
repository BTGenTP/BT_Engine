const MAX_TIME_S = 240;
let progressInterval = null;
let progressStart = 0;
let lastXml = "";

document.addEventListener("DOMContentLoaded", () => {
    checkStatus();
    loadExamples();
});

async function checkStatus() {
    const badge = document.getElementById("model-status");
    const btn = document.getElementById("btn-generate");
    try {
        const res = await fetch("/api/status");
        const data = await res.json();

        // Mode selector: enable/disable from data.modes
        const modes = data.modes || [];
        let firstAvailableId = null;
        modes.forEach((m) => {
            const radio = document.getElementById("mode-" + m.id);
            const reasonEl = document.getElementById("mode-reason-" + m.id);
            const label = radio ? radio.closest(".mode-option") : null;
            if (radio) {
                radio.disabled = !m.available;
                if (m.available && !firstAvailableId) firstAvailableId = m.id;
            }
            if (reasonEl) {
                reasonEl.textContent = m.reason || "";
                reasonEl.title = m.reason || "";
            }
            if (label) {
                if (m.available) label.classList.remove("unavailable"); else label.classList.add("unavailable");
            }
        });
        if (firstAvailableId) {
            const firstRadio = document.getElementById("mode-" + firstAvailableId);
            if (firstRadio && !firstRadio.checked) firstRadio.checked = true;
        }

        if (data.loaded) {
            const provider = data.provider ? `${data.provider}` : "backend";
            badge.textContent = `Modele Nav2 pret (${data.model_key || "—"}, ${provider})`;
            badge.className = "status-badge ready";
            btn.disabled = false;
        } else if (data.configured) {
            badge.textContent = "Adapter Nav2 configure, chargement a la demande";
            badge.className = "status-badge loading";
            btn.disabled = false;
        } else {
            const hasAnyMode = modes.some((m) => m.available);
            badge.textContent = hasAnyMode ? "Au moins un mode disponible" : "Aucun mode disponible (config ou deps)";
            badge.className = hasAnyMode ? "status-badge loading" : "status-badge error";
            btn.disabled = !hasAnyMode;
        }
    } catch {
        badge.textContent = "Serveur injoignable";
        badge.className = "status-badge error";
        btn.disabled = true;
    }
}

async function loadExamples() {
    try {
        const res = await fetch("/api/examples");
        const data = await res.json();
        const container = document.getElementById("example-list");
        container.innerHTML = "";
        data.missions.forEach((mission) => {
            const btn = document.createElement("button");
            btn.className = "example-btn";
            btn.textContent = mission;
            btn.onclick = () => { document.getElementById("mission").value = mission; };
            container.appendChild(btn);
        });
    } catch {
        // ignore
    }
}

function startProgress() {
    progressStart = Date.now();
    const bar = document.getElementById("progress-bar");
    const text = document.getElementById("progress-text");
    const timer = document.getElementById("progress-timer");
    bar.style.width = "0%";
    text.textContent = "Preparation du prompt...";
    progressInterval = setInterval(() => {
        const elapsed = (Date.now() - progressStart) / 1000;
        const pct = Math.min((elapsed / MAX_TIME_S) * 100, 100);
        bar.style.width = pct + "%";
        const elMin = Math.floor(elapsed / 60);
        const elSec = Math.floor(elapsed % 60);
        timer.textContent = `${elMin}:${String(elSec).padStart(2, "0")} / 4:00`;
        if (pct > 50) {
            text.textContent = "Inference en cours...";
        }
        if (elapsed >= MAX_TIME_S) {
            clearInterval(progressInterval);
            text.textContent = "Timeout depasse (4 min)";
        }
    }, 1000);
}

function stopProgress() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    document.getElementById("progress-bar").style.width = "100%";
    document.getElementById("progress-text").textContent = "Termine !";
}

async function generate() {
    const mission = document.getElementById("mission").value.trim();
    if (!mission) return;

    const btn = document.getElementById("btn-generate");
    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("result").classList.add("hidden");
    btn.disabled = true;
    startProgress();

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), MAX_TIME_S * 1000);
    try {
        const selectedMode = document.querySelector('input[name="gen-mode"]:checked');
        const mode = selectedMode && !selectedMode.disabled ? selectedMode.value : null;

        const res = await fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                mission,
                mode: mode || undefined,
                constrained: document.getElementById("use-constraint").checked ? "regex" : "off",
                max_new_tokens: parseInt(document.getElementById("max-new-tokens").value, 10) || 1024,
                temperature: parseFloat(document.getElementById("temperature").value || "0"),
                write_run: document.getElementById("write-run").checked,
            }),
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        const data = await res.json();
        if (!res.ok) {
            alert(data.error || "Erreur serveur");
            return;
        }
        stopProgress();
        displayResult(data);
    } catch (e) {
        clearTimeout(timeoutId);
        alert(e.name === "AbortError" ? "Timeout depasse." : `Erreur: ${e.message}`);
    } finally {
        document.getElementById("loading").classList.add("hidden");
        btn.disabled = false;
    }
}

async function validateOnly() {
    const xml = document.getElementById("xml-input").value.trim();
    if (!xml) return;
    const res = await fetch("/api/validate/xml", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ xml }),
    });
    const data = await res.json();
    displayResult({ ...data, generation_time_s: null, run_dir: null });
}

function displayResult(data) {
    document.getElementById("result").classList.remove("hidden");
    document.getElementById("xml-output").innerHTML = data.xml ? highlightXml(data.xml) : escapeHtml("(aucun XML genere)");
    lastXml = data.xml || "";
    document.getElementById("gen-time").textContent = data.generation_time_s != null ? `Genere en ${data.generation_time_s}s` : "";

    const runDir = document.getElementById("run-dir");
    if (data.run_dir) {
        runDir.classList.remove("hidden");
        runDir.textContent = `Run: ${data.run_dir}`;
    } else {
        runDir.classList.add("hidden");
        runDir.textContent = "";
    }

    const score = data.score != null ? data.score : (data.valid ? 1.0 : 0.0);
    const scoreBar = document.getElementById("score-bar");
    scoreBar.style.width = (score * 100) + "%";
    scoreBar.style.background = score > 0.8 ? "var(--green)" : score > 0.5 ? "var(--yellow)" : "var(--red)";
    document.getElementById("score-value").textContent = score.toFixed(2);
    const badge = document.getElementById("valid-badge");
    badge.textContent = data.valid ? "VALIDE" : "INVALIDE";
    badge.className = "status-badge " + (data.valid ? "valid" : "invalid");

    renderList("warnings-list", data.warnings || [], "warning-item");
    renderList("errors-list", data.errors || [], "error-item");
}

function currentXmlForRos2() {
    const xmlInput = document.getElementById("xml-input");
    const xml = (xmlInput && xmlInput.value ? xmlInput.value.trim() : "");
    return xml || (lastXml || "");
}

function showRos2Output(obj) {
    const pre = document.getElementById("ros2-output-pre");
    const out = document.getElementById("ros2-output");
    if (!pre || !out) return;
    pre.classList.remove("hidden");
    out.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function getRos2Form() {
    return {
        filename: (document.getElementById("ros2-filename")?.value || "").trim() || null,
        initial_pose: (document.getElementById("ros2-initial-pose")?.value || "").trim() || "0.0,0.0,0.0",
        goal_name: (document.getElementById("ros2-goal-name")?.value || "").trim() || null,
        goal_pose: (document.getElementById("ros2-goal-pose")?.value || "").trim() || null,
        start_stack_if_needed: !!document.getElementById("ros2-start-stack")?.checked,
        restart_navigation: !!document.getElementById("ros2-restart-nav")?.checked,
        allow_invalid: !!document.getElementById("ros2-allow-invalid")?.checked,
    };
}

async function transferToRos2() {
    const xml = currentXmlForRos2();
    if (!xml) {
        alert("Aucun XML a transferer (genere ou colle).");
        return;
    }
    const form = getRos2Form();
    const res = await fetch("/api/transfer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ xml, filename: form.filename }),
    });
    const data = await res.json();
    if (!res.ok) {
        showRos2Output(data);
        alert(data.error || "Erreur transfert ROS2");
        return;
    }
    showRos2Output(data);
}

async function executeOnRos2() {
    const xml = currentXmlForRos2();
    const form = getRos2Form();
    if (form.goal_name && form.goal_pose) {
        const msg = "Fournir soit goal_name soit goal_pose, pas les deux.";
        showRos2Output({ error: msg });
        alert(msg);
        return;
    }
    if (!xml && !form.filename) {
        const msg = "Fournir un XML ou un filename.";
        showRos2Output({ error: msg });
        alert(msg);
        return;
    }
    const res = await fetch("/api/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            xml: xml || null,
            filename: form.filename,
            goal_name: form.goal_name,
            goal_pose: form.goal_pose,
            initial_pose: form.initial_pose,
            allow_invalid: form.allow_invalid,
            start_stack_if_needed: form.start_stack_if_needed,
            restart_navigation: form.restart_navigation,
        }),
    });
    const data = await res.json();
    showRos2Output(data);
}

function renderList(id, items, className) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = "";
    if (!items || items.length === 0) {
        el.classList.add("hidden");
        return;
    }
    el.classList.remove("hidden");
    items.forEach((t) => {
        const div = document.createElement("div");
        div.className = className;
        div.textContent = t;
        el.appendChild(div);
    });
}

function escapeHtml(str) {
    return (str || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

function highlightXml(xml) {
    const s = escapeHtml(xml);
    return s.replace(/(&lt;\/?)([A-Za-z0-9_:-]+)([^&]*?)(\/?&gt;)/g, (_m, p1, tag, rest, p4) => {
        const attrs = rest.replace(/([A-Za-z0-9_:-]+)=(&quot;[^&]*?&quot;)/g, (_m2, a, v) => ` <span class="attr">${a}</span>=<span class="val">${v}</span>`);
        return `${p1}<span class="tag">${tag}</span>${attrs}${p4}`;
    });
}

function formatJsonBlock(text) {
    try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; }
}

