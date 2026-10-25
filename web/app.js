// RiverWatch status page: live status, flood replay and evaluation. No dependencies.
const $ = (id) => document.getElementById(id);
const LEVEL_TEXT = { clear: "Crossing open", caution: "Caution", close_crossing: "Crossing closed", danger: "Danger: SUVs and trucks at risk" };
const LEVELS = [
  { level: "caution", stage: 4.5, color: "#b7791f" },
  { level: "close_crossing", stage: 5.0, color: "#c2511a" },
  { level: "danger", stage: 6.0, color: "#b00020" },
];
// times are shown in the crossing's local time (Texas), whatever the viewer's zone
const TZ = "America/Chicago";
const fmtTime = (iso) => new Date(iso).toLocaleString("en-US", { timeZone: TZ, month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
const ft = (v) => (v == null ? "–" : `${Number(v).toFixed(2)} ft`);

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function setBadge(el, level) {
  el.className = `alert-badge ${level}`;
  el.textContent = LEVEL_TEXT[level] || level;
}

// ---------------------------------------------------------------- charts (SVG)
function chart(el, { series, t0, t1, yMin, yMax, markers = [], cursor = null, thresholds = true }) {
  const W = el.clientWidth || 800, H = el.clientHeight || 220, L = 40, R = 10, T = 10, B = 24;
  const x = (t) => L + ((t - t0) / (t1 - t0)) * (W - L - R);
  const y = (v) => T + (1 - (v - yMin) / (yMax - yMin)) * (H - T - B);
  let s = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  for (let v = Math.ceil(yMin); v <= yMax; v += yMax - yMin > 8 ? 2 : 1) {
    s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="currentColor" opacity=".08"/>`;
    s += `<text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  }
  if (thresholds) for (const l of LEVELS) {
    if (l.stage < yMin || l.stage > yMax) continue;
    s += `<line x1="${L}" x2="${W - R}" y1="${y(l.stage)}" y2="${y(l.stage)}" stroke="${l.color}" stroke-dasharray="4 4" opacity=".7"/>`;
    s += `<text x="${W - R - 4}" y="${y(l.stage) - 4}" text-anchor="end" style="fill:${l.color}">${LEVEL_TEXT[l.level]}</text>`;
  }
  const ticks = 6;
  for (let i = 0; i <= ticks; i++) {
    const t = t0 + ((t1 - t0) * i) / ticks;
    s += `<text x="${x(t)}" y="${H - 6}" text-anchor="middle">${new Date(t * 1000).toLocaleTimeString("en-US", { timeZone: TZ, hour: "2-digit", minute: "2-digit" })}</text>`;
  }
  for (const m of markers) s += `<line x1="${x(m.t)}" x2="${x(m.t)}" y1="${T}" y2="${H - B}" stroke="${m.color}" stroke-width="1.5" opacity=".8"><title>${m.label}</title></line>`;
  for (const se of series) {
    const pts = se.points.filter((p) => p[0] >= t0 && p[0] <= t1 && p[1] != null);
    if (se.dots) for (const p of pts) s += `<circle cx="${x(p[0])}" cy="${y(Math.min(Math.max(p[1], yMin), yMax))}" r="3.2" fill="${se.color}"/>`;
    else if (pts.length) s += `<polyline fill="none" stroke="${se.color}" stroke-width="2" points="${pts.map((p) => `${x(p[0])},${y(p[1])}`).join(" ")}"/>`;
  }
  if (cursor != null) s += `<line x1="${x(cursor)}" x2="${x(cursor)}" y1="${T}" y2="${H - B}" stroke="currentColor" opacity=".5"/>`;
  const legend = series.map((se) => `<tspan style="fill:${se.color}">● ${se.name}</tspan>`).join("   ");
  s += `<text x="${L + 6}" y="${T + 12}">${legend}</text></svg>`;
  el.innerHTML = s;
}

// ---------------------------------------------------------------- trace rendering
function traceItems(steps) {
  const out = [];
  for (const st of steps || []) {
    if (st.step === "tool") {
      const o = st.output || {};
      let txt = "";
      if (st.tool === "measure_camera") txt = o.usable ? `stage ${ft(o.stage_ft)}${o.below_range ? " (no flowing water)" : ""}${o.above_range ? " (above calibrated range)" : ""}, quality ok` : `unusable: ${(o.reasons || []).join("; ")}`;
      else if (st.tool === "get_gage") txt = o.available ? `gage ${ft(o.latest_ft)}, trend ${o.trend_ft_per_h ?? "–"} ft/h` : (o.note || "gage silent");
      else if (st.tool === "get_upstream") txt = (o.upstream || []).map((u) => u.available ? `upstream ${ft(u.latest_ft)}, trend ${u.trend_ft_per_h ?? "–"} ft/h` : "upstream silent").join("; ");
      else if (st.tool === "view_camera_frame") txt = "looked at the annotated frame";
      else if (st.tool === "set_watch_level") txt = `watch ${st.input.level}, next check in ${st.input.next_check_minutes} min`;
      else if (st.tool === "request_human_approval") txt = o.blocked_by_guardrail ? `BLOCKED by guardrail: ${o.note}` : `asked operator for "${st.input.level}": ${st.input.reason}`;
      else if (st.tool === "flag_sensor_issue") txt = `flagged ${st.input.sensor}: ${st.input.detail}`;
      else if (st.tool === "finish") continue;
      out.push(`<li><span class="tool">${st.tool}</span> <span class="${o.blocked_by_guardrail ? "blocked" : ""}">${escapeHTML(txt)}</span></li>`);
    } else if (st.step === "guard" || st.step === "model_error") out.push(`<li class="blocked">${escapeHTML(st.detail || st.step)}</li>`);
  }
  return out.join("");
}
const escapeHTML = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- live
let liveCam = null;
async function loadLive() {
  let st;
  try { st = await getJSON("data/live/status.json"); } catch { $("live-updated").textContent = "no live data yet"; return; }
  const cam = Object.values(st.cameras)[0];
  liveCam = cam;
  $("live-updated").textContent = `updated ${fmtTime(cam.updated)}`;
  if (cam.frame) $("live-frame").src = cam.frame;
  setBadge($("live-alert"), cam.alert);
  $("live-summary").textContent = cam.summary;
  $("live-watch").textContent = cam.watch;
  $("live-next").textContent = `${cam.next_check_minutes} min`;
  $("live-cycle").textContent = cam.last_cycle ? `${fmtTime(cam.last_cycle.at)} · ${cam.last_cycle.model_calls} model calls · ${cam.last_cycle.latency_s}s` : "–";
  $("live-issues").innerHTML = (cam.issues || []).slice(-2).map((i) => `<div class="issue"><b>${i.sensor} issue</b> (${fmtTime(i.t)}): ${escapeHTML(i.detail)}</div>`).join("");
  const p = cam.pending;
  $("live-pending").hidden = !p;
  if (p) $("pending-text").innerHTML = `<b>${LEVEL_TEXT[p.level] || p.level}</b> requested ${fmtTime(p.requested)}. ${escapeHTML(p.reason)}<br><span class="muted">${escapeHTML(p.evidence)}</span>`;
  const now = Date.now() / 1000, t0 = now - 12 * 3600;
  const cams = cam.readings.filter((r) => r.source === "camera").map((r) => [r.t, r.stage]);
  const gages = cam.readings.filter((r) => r.source === "gage").map((r) => [r.t, r.stage]);
  const vals = [...cams, ...gages].map((p) => p[1]);
  chart($("live-chart"), {
    series: [{ name: "official gage", color: "#1f4e79", points: gages, dots: true }, { name: "camera (OpenCV 5)", color: "#e07a1f", points: cams, dots: true }],
    t0, t1: now, yMin: Math.min(3, ...vals), yMax: Math.max(7, ...vals) + 0.5,
  });
  if (cam.last_cycle?.trace_key) {
    try { const tr = await getJSON(cam.last_cycle.trace_key); $("live-trace").innerHTML = traceItems(tr.steps); } catch {}
  }
}

$("decision-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const decision = e.submitter?.dataset.d || "approve";
  $("decision-msg").textContent = "sending…";
  try {
    const r = await fetch("api/decision", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ cam_id: liveCam.cam_id, decision, code: $("op-code").value }) });
    const j = await r.json();
    $("decision-msg").textContent = r.ok ? `done: public alert is now "${j.alert}"` : j.error;
    if (r.ok) setTimeout(loadLive, 800);
  } catch (err) { $("decision-msg").textContent = String(err); }
});

// ---------------------------------------------------------------- replay
let RP = null, idx = 0, timer = null;
async function loadReplay() {
  RP = await getJSON("data/replay/flood-2026-10-02/replay.json");
  $("scrub").max = RP.cycles.length - 1;
  $("scrub").addEventListener("input", (e) => showCycle(+e.target.value));
  $("play").addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = null; $("play").textContent = "▶ Play"; return; }
    if (idx >= RP.cycles.length - 1) idx = 0;
    $("play").textContent = "❚❚ Pause";
    timer = setInterval(() => { if (idx >= RP.cycles.length - 1) { $("play").click(); return; } showCycle(idx + 1); }, 1100);
  });
  showCycle(firstInteresting());
}
const firstInteresting = () => Math.max(0, RP.cycles.findIndex((c) => c.state.pending || c.state.alert !== "clear") - 3);

function showCycle(i) {
  idx = i; $("scrub").value = i;
  const c = RP.cycles[i];
  $("replay-time").textContent = fmtTime(c.at);
  if (c.frame) $("replay-frame").src = `data/replay/flood-2026-10-02/frames/${c.frame}`;
  $("replay-caption").textContent = `Frame used by the agent at ${fmtTime(c.at)}`;
  setBadge($("replay-alert"), c.state.alert);
  $("replay-summary").textContent = c.summary + (c.state.pending ? ` (Waiting for operator: ${LEVEL_TEXT[c.state.pending.level]}.)` : "");
  $("replay-truth").textContent = ft(c.truth_stage_ft);
  $("replay-cam").textContent = c.camera_stage_ft == null ? (c.camera_note || "unusable") : ft(c.camera_stage_ft);
  $("replay-next").textContent = `${c.state.next_check_minutes} min`;
  $("replay-trace").innerHTML = traceItems(c.steps);
  const markers = RP.cycles.flatMap((cc) => (cc.steps || []).filter((s) => s.tool === "request_human_approval" && s.output && s.output.ok)
    .map(() => ({ t: cc.t, color: "#b00020", label: `operator asked at ${fmtTime(cc.at)}` })));
  chart($("replay-chart"), {
    series: [
      { name: "official gage", color: "#1f4e79", points: RP.gage },
      { name: "upstream gage (12 km)", color: "#7aa6c8", points: RP.upstream },
      { name: "camera (OpenCV 5)", color: "#e07a1f", points: RP.cycles.map((cc) => [cc.camera_t || cc.t, cc.camera_stage_ft]), dots: true },
    ],
    t0: RP.t0, t1: RP.t1, yMin: 3, yMax: 12.5, markers, cursor: c.t,
  });
}

// ---------------------------------------------------------------- evaluation
async function loadEval() {
  const s = await getJSON("data/eval/summary.json");
  const v = s.vision_only, f = v.flood;
  const stat = (b, t) => `<div class="stat"><b>${b}</b><span>${t}</span></div>`;
  $("eval-stats").innerHTML = [
    stat(`${(v.mae_ft * 30.48).toFixed(0)} cm`, `mean absolute error when water flows (${v.frames_above_floor} held-out daytime frames)`),
    stat(`${Math.round(v.within_1ft * 100)}%`, "of readings within ±1 ft (30 cm) of the official gage"),
    stat(`${Math.round(f.recall * 100)}%`, `of frames with the crossing ≥ 1 ft under water detected by the camera alone`),
    stat(`${Math.round(f.precision * 100)}%`, "precision of those camera-only detections (held-out days)"),
    stat(`${s.frames}`, `frames processed, ${s.usable_daytime} usable in daylight`),
  ].join("");
  $("eval-fails").innerHTML = [
    "Night: the camera switches to infrared and the water texture disappears. The quality gate rejects these frames and the agent falls back to the gage.",
    "Sunlit dry gravel: between 09:00 and 12:00 the bridge shadow sweeps the bed and looks like flow. A density and smoothness test on each water body removed most of these (crossing-alert precision went from 33% to 97%); the thresholds were tuned on this camera, so a new site needs checking. Escalation still requires two independent readings.",
    "Above the calibrated range: with the flood day held out, the camera underestimates the 11.7 ft peak (it never saw water that high). Readings far above the calibration are flagged instead of extrapolated, and the agent switches to the gage.", "Missed water: twice during the replay the camera reported no flow while the river was high; the agent looked at the frame, saw water and flagged the camera.",
    "Live, 2026-10-25: the official gage at this site has a silent primary sensor and a secondary sensor stuck at 6.2 ft. The camera shows a dry crossing; the agent flags the gage instead of raising an alert.",
  ].map((t) => `<li>${t}</li>`).join("");
}

loadLive(); loadReplay().catch(() => {}); loadEval().catch(() => {});
setInterval(loadLive, 60_000);
addEventListener("resize", () => { if (RP) showCycle(idx); });
