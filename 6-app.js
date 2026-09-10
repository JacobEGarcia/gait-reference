'use strict';
// GAIT REFERENCE LAB - 2D biped tracks a reference gait through a noisy PD actuator model.
// Joints: hipL, kneeL, hipR, kneeR, plus a body pitch DOF that integrates leg torque mismatch.
// Honest model: kinematic reference + 2nd-order actuator dynamics per joint. Not a physics engine.

const Q = new URLSearchParams(location.search);
const FF = parseFloat(Q.get('t') || '0');

const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const rewCv = document.getElementById('rewardCv'), rctx = rewCv.getContext('2d');
const errCv = document.getElementById('errCv'), ectx = errCv.getContext('2d');

const P = { kp: 140, kd: 12, noise: 18, torque: 80, speed: 100 };
for (const k of ['kp','kd','noise','torque','speed']) if (Q.has(k)) P[k] = +Q.get(k);
for (const k of ['kp','kd','noise','torque','speed']) {
  const el = document.getElementById(k);
  el.value = P[k];
  el.addEventListener('input', () => { P[k] = +el.value; labels(); });
}
function labels() {
  kpV.textContent = P.kp; kdV.textContent = P.kd; noV.textContent = (P.noise/100).toFixed(2);
  tqV.textContent = P.torque + '%'; spV.textContent = P.speed + '%';
}
labels();

// ---------- reference gait ----------
let gait = null;
function newGait() {
  gait = {
    freq: 1.1 + Math.random() * 0.9,                 // steps/s factor
    hipAmp: 0.45 + Math.random() * 0.35,
    kneeAmp: 0.5 + Math.random() * 0.5,
    kneeBase: 0.25 + Math.random() * 0.3,
    bob: 0.02 + Math.random() * 0.03,
    sway: 0.04 + Math.random() * 0.06,
    lean: 0.05 + Math.random() * 0.12,
    armAmp: 0.3 + Math.random() * 0.4,
  };
}
newGait();

// reference joint angles at phase ph (radians): {hipL,kneeL,hipR,kneeR, pitch}
function refPose(ph) {
  const s = Math.sin(ph), c = Math.cos(ph);
  return {
    hipL:  gait.hipAmp * s,
    kneeL: gait.kneeBase + gait.kneeAmp * Math.max(0, -Math.cos(ph - 0.6)),
    hipR:  gait.hipAmp * Math.sin(ph + Math.PI),
    kneeR: gait.kneeBase + gait.kneeAmp * Math.max(0, -Math.cos(ph + Math.PI - 0.6)),
    pitch: gait.lean + gait.sway * Math.sin(2 * ph),
    bob:   gait.bob * Math.abs(Math.sin(ph)),
  };
}

// ---------- executed robot state ----------
const J = ['hipL','kneeL','hipR','kneeR'];
let th, thd, pitch, pitchVel, phase, fallen, fallT, ep, epReward, best, rewards, shoveVel, errHist;
function reset(full) {
  th = {hipL:0,kneeL:0.3,hipR:0,kneeR:0.3}; thd = {hipL:0,kneeL:0,hipR:0,kneeR:0};
  pitch = 0; pitchVel = 0; phase = 0; fallen = false; fallT = 0; shoveVel = 0;
  errHist = {hipL:0,kneeL:0,hipR:0,kneeR:0};
  if (full) { ep = 1; epReward = 0; best = null; rewards = []; }
}
reset(true);

document.getElementById('newGait').onclick = () => { newGait(); endEpisode(); };
document.getElementById('shove').onclick = () => { shoveVel += (Math.random() > 0.5 ? 1 : -1) * 2.2; };

function endEpisode() {
  epTime = 0;
  rewards.push(epReward);
  if (rewards.length > 60) rewards.shift();
  if (best === null || epReward > best) best = epReward;
  ep++; epReward = 0;
  pitch = 0; pitchVel = 0; fallen = false; fallT = 0; shoveVel = 0;
  const r = refPose(phase);
  for (const j of J) { th[j] = r[j]; thd[j] = 0; }
}

const DT = 1 / 240;
let acc = 0, last = performance.now(), simTime = 0, epTime = 0;

function step(dt) {
  simTime += dt;
  if (fallen) {
    fallT += dt;
    pitchVel += (Math.sign(pitch || 1) * 6) * dt;      // topple
    pitch += pitchVel * dt;
    if (fallT > 1.6) endEpisode();
    return;
  }
  phase += dt * gait.freq * Math.PI * 2 * 0.5;
  epTime += dt;
  if (epTime > 20) { endEpisode(); return; }           // episode length cap

  const r = refPose(phase);
  const r2 = refPose(phase + 0.01);
  let meanErr = 0;
  for (const j of J) {
    const refVel = (r2[j] - r[j]) / 0.01;
    let tq = P.kp * (r[j] - th[j]) + P.kd * (refVel - thd[j]);
    tq += (Math.random() * 2 - 1) * P.noise;           // actuator noise
    const lim = P.torque * 3;
    tq = Math.max(-lim, Math.min(lim, tq));            // saturation
    thd[j] += (tq / 100) * dt * 60;                    // inertia
    thd[j] *= (1 - 0.6 * dt);
    th[j] += thd[j] * dt;
    const e = Math.abs(r[j] - th[j]);
    meanErr += e / J.length;
    errHist[j] = errHist[j] * 0.98 + e * 0.02;
    epReward += (1 - Math.min(1, e * 3)) * dt * 10;    // dense reward
  }
  // body pitch follows hip torque mismatch + shove
  const hipMismatch = (th.hipL - r.hipL) - (th.hipR - r.hipR);
  pitchVel += (hipMismatch * 3 - pitch * 2 - pitchVel * 1.5) * dt + shoveVel * dt * 8;
  shoveVel *= (1 - 4 * dt);
  pitch += pitchVel * dt;
  epReward -= Math.abs(pitch) * dt * 8;
  if (Math.abs(pitch) > 0.55) { fallen = true; fallT = 0; stateLine.textContent = 'FALLEN - resetting episode'; stateLine.className = 'fall'; }
  meanErrDeg = meanErr * 180 / Math.PI;
}
let meanErrDeg = 0;

// ---------- rendering ----------
function fit() {
  const r = cv.getBoundingClientRect();
  cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio;
}
addEventListener('resize', fit); fit();

function drawBiped(g, pose, opts) {
  const { x, y, scale, color, alpha, lw } = opts;
  g.save();
  g.globalAlpha = alpha; g.strokeStyle = color; g.lineWidth = lw; g.lineCap = 'round';
  const hip = { x, y: y - 0.52 * scale - (pose.bob || 0) * scale };
  const pitchA = pose.pitch;
  // torso: from hip up, tilted by pitch
  const torsoLen = 0.42 * scale;
  const chest = { x: hip.x - Math.sin(pitchA) * torsoLen, y: hip.y - Math.cos(pitchA) * torsoLen };
  const headR = 0.09 * scale;
  g.beginPath(); g.moveTo(hip.x, hip.y); g.lineTo(chest.x, chest.y); g.stroke();
  g.beginPath(); g.arc(chest.x - Math.sin(pitchA) * (headR + 4), chest.y - Math.cos(pitchA) * (headR + 4), headR, 0, 7); g.stroke();
  // legs
  const L1l = 0.30 * scale, L2l = 0.30 * scale;
  for (const side of ['L', 'R']) {
    const hipA = pose['hip' + side] + pitchA * 0.3;
    const kneeA = pose['knee' + side];
    const knee = { x: hip.x + Math.sin(hipA) * L1l, y: hip.y + Math.cos(hipA) * L1l };
    const a2 = hipA - kneeA;
    const foot = { x: knee.x + Math.sin(a2) * L2l, y: knee.y + Math.cos(a2) * L2l };
    g.beginPath(); g.moveTo(hip.x, hip.y); g.lineTo(knee.x, knee.y); g.lineTo(foot.x, foot.y);
    g.lineTo(foot.x + 0.07 * scale, foot.y); g.stroke();
  }
  // arms
  const armA1 = -pose.hipL * 0.8, armA2 = -pose.hipR * 0.8;
  for (const a of [armA1, armA2]) {
    const el = { x: chest.x + Math.sin(a + pitchA) * 0.2 * scale, y: chest.y + Math.cos(a + pitchA) * 0.2 * scale };
    const hd = { x: el.x + Math.sin(a * 1.2 + pitchA - 0.3) * 0.18 * scale, y: el.y + Math.cos(a * 1.2 + pitchA - 0.3) * 0.18 * scale };
    g.beginPath(); g.moveTo(chest.x, chest.y); g.lineTo(el.x, el.y); g.lineTo(hd.x, hd.y); g.stroke();
  }
  g.restore();
}

function draw() {
  const W = cv.width, H = cv.height;
  ctx.fillStyle = '#0b0d10'; ctx.fillRect(0, 0, W, H);
  const scale = Math.min(W, H) * 0.42;
  const groundY = H * 0.82, cx = W * 0.42;
  // floor
  ctx.strokeStyle = 'rgba(255,255,255,0.14)'; ctx.lineWidth = 2 * devicePixelRatio;
  ctx.beginPath(); ctx.moveTo(0, groundY); ctx.lineTo(W, groundY); ctx.stroke();
  // treadmill ticks moving backward
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
  const off = (simTime * gait.freq * 120 * devicePixelRatio) % (60 * devicePixelRatio);
  for (let x = -off; x < W; x += 60 * devicePixelRatio) {
    ctx.beginPath(); ctx.moveTo(x, groundY); ctx.lineTo(x - 12 * devicePixelRatio, groundY + 10 * devicePixelRatio); ctx.stroke();
  }
  // reference ghost
  const rp = refPose(phase);
  drawBiped(ctx, rp, { x: cx, y: groundY, scale, color: '#3b82f6', alpha: fallen ? 0.25 : 0.55, lw: 2.5 * devicePixelRatio });
  // executed
  const ep2 = { hipL: th.hipL, kneeL: th.kneeL, hipR: th.hipR, kneeR: th.kneeR, pitch: pitch + rp.pitch * 0.3, bob: 0 };
  drawBiped(ctx, ep2, { x: cx, y: groundY, scale, color: fallen ? '#f87171' : '#e8edf2', alpha: 0.95, lw: 3 * devicePixelRatio });
  if (!fallen) { stateLine.textContent = 'TRACKING'; stateLine.className = 'ok'; }

  // sidebar stats
  epEl.textContent = ep; rewEl.textContent = epReward.toFixed(0);
  bestEl.textContent = best === null ? '-' : best.toFixed(0);
  errEl.textContent = meanErrDeg.toFixed(1) + ' deg';

  // reward curve
  rctx.fillStyle = '#0b0d10'; rctx.fillRect(0, 0, 272, 56);
  if (rewards.length > 1) {
    const mx = Math.max(...rewards, 1);
    rctx.strokeStyle = '#4ade80'; rctx.lineWidth = 1.5; rctx.beginPath();
    rewards.forEach((r, i) => {
      const x = i / 59 * 292, y = 88 - (r / mx) * 84;
      i ? rctx.lineTo(x, y) : rctx.moveTo(x, y);
    });
    rctx.stroke();
  }
  // per-joint error bars
  ectx.fillStyle = '#0b0d10'; ectx.fillRect(0, 0, 272, 80);
  J.forEach((j, i) => {
    const e = errHist[j] * 180 / Math.PI;
    const h = Math.min(1, e / 25);
    ectx.fillStyle = h > 0.6 ? '#f87171' : h > 0.3 ? '#facc15' : '#3b82f6';
    ectx.fillRect(8 + i * 68, 72 - h * 64, 36, h * 64);
    ectx.fillStyle = '#7d8a97'; ectx.font = '9px JetBrains Mono';
    ectx.fillText(j, 8 + i * 68, 79);
    ectx.fillText(e.toFixed(0) + '°', 10 + i * 72 + 24, 108);
  });
}
const epEl = document.getElementById('ep'), rewEl = document.getElementById('rew'),
      bestEl = document.getElementById('best'), errEl = document.getElementById('err'),
      stateLine = document.getElementById('stateLine');

// fast-forward for tests
if (FF > 0) { let t = 0; while (t < FF) { step(DT); t += DT; } }

function frame(now) {
  requestAnimationFrame(frame);
  let dt = Math.min((now - last) / 1000, 0.1); last = now;
  acc += dt * (P.speed / 100);
  while (acc > DT) { step(DT); acc -= DT; }
  draw();
}
requestAnimationFrame(frame);
