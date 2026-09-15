'use strict';
// Clonagem de comportamento: treina a rede (supervisionado, backprop em JS puro)
// para imitar os bots script a partir das mesmas observações. Gera um ponto de
// partida competente para o ES/self-play.
// Uso: node train/clone.js [--minutes 25] [--epochs 6] [--dagger 4] [--lr 0.001] [--batch 256] [--out js/nn_weights.js]
const fs = require('fs');
const path = require('path');
const { loadSim, playMatch } = require('./bundle');

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i >= 0 && args[i + 1] ? args[i + 1] : d; };
const MINUTES = parseFloat(opt('minutes', '25'));
const EPOCHS = parseInt(opt('epochs', '6'), 10);
const LR = parseFloat(opt('lr', '0.001'));
const BATCH = parseInt(opt('batch', '256'), 10);
const OUT = path.join(__dirname, '..', opt('out', 'js/nn_weights.js'));

const sim = loadSim();
const { Game, AI, CFG, V, Features, NN, NNBot } = sim;
const sizes = NNBot.SIZES_DEFAULT;
const [NI, H1, H2, NO] = sizes;
const N = NN.paramCount(sizes);

// ---------- 1) dados ----------
// collect(minutes, actor): partidas onde os jogadores agem pelo script (actor=null)
// ou pela rede (actor={policy, beta}: cada jogador segue o script com prob. beta,
// senão a rede). Os RÓTULOS vêm sempre do script (DAgger).
function mulberry(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rng = mulberry(777);
const X = [], Y = [];
let seed = 100;
function collect(minutes, actor) {
  const t0 = Date.now(), n0 = X.length;
  const ticksTarget = Math.floor(minutes * 60 / CFG.DT);
  let ticks = 0;
  while (ticks < ticksTarget) {
    const teamSize = 3 + Math.floor(rng() * 3);
    const g = new Game({ teamSize, seed: seed++ });
    g.time = 120;
    const useNN = g.players.map(() => actor && rng() >= actor.beta);
    for (let i = 0; i < 120 / CFG.DT && ticks < ticksTarget; i++) {
      const labels = [];
      for (const p of g.players) {
        const expert = AI.think(p, g, CFG.DT);           // rótulo (mantém o estado interno do script)
        labels.push([p, expert]);
        g.setInput(p.id, useNN[p.id] ? NNBot.think(p, g, CFG.DT, actor.policy) : expert);
      }
      if (i % 3 === 0 && g.state === 'play') {
        for (const [p, inp] of labels) {
          if (!p.active) continue;
          const dir = p.team === 0 ? 1 : -1;
          const x = Features.build(p, g, new Float32Array(NI));
          const ax = V.clamp((inp.aim.x - p.pos.x) * dir / 600, -0.95, 0.95), ay = V.clamp((inp.aim.y - p.pos.y) / 600, -0.95, 0.95);
          const y = new Float32Array(NO);
          y[0] = V.clamp(inp.mx * dir, -0.95, 0.95); y[1] = V.clamp(inp.my, -0.95, 0.95); y[2] = ax; y[3] = ay;
          y[4] = inp.shoot ? 1 : 0; y[5] = inp.pass ? 1 : 0; y[6] = inp.sprint ? 1 : 0; y[7] = inp.stance ? 1 : 0;
          y[8] = inp.special ? 1 : 0; y[9] = inp.tackle ? 1 : 0; y[10] = inp.throwBall ? 1 : 0; y[11] = inp.call ? 1 : 0;
          X.push(x); Y.push(y);
        }
      }
      g.step(CFG.DT);
      ticks++;
      if (g.state === 'end') break;
    }
  }
  console.log(`  coleta: +${X.length - n0} amostras (${minutes} min, ${actor ? 'rede agindo, beta ' + actor.beta : 'script agindo'}) em ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
collect(MINUTES, null);
const posRate = new Float32Array(NO);
for (const y of Y) for (let k = 4; k < NO; k++) posRate[k] += y[k];
console.log('frequência dos botões:', ['chute', 'passe', 'sprint', 'postura', 'especial', 'tackle', 'arremesso', 'pedir'].map((n, i) => `${n} ${(100 * posRate[i + 4] / Y.length).toFixed(1)}%`).join(' · '));
// botões raros (chute, passe, tackle…) recebem peso maior nos positivos, senão a rede aprende a nunca apertar
const posW = new Float32Array(NO);
for (let k = 4; k < NO; k++) { const p = Math.max(1e-4, posRate[k] / Y.length); posW[k] = Math.min(5, Math.sqrt((1 - p) / p)); }
const MOVE_W = 3, AIM_W = 1.5;   // o movimento é o que mais importa em campo

// ---------- 2) treino supervisionado ----------
const w = NN.init(sizes, mulberry(4242));
const off1 = 0, ob1 = NI * H1, off2 = ob1 + H1, ob2 = off2 + H1 * H2, off3 = ob2 + H2, ob3 = off3 + H2 * NO;
const grad = new Float32Array(N), mA = new Float32Array(N), vA = new Float32Array(N);
let adamT = 0;
const a1 = new Float32Array(H1), a2 = new Float32Array(H2), out = new Float32Array(NO);
const d3 = new Float32Array(NO), d2 = new Float32Array(H2), d1 = new Float32Array(H1);
const sig = (v) => 1 / (1 + Math.exp(-v));

function forwardSample(x) {
  for (let o = 0; o < H1; o++) { let s = w[ob1 + o]; const b = off1 + o * NI; for (let j = 0; j < NI; j++) s += w[b + j] * x[j]; a1[o] = Math.tanh(s); }
  for (let o = 0; o < H2; o++) { let s = w[ob2 + o]; const b = off2 + o * H1; for (let j = 0; j < H1; j++) s += w[b + j] * a1[j]; a2[o] = Math.tanh(s); }
  for (let o = 0; o < NO; o++) { let s = w[ob3 + o]; const b = off3 + o * H2; for (let j = 0; j < H2; j++) s += w[b + j] * a2[j]; out[o] = s; }
}
// perda: MSE em tanh(out) para as 4 saídas contínuas; BCE em sigmoid(out) para os botões
function backwardSample(x, y) {
  let loss = 0;
  for (let o = 0; o < NO; o++) {
    if (o < 4) { const wgt = o < 2 ? MOVE_W : AIM_W; const t = Math.tanh(out[o]); const e = t - y[o]; loss += wgt * e * e; d3[o] = wgt * 2 * e * (1 - t * t); }
    else { const s = sig(out[o]); const wgt = y[o] > 0.5 ? posW[o] : 1; loss -= wgt * (y[o] * Math.log(s + 1e-7) + (1 - y[o]) * Math.log(1 - s + 1e-7)); d3[o] = wgt * (s - y[o]) * 0.5; }
  }
  for (let j = 0; j < H2; j++) d2[j] = 0;
  for (let o = 0; o < NO; o++) { const b = off3 + o * H2; for (let j = 0; j < H2; j++) { grad[b + j] += d3[o] * a2[j]; d2[j] += w[b + j] * d3[o]; } grad[ob3 + o] += d3[o]; }
  for (let j = 0; j < H2; j++) d2[j] *= (1 - a2[j] * a2[j]);
  for (let j = 0; j < H1; j++) d1[j] = 0;
  for (let o = 0; o < H2; o++) { const b = off2 + o * H1; for (let j = 0; j < H1; j++) { grad[b + j] += d2[o] * a1[j]; d1[j] += w[b + j] * d2[o]; } grad[ob2 + o] += d2[o]; }
  for (let j = 0; j < H1; j++) d1[j] *= (1 - a1[j] * a1[j]);
  for (let o = 0; o < H1; o++) { const b = off1 + o * NI; for (let j = 0; j < NI; j++) grad[b + j] += d1[o] * x[j]; grad[ob1 + o] += d1[o]; }
  return loss;
}
function adamStep(n) {
  adamT++;
  const b1 = 0.9, b2 = 0.999;
  for (let k = 0; k < N; k++) {
    const g = grad[k] / n;
    mA[k] = b1 * mA[k] + (1 - b1) * g; vA[k] = b2 * vA[k] + (1 - b2) * g * g;
    w[k] -= LR * (mA[k] / (1 - Math.pow(b1, adamT))) / (Math.sqrt(vA[k] / (1 - Math.pow(b2, adamT))) + 1e-8);
    grad[k] = 0;
  }
}
let idx = X.map((_, i) => i);
let nVal = Math.min(20000, Math.floor(X.length * 0.05));
function validate() {
  let loss = 0, accB = 0, cnt = 0, tp = 0, pos = 0;
  for (let i = 0; i < nVal; i++) {
    const x = X[idx[X.length - 1 - i]], y = Y[idx[X.length - 1 - i]];
    forwardSample(x);
    for (let o = 0; o < 4; o++) { const e = Math.tanh(out[o]) - y[o]; loss += e * e; }
    for (let o = 4; o < NO; o++) { const pr = sig(out[o]) > 0.5; if (pr === (y[o] > 0.5)) accB++; cnt++; if (y[o] > 0.5) { pos++; if (pr) tp++; } }
  }
  return { mse: loss / nVal / 4, acc: accB / cnt, recall: pos ? tp / pos : 0 };
}
function train(epochs) {
  idx = X.map((_, i) => i);
  nVal = Math.min(20000, Math.floor(X.length * 0.05));
  for (let ep = 1; ep <= epochs; ep++) {
    for (let i = X.length - nVal - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    let total = 0, n = 0;
    const te = Date.now();
    for (let s = 0; s < X.length - nVal; s++) {
      const i = idx[s];
      forwardSample(X[i]);
      total += backwardSample(X[i], Y[i]);
      n++;
      if (n % BATCH === 0) adamStep(BATCH);
    }
    if (n % BATCH) adamStep(n % BATCH);
    const v = validate();
    console.log(`  época ${ep}: perda ${(total / n).toFixed(3)} · validação: mse ${v.mse.toFixed(4)}, acerto botões ${(100 * v.acc).toFixed(1)}%, recall ${(100 * v.recall).toFixed(1)}% · ${((Date.now() - te) / 1000).toFixed(0)}s`);
  }
}
const policy = { sizes, w };
function quickEval(n) {
  let gf = 0, ga = 0, poss = 0, ticks = 0, shots = 0;
  for (let i = 0; i < n; i++) {
    const m = playMatch(sim, policy, 'script', { seed: 5000 + i, nnTeam: i % 2, seconds: 90, teamSize: 4 });
    gf += m.gf; ga += m.ga; poss += m.poss; ticks += m.ticks; shots += m.shots;
  }
  console.log(`  >> rede x script (${n} partidas de 90 s): ${gf} x ${ga} · posse ${(100 * poss / ticks).toFixed(0)}% · chutes ${shots}`);
  return gf - ga;
}

console.log('treino inicial (clonagem pura)');
train(EPOCHS);
quickEval(4);
// DAgger: a rede joga, o script rotula os estados visitados, agrega e retreina
const DAGGER = parseInt(opt('dagger', '4'), 10);
const betas = [0.5, 0.25, 0, 0, 0, 0, 0, 0];
for (let it = 1; it <= DAGGER; it++) {
  console.log(`DAgger ${it}/${DAGGER}`);
  collect(Math.max(10, MINUTES / 3), { policy, beta: betas[Math.min(it - 1, betas.length - 1)] });
  train(Math.max(2, Math.round(EPOCHS / 2)));
  quickEval(4);
}

// ---------- 3) salva ----------
const info = `clonagem dos bots script + DAgger ${DAGGER}x: ${X.length} amostras`;
const obj = { sizes, obs: Features.SIZE, info, w: Array.from(w).map((v) => Math.round(v * 10000) / 10000) };
fs.writeFileSync(OUT, `'use strict';\n// Pesos: ${info} (${new Date().toISOString()})\nconst NN_WEIGHTS = ${JSON.stringify(obj)};\n`);
fs.writeFileSync(path.join(__dirname, 'clone.json'), JSON.stringify({ sizes, w: obj.w }));
console.log('salvo em', OUT, 'e train/clone.json');
