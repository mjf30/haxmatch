'use strict';
// Clonagem da DECISÃO tática do script (7 classes) + DAgger. A execução continua
// sendo a do script, então a rede só precisa aprender "o que fazer", não "como".
// Uso: node train/macro_clone.js [--minutes 40] [--epochs 6] [--dagger 3] [--lr 0.002] [--out js/nn_weights.js]
const fs = require('fs');
const path = require('path');
const { loadSim, playMatch } = require('./bundle');

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i >= 0 && args[i + 1] ? args[i + 1] : d; };
const MINUTES = parseFloat(opt('minutes', '40'));
const EPOCHS = parseInt(opt('epochs', '6'), 10);
const DAGGER = parseInt(opt('dagger', '3'), 10);
const LR = parseFloat(opt('lr', '0.002'));
const BATCH = 256;
const OUT = path.join(__dirname, '..', opt('out', 'js/nn_weights.js'));

const sim = loadSim();
const { Game, AI, CFG, Features, NN, MacroBot } = sim;
const sizes = MacroBot.SIZES_DEFAULT;
const [NI, H1, NO] = sizes;
const N = NN.paramCount(sizes);
const M = AI.MACROS;

function mulberry(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rng = mulberry(991);
const X = [], Y = [];
let seed = 300;
const policy = { sizes, w: NN.init(sizes, mulberry(4242)), kind: 'macro' };

// coleta: rótulo = decisão do script no estado atual; ação = script (actor=null) ou rede
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
      for (const p of g.players) {
        if (!p.active) continue;
        if (!p.isKeeper && g.state === 'play' && i % 2 === 0) {
          const c = AI.context(p, g);
          const label = M.indexOf(AI.chooseMacro(p, g, c));
          X.push(Features.build(p, g, new Float32Array(NI)));
          Y.push(label);
        }
        g.setInput(p.id, useNN[p.id] ? MacroBot.think(p, g, CFG.DT, actor.policy) : AI.think(p, g, CFG.DT));
      }
      g.step(CFG.DT);
      ticks++;
      if (g.state === 'end') break;
    }
  }
  console.log(`  coleta: +${X.length - n0} amostras (${minutes} min, ${actor ? 'rede agindo, beta ' + actor.beta : 'script agindo'}) em ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}

// ---- rede [NI, H1, NO], softmax + entropia cruzada, Adam ----
const w = policy.w;
const off1 = 0, ob1 = NI * H1, off2 = ob1 + H1, ob2 = off2 + H1 * NO;
const grad = new Float32Array(N), mA = new Float32Array(N), vA = new Float32Array(N);
let adamT = 0;
const a1 = new Float32Array(H1), out = new Float32Array(NO), prob = new Float32Array(NO), d2 = new Float32Array(NO), d1 = new Float32Array(H1);
function fwd(x) {
  for (let o = 0; o < H1; o++) { let s = w[ob1 + o]; const b = off1 + o * NI; for (let j = 0; j < NI; j++) s += w[b + j] * x[j]; a1[o] = Math.tanh(s); }
  let mx = -Infinity;
  for (let o = 0; o < NO; o++) { let s = w[ob2 + o]; const b = off2 + o * H1; for (let j = 0; j < H1; j++) s += w[b + j] * a1[j]; out[o] = s; if (s > mx) mx = s; }
  let z = 0; for (let o = 0; o < NO; o++) { prob[o] = Math.exp(out[o] - mx); z += prob[o]; }
  for (let o = 0; o < NO; o++) prob[o] /= z;
}
function bwd(x, y, wgt) {
  for (let o = 0; o < NO; o++) d2[o] = wgt * (prob[o] - (o === y ? 1 : 0));
  for (let j = 0; j < H1; j++) d1[j] = 0;
  for (let o = 0; o < NO; o++) { const b = off2 + o * H1; for (let j = 0; j < H1; j++) { grad[b + j] += d2[o] * a1[j]; d1[j] += w[b + j] * d2[o]; } grad[ob2 + o] += d2[o]; }
  for (let j = 0; j < H1; j++) d1[j] *= (1 - a1[j] * a1[j]);
  for (let o = 0; o < H1; o++) { const b = off1 + o * NI; for (let j = 0; j < NI; j++) grad[b + j] += d1[o] * x[j]; grad[ob1 + o] += d1[o]; }
  return -Math.log(prob[y] + 1e-7) * wgt;
}
function adam(n) {
  adamT++;
  const b1 = 0.9, b2 = 0.999;
  for (let k = 0; k < N; k++) {
    const gk = grad[k] / n;
    mA[k] = b1 * mA[k] + (1 - b1) * gk; vA[k] = b2 * vA[k] + (1 - b2) * gk * gk;
    w[k] -= LR * (mA[k] / (1 - Math.pow(b1, adamT))) / (Math.sqrt(vA[k] / (1 - Math.pow(b2, adamT))) + 1e-8);
    grad[k] = 0;
  }
}
let classW = new Float32Array(NO).fill(1);
function train(epochs) {
  const idx = X.map((_, i) => i);
  const nVal = Math.min(20000, Math.floor(X.length * 0.05));
  // peso por classe (raras valem mais, limitado)
  const cnt = new Float32Array(NO); for (const y of Y) cnt[y]++;
  for (let k = 0; k < NO; k++) classW[k] = Math.min(4, Math.sqrt(Y.length / NO / Math.max(1, cnt[k])));
  for (let ep = 1; ep <= epochs; ep++) {
    for (let i = X.length - nVal - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    let total = 0, n = 0;
    const te = Date.now();
    for (let s = 0; s < X.length - nVal; s++) { const i = idx[s]; fwd(X[i]); total += bwd(X[i], Y[i], classW[Y[i]]); n++; if (n % BATCH === 0) adam(BATCH); }
    if (n % BATCH) adam(n % BATCH);
    let acc = 0; const per = new Float32Array(NO), perN = new Float32Array(NO);
    for (let i = 0; i < nVal; i++) { const k = idx[X.length - 1 - i]; fwd(X[k]); let b = 0; for (let o = 1; o < NO; o++) if (prob[o] > prob[b]) b = o; perN[Y[k]]++; if (b === Y[k]) { acc++; per[Y[k]]++; } }
    console.log(`  época ${ep}: perda ${(total / n).toFixed(3)} · acerto ${(100 * acc / nVal).toFixed(1)}% · por classe: ${M.map((m, k) => `${m} ${perN[k] ? (100 * per[k] / perN[k]).toFixed(0) : '-'}%`).join(' ')} · ${((Date.now() - te) / 1000).toFixed(0)}s`);
  }
}
function quickEval(n) {
  let gf = 0, ga = 0, poss = 0, ticks = 0, shots = 0;
  for (let i = 0; i < n; i++) {
    const m = playMatch(sim, policy, 'script', { seed: 6000 + i, nnTeam: i % 2, seconds: 120, teamSize: 3 + (i % 3) });
    gf += m.gf; ga += m.ga; poss += m.poss; ticks += m.ticks; shots += m.shots;
  }
  console.log(`  >> híbrido x script (${n} partidas de 120 s): ${gf} x ${ga} · posse ${(100 * poss / ticks).toFixed(0)}% · chutes ${shots}`);
}

collect(MINUTES, null);
const cnt = new Float32Array(NO); for (const y of Y) cnt[y]++;
console.log('distribuição das decisões:', M.map((m, k) => `${m} ${(100 * cnt[k] / Y.length).toFixed(1)}%`).join(' · '));
console.log('treino inicial');
train(EPOCHS);
quickEval(6);
const betas = [0.5, 0.2, 0, 0, 0];
for (let it = 1; it <= DAGGER; it++) {
  console.log(`DAgger ${it}/${DAGGER}`);
  collect(Math.max(10, MINUTES / 3), { policy, beta: betas[Math.min(it - 1, betas.length - 1)] });
  train(Math.max(2, Math.round(EPOCHS / 2)));
  quickEval(6);
}
const info = `híbrido: decisão tática clonada do script (${X.length} amostras, DAgger ${DAGGER}x); execução pelo script`;
const obj = { kind: 'macro', sizes, obs: Features.SIZE, info, w: Array.from(w).map((v) => Math.round(v * 10000) / 10000) };
fs.writeFileSync(OUT, `'use strict';\n// Pesos: ${info} (${new Date().toISOString()})\nconst NN_WEIGHTS = ${JSON.stringify(obj)};\n`);
fs.writeFileSync(path.join(__dirname, 'macro_clone.json'), JSON.stringify({ kind: 'macro', sizes, w: obj.w }));
console.log('salvo em', OUT, 'e train/macro_clone.json');
