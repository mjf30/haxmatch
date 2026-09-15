'use strict';
// Avalia js/nn_raw_weights.js (PPO, controle total) contra os bots script e o híbrido.
// Uso: node train/eval_raw.js [partidas=6] [segundos=180]
const fs = require('fs');
const path = require('path');
const { loadSim, playMatch } = require('./bundle');
const sim = loadSim();
const load = (file) => { const t = fs.readFileSync(path.join(__dirname, '..', 'js', file), 'utf8'); return JSON.parse(t.slice(t.indexOf('{'), t.lastIndexOf('}') + 1)); };
const raw = load('nn_raw_weights.js');
const pol = { kind: 'raw2', sizes: raw.sizes, heads: raw.heads, w: Float32Array.from(raw.w) };
const n = parseInt(process.argv[2] || '6', 10), sec = parseFloat(process.argv[3] || '180');
let gf = 0, ga = 0;
for (let i = 0; i < n; i++) {
  const m = playMatch(sim, pol, 'script', { seed: 700 + i, nnTeam: i % 2, seconds: sec, teamSize: 3 + (i % 3) });
  gf += m.gf; ga += m.ga;
  console.log(`vs script ${3 + (i % 3)}v${3 + (i % 3)} (rede no time ${i % 2 ? 'azul' : 'vermelho'}): ${m.gf}x${m.ga} · posse ${(100 * m.poss / m.ticks).toFixed(0)}% · chutes ${m.shots} (no gol ${m.onTarget}) · passes ${m.passOk}/${m.passes}`);
}
console.log(`total PPO ${gf} x ${ga} script · ${raw.info || ''}`);
if (fs.existsSync(path.join(__dirname, '..', 'js', 'nn_weights.js'))) {
  const h = load('nn_weights.js');
  if (h.kind === 'macro') {
    const hyb = { kind: 'macro', sizes: h.sizes, w: Float32Array.from(h.w) };
    let a = 0, b = 0;
    for (let i = 0; i < n; i++) { const m = playMatch(sim, pol, hyb, { seed: 800 + i, nnTeam: i % 2, seconds: sec, teamSize: 3 + (i % 3) }); a += m.gf; b += m.ga; }
    console.log(`total PPO ${a} x ${b} híbrido`);
  }
}
