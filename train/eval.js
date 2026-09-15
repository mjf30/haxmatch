'use strict';
// Avalia js/nn_weights.js contra os bots script em partidas completas.
// Uso: node train/eval.js [partidas=6] [segundos=360]
const fs = require('fs');
const path = require('path');
const { loadSim, playMatch, fitnessOf } = require('./bundle');
const sim = loadSim();
const txt = fs.readFileSync(process.argv[4] || path.join(__dirname, '..', 'js', 'nn_weights.js'), 'utf8');
const json = JSON.parse(txt.slice(txt.indexOf('{'), txt.lastIndexOf('}') + 1));
const policy = { sizes: json.sizes, w: Float32Array.from(json.w), kind: json.kind || 'raw' };
const n = parseInt(process.argv[2] || '6', 10), sec = parseFloat(process.argv[3] || '360');
let gf = 0, ga = 0, fit = 0;
for (let i = 0; i < n; i++) {
  const m = playMatch(sim, policy, 'script', { seed: 500 + i, nnTeam: i % 2, seconds: sec, teamSize: 3 + (i % 3) });   // 3v3, 4v4, 5v5
  gf += m.gf; ga += m.ga; fit += fitnessOf(m);
  console.log(`partida ${i + 1} (${3 + (i % 3)}v${3 + (i % 3)}, rede no time ${i % 2 === 0 ? 'vermelho' : 'azul'}): ${m.gf}x${m.ga} · posse ${(100 * m.poss / m.ticks).toFixed(0)}% · chutes ${m.shots} (no gol ${m.onTarget}) · passes ${m.passOk}/${m.passes} (longos ${m.longOk}) · espaçamento no ataque ${(500 * m.spread / Math.max(1, m.attackTicks || 0)).toFixed(0)}px`);
}
console.log(`total rede ${gf} x ${ga} script · fitness média ${(fit / n).toFixed(2)} · ${json.info || ''}`);
