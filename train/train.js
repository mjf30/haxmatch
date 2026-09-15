'use strict';
// Treino por neuroevolução (OpenAI-ES com amostragem antitética e ranks).
// Uso: node train/train.js [--gens 200] [--pop 32] [--seconds 60] [--sigma 0.05] [--lr 0.03]
//                          [--attack 0.5] [--build 0.3] (frações de partidas nos currículos de finalização / construção)
//                          [--league] (self-play contra uma liga de versões anteriores + script)
//                          [--policy macro] (rede híbrida: decisão tática; execução pelo script)
//                          [--resume js/nn_weights.js] [--workers N]
// Salva os melhores pesos em js/nn_weights.js (usado pelo navegador) a cada avaliação.
const os = require('os');
const fs = require('fs');
const path = require('path');
const { Worker } = require('worker_threads');
const { loadSim, playMatch, fitnessOf } = require('./bundle');

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf('--' + name); return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : def; };
const flag = (name) => args.includes('--' + name);
const GENS = parseInt(opt('gens', '200'), 10);
const POP = parseInt(opt('pop', '32'), 10);          // pares antitéticos por geração (2*POP avaliações)
const SECONDS = parseFloat(opt('seconds', '60'));
const SIGMA = parseFloat(opt('sigma', '0.05'));
const LR = parseFloat(opt('lr', '0.03'));
const MATCHES = parseInt(opt('matches', '2'), 10);   // partidas por candidato
const TEAM = parseInt(opt('team', '0'), 10);   // 0 = sorteia 3v3/4v4/5v5 por partida
const WORKERS = parseInt(opt('workers', String(Math.max(1, os.cpus().length - 1))), 10);
const SELFPLAY = flag('selfplay') || flag('league');
const ATTACK = parseFloat(opt('attack', '0'));     // fração de partidas no cenário de ataque
const BUILD = parseFloat(opt('build', '0'));       // fração no cenário de construção
const LEAGUE_MAX = 6;
const RESUME = opt('resume', null);
const OUT = path.join(__dirname, '..', 'js', 'nn_weights.js');
const LATEST = path.join(__dirname, 'latest.json');   // theta mais recente (retomar com --resume train/latest.json)

const sim = loadSim();
const { NN, NNBot, MacroBot, Features } = sim;
const KIND = opt('policy', 'macro');
const sizes = KIND === 'macro' ? MacroBot.SIZES_DEFAULT : NNBot.SIZES_DEFAULT;
const N = NN.paramCount(sizes);

function mulberry(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rng = mulberry(12345 + Date.now() % 10000);
function gauss() { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

let theta;
if (RESUME && fs.existsSync(RESUME)) {
  const txt = fs.readFileSync(RESUME, 'utf8');
  const json = JSON.parse(txt.slice(txt.indexOf('{'), txt.lastIndexOf('}') + 1));
  theta = Float32Array.from(json.w);
  console.log('retomando de', RESUME, '(', theta.length, 'parâmetros )');
} else {
  theta = NN.init(sizes, rng);
}
console.log(`política ${KIND} · rede ${sizes.join('x')} = ${N} parâmetros · observação ${Features.SIZE} · workers ${WORKERS} · pop ${2 * POP} · ${SECONDS}s/partida · ${MATCHES} partidas/candidato · ${SELFPLAY ? 'self-play' : 'vs bots script'}`);

// ---- pool de workers ----
const workers = [];
const pending = new Map();
let nextId = 1;
for (let i = 0; i < WORKERS; i++) {
  const w = new Worker(path.join(__dirname, 'worker.js'));
  w.on('message', (m) => { const r = pending.get(m.id); if (r) { pending.delete(m.id); r(m); } });
  w.on('error', (e) => { console.error('worker erro', e); process.exit(1); });
  workers.push(w);
}
let rr = 0;
// opps: lista de 'script' | Float32Array (um por partida, cíclico); scenarios: 'match' | 'attack'
function evaluate(w, opps, seeds, scenarios) {
  return new Promise((resolve) => {
    const id = nextId++;
    pending.set(id, resolve);
    workers[rr++ % workers.length].postMessage({
      id, sizes, kind: KIND, w: Array.from(w),
      opps: opps.map((o) => (o === 'script' ? 'script' : Array.from(o))),
      seeds, scenarios, opts: { seconds: SECONDS, teamSize: TEAM },
    });
  });
}

function save(w, info) {
  const obj = { kind: KIND, sizes, obs: Features.SIZE, info, w: Array.from(w).map((v) => Math.round(v * 10000) / 10000) };
  fs.writeFileSync(OUT, `'use strict';\n// Pesos treinados por train/train.js (${new Date().toISOString()}). ${info}\nconst NN_WEIGHTS = ${JSON.stringify(obj)};\n`);
}

// Adam para o gradiente estimado
const mAdam = new Float32Array(N), vAdam = new Float32Array(N);
let adamT = 0;

(async () => {
  const base0 = theta.slice();   // versão inicial: referência fixa na avaliação e na liga
  const t0 = Date.now();
  const league = [base0];   // versões anteriores (self-play em liga)
  const EVAL_SEEDS = [7, 21, 3, 11, 42, 99];
  const evalOpps = SELFPLAY ? ['script', base0] : ['script'];
  // avalia a versão inicial: só salva o que a superar
  const r0 = await evaluate(base0, evalOpps, EVAL_SEEDS, ['match']);
  let best = r0.fitness, bestW = theta.slice();
  console.log(`avaliação da versão inicial: fitness ${best.toFixed(2)} · gols ${r0.ms.reduce((s, m) => s + m.gf, 0)}:${r0.ms.reduce((s, m) => s + m.ga, 0)}`);
  for (let gen = 1; gen <= GENS; gen++) {
    const gt = Date.now();
    // oponentes desta geração: sempre um script; em liga, o resto sorteado da liga
    const opps = ['script'];
    if (SELFPLAY && league.length) for (let m = 1; m < MATCHES; m++) opps.push(league[Math.floor(rng() * league.length)]);
    // cenários: fração ATTACK no currículo de finalização
    const scenarios = [];
    // mistura FIXA por geração (evita gradiente ruidoso): nA de ataque, nB de construção, resto partida
    const nA = Math.round(ATTACK * MATCHES), nB = Math.round(BUILD * MATCHES);
    for (let m = 0; m < MATCHES; m++) scenarios.push(m < nA ? 'attack' : (m < nA + nB ? 'build' : 'match'));
    const eps = [];
    const evals = [];
    const seeds = [];
    for (let m = 0; m < MATCHES; m++) seeds.push(1000 * gen + m);
    for (let i = 0; i < POP; i++) {
      const e = new Float32Array(N);
      for (let k = 0; k < N; k++) e[k] = gauss();
      eps.push(e);
      const plus = new Float32Array(N), minus = new Float32Array(N);
      for (let k = 0; k < N; k++) { plus[k] = theta[k] + SIGMA * e[k]; minus[k] = theta[k] - SIGMA * e[k]; }
      evals.push(evaluate(plus, opps, seeds, scenarios), evaluate(minus, opps, seeds, scenarios));
    }
    const res = await Promise.all(evals);
    const fits = res.map((r) => r.fitness);
    // ranks centrados em [-0.5, 0.5]
    const order = fits.map((f, i) => [f, i]).sort((a, b) => a[0] - b[0]);
    const rank = new Float32Array(fits.length);
    order.forEach(([, i], r) => { rank[i] = r / (fits.length - 1) - 0.5; });
    const grad = new Float32Array(N);
    for (let i = 0; i < POP; i++) {
      const rp = rank[2 * i], rm = rank[2 * i + 1];
      const c = (rp - rm) / (2 * POP * SIGMA);
      const e = eps[i];
      for (let k = 0; k < N; k++) grad[k] += c * e[k];
    }
    // Adam
    adamT++;
    const b1 = 0.9, b2 = 0.999;
    for (let k = 0; k < N; k++) {
      mAdam[k] = b1 * mAdam[k] + (1 - b1) * grad[k];
      vAdam[k] = b2 * vAdam[k] + (1 - b2) * grad[k] * grad[k];
      const mh = mAdam[k] / (1 - Math.pow(b1, adamT)), vh = vAdam[k] / (1 - Math.pow(b2, adamT));
      theta[k] += LR * mh / (Math.sqrt(vh) + 1e-8);
    }
    const mean = fits.reduce((a, b) => a + b, 0) / fits.length;
    const mx = Math.max(...fits);
    const gf = res.reduce((a, r) => a + r.ms.reduce((s, m) => s + m.gf, 0), 0), ga = res.reduce((a, r) => a + r.ms.reduce((s, m) => s + m.ga, 0), 0);
    console.log(`gen ${gen} · fitness média ${mean.toFixed(2)} máx ${mx.toFixed(2)} · gols ${gf}:${ga} · ${((Date.now() - gt) / 1000).toFixed(1)}s`);
    fs.writeFileSync(LATEST, JSON.stringify({ kind: KIND, sizes, gen, w: Array.from(theta).map((v) => Math.round(v * 10000) / 10000) }));
    // avaliação do theta atual contra os bots script em sementes fixas
    if (gen % 5 === 0 || gen === GENS) {
      // sementes fixas: metade contra o script, metade contra a versão inicial
      const r = await evaluate(theta, evalOpps, EVAL_SEEDS, ['match']);
      const gfe = r.ms.reduce((s, m) => s + m.gf, 0), gae = r.ms.reduce((s, m) => s + m.ga, 0);
      console.log(`   >> avaliação vs ${SELFPLAY ? 'script+inicial' : 'script'}: fitness ${r.fitness.toFixed(2)} · gols ${gfe}:${gae} em ${r.ms.length} partidas de ${SECONDS}s`);
      if (SELFPLAY) { league.push(theta.slice()); if (league.length > LEAGUE_MAX) league.splice(1, 1); }   // liga: inicial + versões recentes
      if (r.fitness > best) {
        best = r.fitness; bestW = theta.slice();
        save(bestW, `gen ${gen}, fitness vs script ${best.toFixed(2)}, gols ${gfe}:${gae}`);
        console.log('   >> salvo em js/nn_weights.js');
      }
    }
  }
  console.log(`fim · melhor fitness ${best.toFixed(2)} · ${((Date.now() - t0) / 60000).toFixed(1)} min`);
  for (const w of workers) w.terminate();
})();
