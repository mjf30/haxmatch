'use strict';
// Carrega a simulação (mesmos arquivos do navegador) num contexto Node.
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function loadSim() {
  const files = ['config.js', 'vec.js', 'input.js', 'game.js', 'ai.js', 'features.js', 'nn.js', 'nnbot.js'];
  const code = files.map((f) => fs.readFileSync(path.join(__dirname, '..', 'js', f), 'utf8').replace(/^'use strict';/, '')).join('\n')
    + '\nthis.__exports = { Game, AI, CFG, V, emptyInput, Features, NN, NNBot };';
  const sandbox = { console, Math, Infinity, Date, Float32Array };
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox, { filename: 'sim-bundle.js' });
  return sandbox.__exports;
}

// Partida: time `nnTeam` controlado pela rede `policy`; o outro por `opp`
// ('script' = bots atuais, ou outra policy). Devolve métricas para o fitness.
// Currículo 'attack': começa com a bola no pé de um jogador da rede perto do gol
// adversário (ensina a finalizar); 'match': partida normal.
function setupScenario(sim, g, nnTeam, scenario, rng) {
  const { CFG } = sim;
  if (scenario !== 'attack') return;
  const dir = nnTeam === 0 ? 1 : -1;
  const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2;
  g.state = 'play'; g.stateT = 0; g.msg = null;
  const mine = g.players.filter((p) => p.team === nnTeam && p.active && p.idx !== 0);
  const carrier = mine[Math.floor(rng() * mine.length)];
  const x = dir * (W2 - 300 - rng() * 500), y = (rng() - 0.5) * H2 * 1.2;
  carrier.pos = { x, y };
  for (const p of mine) if (p !== carrier) p.pos = { x: x - dir * (100 + rng() * 300), y: (rng() - 0.5) * H2 * 1.6 };
  for (const p of g.players.filter((q) => q.team !== nnTeam && q.active)) {
    p.pos = p.idx === 0 ? { x: dir * (W2 - 60), y: (rng() - 0.5) * 100 } : { x: dir * (W2 - 150 - rng() * 600), y: (rng() - 0.5) * H2 * 1.6 };
  }
  g.ball.owner = carrier; g.ball.pos = { x: x + dir * (carrier.r + g.ball.r + 3), y }; g.ball.vel = { x: 0, y: 0 }; g.ball.lock = null;
  carrier.facing = { x: dir, y: 0 }; carrier.moveDir = { x: dir, y: 0 };
}

function mulberry(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function playMatch(sim, policy, opp, opts) {
  const { Game, AI, CFG, NNBot } = sim;
  const seconds = opts.seconds || 60;
  const nnTeam = opts.nnTeam || 0;
  const g = new Game({ teamSize: opts.teamSize || 4, seed: opts.seed || 1 });
  g.time = seconds;   // partida curta
  const rng = mulberry(opts.seed || 1);
  const scenario = opts.scenario || 'match';
  setupScenario(sim, g, nnTeam, scenario, rng);
  const m = { gf: 0, ga: 0, poss: 0, ballX: 0, shots: 0, ticks: 0, touches: 0, onTarget: 0 };
  const steps = Math.floor(seconds / CFG.DT);
  for (let i = 0; i < steps; i++) {
    for (const p of g.players) {
      if (!p.active) continue;
      const mine = p.team === nnTeam;
      const pol = mine ? policy : (opp === 'script' ? null : opp);
      g.setInput(p.id, pol ? NNBot.think(p, g, CFG.DT, pol) : AI.think(p, g, CFG.DT));
    }
    g.step(CFG.DT);
    if (g.state === 'play') {
      m.ticks++;
      const dir = nnTeam === 0 ? 1 : -1;
      m.ballX += g.ball.pos.x * dir / (CFG.FIELD_W / 2);
      if (g.ball.owner && g.ball.owner.team === nnTeam) m.poss++;
    }
    for (const e of g.events) {
      if (e.type === 'goal') {
        if (e.team === nnTeam) m.gf++; else m.ga++;
        // no currículo de ataque, recomeça o cenário após o gol
        if (scenario === 'attack') { g.kickoff(false); setupScenario(sim, g, nnTeam, scenario, rng); }
      }
      if (e.type === 'shot' && e.p.team === nnTeam) {
        m.shots++;
        // chute na direção do gol adversário (linha da bola cruza a boca do gol)
        const dir = nnTeam === 0 ? 1 : -1, b = g.ball;
        if (b.vel.x * dir > 100) {
          const t = (dir * CFG.FIELD_W / 2 - b.pos.x) / b.vel.x;
          if (t > 0 && Math.abs(b.pos.y + b.vel.y * t) < CFG.GOAL_W / 2) m.onTarget++;
        }
      }
      if ((e.type === 'control' || e.type === 'first-touch') && e.p.team === nnTeam) m.touches++;
      if ((e.type === 'save' || e.type === 'catch' || e.type === 'parry') && e.p.team !== nnTeam) m.onTarget++;
    }
    if (g.state === 'end') break;
  }
  return m;
}

function fitnessOf(m) {
  const t = Math.max(1, m.ticks);
  return 10 * (m.gf - m.ga)
    + 3 * (m.poss / t)                 // posse
    + 2 * (m.ballX / t)                // bola no campo de ataque
    + 0.15 * Math.min(12, m.shots)     // finalizar
    + 0.8 * Math.min(8, m.onTarget)    // finalizar no gol
    + 0.02 * Math.min(50, m.touches);  // ir na bola
}

module.exports = { loadSim, playMatch, fitnessOf };
