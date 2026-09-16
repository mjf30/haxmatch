'use strict';
// Carrega a simulação (mesmos arquivos do navegador) num contexto Node.
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function loadSim() {
  const files = ['config.js', 'vec.js', 'input.js', 'game.js', 'value_map.js', 'ai.js', 'features.js', 'nn.js', 'nnbot.js', 'macrobot.js', 'rawbot.js'];
  const code = files.map((f) => fs.readFileSync(path.join(__dirname, '..', 'js', f), 'utf8').replace(/^'use strict';/, '')).join('\n')
    + '\nthis.__exports = { Game, AI, CFG, V, emptyInput, Features, NN, NNBot, MacroBot, RawBot, VALUE_MAP: (typeof VALUE_MAP === "undefined" ? null : VALUE_MAP) };';
  const sandbox = { console, Math, Infinity, Date, Float32Array };
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox, { filename: 'sim-bundle.js' });
  return sandbox.__exports;
}

// Partida: time `nnTeam` controlado pela rede `policy`; o outro por `opp`
// ('script' = bots atuais, ou outra policy). Devolve métricas para o fitness.
// Currículo 'attack': começa com a bola no pé de um jogador da rede perto do gol
// adversário (ensina a finalizar); 'match': partida normal.
// 'build': bola no pé no meio-campo com a defesa adversária posicionada (ponte
// entre o ataque puro e a partida completa).
function setupScenario(sim, g, nnTeam, scenario, rng) {
  const { CFG } = sim;
  if (scenario !== 'attack' && scenario !== 'build') return;
  const dir = nnTeam === 0 ? 1 : -1;
  const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2;
  g.state = 'play'; g.stateT = 0; g.msg = null;
  const mine = g.players.filter((p) => p.team === nnTeam && p.active && p.idx !== 0);
  const carrier = mine[Math.floor(rng() * mine.length)];
  const attack = scenario === 'attack';
  const x = attack ? dir * (W2 - 300 - rng() * 500) : dir * (rng() * 400 - 300), y = (rng() - 0.5) * H2 * 1.2;
  carrier.pos = { x, y };
  for (const p of mine) if (p !== carrier) p.pos = { x: x - dir * (rng() * 300 - (attack ? 100 : -150)), y: (rng() - 0.5) * H2 * 1.6 };
  for (const p of g.players.filter((q) => q.team !== nnTeam && q.active)) {
    if (p.idx === 0) p.pos = { x: dir * (W2 - 60), y: (rng() - 0.5) * 100 };
    else p.pos = attack ? { x: dir * (W2 - 150 - rng() * 600), y: (rng() - 0.5) * H2 * 1.6 } : { x: dir * (W2 - 250 - rng() * 700), y: (rng() - 0.5) * H2 * 1.6 };
  }
  g.ball.owner = carrier; g.ball.pos = { x: x + dir * (carrier.r + g.ball.r + 3), y }; g.ball.vel = { x: 0, y: 0 }; g.ball.lock = null;
  carrier.facing = { x: dir, y: 0 }; carrier.moveDir = { x: dir, y: 0 };
}

function mulberry(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function playMatch(sim, policy, opp, opts) {
  const { Game, AI, CFG, V, NNBot, MacroBot, RawBot } = sim;
  const act = (p, g, pol) => (pol.kind === 'macro' ? MacroBot.think(p, g, CFG.DT, pol) : pol.kind === 'raw2' ? RawBot.think(p, g, CFG.DT, pol) : NNBot.think(p, g, CFG.DT, pol));
  const seconds = opts.seconds || 60;
  const nnTeam = opts.nnTeam || 0;
  const rng = mulberry(opts.seed || 1);
  const teamSize = opts.teamSize || (3 + Math.floor(rng() * 3));   // 0 = sorteia 3v3 / 4v4 / 5v5
  const st = process.env.STYLE || 'balanced';   // estilo dos bots script (os dois times)
  const g = new Game({ teamSize, seed: opts.seed || 1, styles: opts.styles || [st, st] });
  g.time = seconds;   // partida curta
  const scenario = opts.scenario || 'match';
  setupScenario(sim, g, nnTeam, scenario, rng);
  const m = { gf: 0, ga: 0, poss: 0, ballX: 0, shots: 0, ticks: 0, touches: 0, onTarget: 0, stall: 0, episodes: 1, scenario, passes: 0, passOk: 0, longOk: 0, spread: 0, crowd: 0 };
  let lastPass = null;   // {tick, team}
  const steps = Math.floor(seconds / CFG.DT);
  const EPISODE = Math.floor((scenario === 'build' ? (opts.buildEpisode || 15) : (opts.attackEpisode || 8)) / CFG.DT);   // episódio: 8 s (ataque) / 15 s (construção) e recomeça
  let epTick = 0, possStreak = 0;
  for (let i = 0; i < steps; i++) {
    for (const p of g.players) {
      if (!p.active) continue;
      const mine = p.team === nnTeam;
      const pol = mine ? policy : (opp === 'script' ? null : opp);
      g.setInput(p.id, pol ? act(p, g, pol) : AI.think(p, g, CFG.DT));
    }
    g.step(CFG.DT);
    if (g.state === 'play') {
      m.ticks++;
      const dir = nnTeam === 0 ? 1 : -1;
      m.ballX += g.ball.pos.x * dir / (CFG.FIELD_W / 2);
      if (g.ball.owner && g.ball.owner.team === nnTeam) {
        m.poss++;
        possStreak++;
        if (possStreak > 4 / CFG.DT) m.stall++;   // enrolando: mais de 4 s seguidos com a bola sem soltar
      } else possStreak = 0;
      // espaçamento/aglomeração só quando o time da rede TEM a bola (na defesa, compactar é legítimo)
      if (g.ball.owner && g.ball.owner.team === nnTeam) {
        m.attackTicks = (m.attackTicks || 0) + 1;
        const mine = g.players.filter((p) => p.active && p.team === nnTeam && !p.isKeeper);
        let sum = 0, n = 0, near = 0;
        for (let a = 0; a < mine.length; a++) {
          if (V.dist(mine[a].pos, g.ball.pos) < 150) near++;
          for (let b = a + 1; b < mine.length; b++) { sum += Math.min(500, V.dist(mine[a].pos, mine[b].pos)); n++; }
        }
        if (n) m.spread += sum / n / 500;
        if (near >= 3) m.crowd++;
      }
    }
    if ((scenario === 'attack' || scenario === 'build') && ++epTick >= EPISODE) { epTick = 0; m.episodes++; g.kickoff(false); setupScenario(sim, g, nnTeam, scenario, rng); }
    for (const e of g.events) {
      if (e.type === 'goal') {
        if (e.team === nnTeam) m.gf++; else m.ga++;
        // no currículo de ataque, recomeça o cenário após o gol
        if (scenario === 'attack' || scenario === 'build') { epTick = 0; m.episodes++; g.kickoff(false); setupScenario(sim, g, nnTeam, scenario, rng); }
      }
      if ((e.type === 'pass' || (e.type === 'shot' && e.p.ai && ['longpass', 'through', 'switch'].includes(e.p.ai.macro))) && e.p.team === nnTeam) { m.passes++; lastPass = { tick: i, team: nnTeam, from: e.p, long: e.p.ai && ['longpass', 'through', 'switch'].includes(e.p.ai.macro) }; }
      if ((e.type === 'control' || e.type === 'first-touch') && lastPass && e.p.team === nnTeam && e.p !== lastPass.from && i - lastPass.tick < 2.5 / CFG.DT) { m.passOk++; if (lastPass.long) m.longOk++; lastPass = null; }
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
  if (m.scenario === 'attack') {
    // currículo de finalização: só marcar e chutar no gol contam (nada de posse)
    return 10 * m.gf - 3 * m.ga + 1.5 * m.onTarget + 0.3 * m.shots - 0.002 * m.stall;
  }
  if (m.scenario === 'build') {
    // construção: levar a bola ao ataque e finalizar; enrolar é penalizado
    return 10 * m.gf - 5 * m.ga + 1.5 * m.onTarget + 0.3 * m.shots + 1.5 * (m.ballX / t) - 0.004 * m.stall;
  }
  // FITNESS=result: só resultado e estilo de passes (sem chutes, progressão e posse, que causaram deriva para bola longa)
  if (process.env.FITNESS === 'result') {
    return 10 * (m.gf - m.ga) + 1.0 * Math.min(30, m.passOk) + 2.0 * Math.min(8, m.longOk)
      + 1.5 * (m.spread / Math.max(1, m.attackTicks || 0)) - 0.006 * m.crowd - 0.004 * m.stall;
  }
  // "ganhar jogando bem": gol pesa mais, mas rodar a bola, lançar e inverter também contam
  return 10 * (m.gf - m.ga)
    + 4.0 * (m.poss / t)               // posse
    + 1.0 * (m.ballX / t)              // bola no campo de ataque
    + 0.2 * Math.min(12, m.shots)      // finalizar
    + 1.0 * Math.min(8, m.onTarget)    // finalizar no gol
    + 0.02 * Math.min(50, m.touches)   // ir na bola
    + 1.0 * Math.min(30, m.passOk)     // passes completados (trocar passes)
    + 2.0 * Math.min(8, m.longOk)      // lançamentos / profundidade / inversões que chegam
    + 3.0 * (m.spread / Math.max(1, m.attackTicks || 0))   // espaçamento entre companheiros (com a bola)
    - 0.006 * m.crowd                  // três ou mais em cima da bola
    - 0.004 * m.stall;                 // enrolar com a bola
}

module.exports = { loadSim, playMatch, fitnessOf };
