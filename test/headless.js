'use strict';
// Roda a simulação sem navegador (bots x bots) para pegar erros e ver se sai gol.
// Uso: node test/headless.js [segundos] [teamSize] [seed]
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const code = ['config.js', 'vec.js', 'input.js', 'game.js', 'ai.js']
  .map((f) => fs.readFileSync(path.join(__dirname, '..', 'js', f), 'utf8').replace(/^'use strict';/, ''))
  .join('\n') + '\nthis.__exports = { Game, AI, CFG, V, emptyInput };';
const sandbox = { console, Math, Infinity, Date };
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'bundle.js' });
const { Game, AI, CFG } = sandbox.__exports;

const seconds = parseFloat(process.argv[2] || '180');
const teamSize = parseInt(process.argv[3] || '4', 10);
const seed = parseInt(process.argv[4] || '7', 10);
const game = new Game({ teamSize, seed });
const counts = {};
let steps = 0;
const maxSteps = Math.floor(seconds / CFG.DT);
for (; steps < maxSteps; steps++) {
  for (const p of game.players) game.setInput(p.id, AI.think(p, game, CFG.DT));
  game.step(CFG.DT);
  for (const e of game.events) counts[e.type] = (counts[e.type] || 0) + 1;
  const b = game.ball;
  if (!Number.isFinite(b.pos.x) || !Number.isFinite(b.pos.y)) throw new Error('bola NaN no passo ' + steps);
  for (const p of game.players) if (!Number.isFinite(p.pos.x) || !Number.isFinite(p.pos.y)) throw new Error('jogador NaN ' + p.name);
  if (Math.abs(b.pos.x) > CFG.FIELD_W / 2 + CFG.GOAL_D + 1 || Math.abs(b.pos.y) > CFG.FIELD_H / 2 + 1) throw new Error(`bola fora: ${b.pos.x.toFixed(1)},${b.pos.y.toFixed(1)}`);
  if (game.state === 'end') break;
}
console.log(`simulados ${(steps * CFG.DT).toFixed(0)}s · placar ${game.score[0]}x${game.score[1]} · estado ${game.state}`);
console.log('eventos:', JSON.stringify(counts));
console.log('goleiros:', game.players.filter((p) => p.isKeeper).map((p) => p.name).join(', '));
