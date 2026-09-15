'use strict';
// Verifica que encode/apply do NetState reproduz o estado (host -> convidado).
// Uso: node test/netstate.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const code = ['config.js', 'vec.js', 'input.js', 'game.js', 'ai.js', 'netstate.js']
  .map((f) => fs.readFileSync(path.join(__dirname, '..', 'js', f), 'utf8').replace(/^'use strict';/, ''))
  .join('\n') + '\nthis.__exports = { Game, AI, CFG, V, NetState };';
const sandbox = { console, Math, Infinity, Date };
vm.createContext(sandbox);
vm.runInContext(code, sandbox, { filename: 'bundle.js' });
const { Game, AI, CFG, NetState } = sandbox.__exports;

const host = new Game({ teamSize: 4, seed: 5 });
const guest = new Game({ teamSize: 4, seed: 5 });
let maxErr = 0, snapshots = 0, bytes = 0;
for (let i = 0; i < 60 * 90; i++) {
  for (const p of host.players) host.setInput(p.id, AI.think(p, host, CFG.DT));
  host.step(CFG.DT);
  if (i % 2 === 0) {
    const s = NetState.encode(host, i, host.events);
    const json = JSON.stringify(s);
    bytes += json.length;
    snapshots++;
    NetState.apply(guest, JSON.parse(json));
    for (let k = 0; k < host.players.length; k++) {
      const a = host.players[k], b = guest.players[k];
      maxErr = Math.max(maxErr, Math.hypot(a.pos.x - b.pos.x, a.pos.y - b.pos.y));
      if (a.stance !== b.stance || a.isKeeper !== b.isKeeper || (a.action ? a.action.type : '') !== (b.action ? b.action.type : '')) throw new Error('campo divergente em ' + a.name);
    }
    const ho = host.ball.owner ? host.ball.owner.id : -1, go = guest.ball.owner ? guest.ball.owner.id : -1;
    if (ho !== go) throw new Error('dono da bola divergente');
    if (host.score[0] !== guest.score[0] || host.time !== guest.time) throw new Error('placar/tempo divergente');
  }
}
console.log(`snapshots ${snapshots} · erro máx de posição ${maxErr.toExponential(2)} · média ${(bytes / snapshots).toFixed(0)} bytes (JSON) · ${(bytes / snapshots * 30 / 1024).toFixed(1)} KB/s a 30 Hz`);
