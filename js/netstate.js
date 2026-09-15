'use strict';
// Serialização compacta do estado do jogo para enviar do host aos convidados.
// Só o que o render/HUD precisa; a simulação continua exclusiva do host.
const NetState = (() => {
  const STANCES = ['none', 'drib', 'def'];
  const FLASH_EVENTS = new Set(['steal', 'tackle-fail', 'knockdown', 'save', 'parry', 'effort', 'gloves', 'fake', 'goal', 'first-touch']);

  // 1 casa decimal para posições/velocidades; 3 para vetores unitários (direções) e frações
  const r1 = (v) => (Math.abs(v) <= 1 ? Math.round(v * 1000) / 1000 : Math.round(v * 10) / 10);

  function encode(game, seq, events) {
    const b = game.ball;
    const out = {
      t: 'state', seq,
      time: game.time, score: game.score, state: game.state, msg: game.msg,
      ball: [b.pos.x, b.pos.y, b.vel.x, b.vel.y, b.spin, b.owner ? b.owner.id : -1, b.rot],
      players: game.players.map((p) => [
        p.pos.x, p.pos.y, p.vel.x, p.vel.y, p.facing.x, p.facing.y,
        STANCES.indexOf(p.stance),
        p.action ? p.action.type : '', p.action ? p.action.dir.x : 0, p.action ? p.action.dir.y : 0,
        p.fallen, p.getup, p.recover, p.isKeeper ? 1 : 0, p.held ? 1 : 0, p.holdT,
        p.charge ? p.charge.kind : '', p.charge ? p.charge.t : 0,
        p.charge && p.charge.dir0 ? p.charge.dir0.x : 0, p.charge && p.charge.dir0 ? p.charge.dir0.y : 0, p.charge ? p.charge.spin || 0 : 0,
        p.queued ? p.queued.kind : '', p.queued ? p.queued.t : 0,
        p.queued && p.queued.dir0 ? p.queued.dir0.x : 0, p.queued && p.queued.dir0 ? p.queued.dir0.y : 0, p.queued ? p.queued.spin || 0 : 0,
        p.stamina, p.effortBar, p.effortT, p.pushFlash, p.callT, p.sprinting ? 1 : 0, p.moving ? 1 : 0, p.human ? 1 : 0, p.name, p.active ? 1 : 0,
      ]),
      ev: events.filter((e) => FLASH_EVENTS.has(e.type)).map((e) => ({ type: e.type, p: e.p ? e.p.id : -1, v: e.victim ? e.victim.id : -1, team: e.team })),
    };
    out.ball = out.ball.map((v, i) => (i === 5 ? v : r1(v)));
    for (const a of out.players) for (let i = 0; i < a.length; i++) if (typeof a[i] === 'number') a[i] = r1(a[i]);
    return out;
  }

  function apply(game, s) {
    game.time = s.time; game.score = s.score; game.state = s.state; game.msg = s.msg;
    s.players.forEach((a, i) => {
      const p = game.players[i];
      let k = 0;
      p.pos = { x: a[k++], y: a[k++] }; p.vel = { x: a[k++], y: a[k++] }; p.facing = { x: a[k++], y: a[k++] };
      p.stance = STANCES[a[k++]];
      const at = a[k++], adx = a[k++], ady = a[k++];
      p.action = at ? { type: at, dir: { x: adx, y: ady }, t: 0, dur: 1 } : null;
      p.fallen = a[k++]; p.getup = a[k++]; p.recover = a[k++]; p.isKeeper = !!a[k++]; p.held = !!a[k++]; p.holdT = a[k++];
      const ck = a[k++], ct = a[k++], cdx = a[k++], cdy = a[k++], cs = a[k++];
      p.charge = ck ? { kind: ck, t: ct, dir0: { x: cdx, y: cdy }, spin: cs } : null;
      const qk = a[k++], qt = a[k++], qdx = a[k++], qdy = a[k++], qs = a[k++];
      p.queued = qk ? { kind: qk, t: qt, dir0: { x: qdx, y: qdy }, spin: qs } : null;
      p.stamina = a[k++]; p.effortBar = a[k++]; p.effortT = a[k++]; p.pushFlash = a[k++]; p.callT = a[k++];
      p.sprinting = !!a[k++]; p.moving = !!a[k++]; p.human = !!a[k++]; p.name = a[k++]; p.active = !!a[k++];
    });
    const b = game.ball, bb = s.ball;
    b.pos = { x: bb[0], y: bb[1] }; b.vel = { x: bb[2], y: bb[3] }; b.spin = bb[4];
    b.owner = bb[5] >= 0 ? game.players[bb[5]] : null; b.rot = bb[6];
    return s.ev.map((e) => ({ type: e.type, p: e.p >= 0 ? game.players[e.p] : null, victim: e.v >= 0 ? game.players[e.v] : null, team: e.team }));
  }

  // avança posições entre snapshots para o desenho ficar suave
  function extrapolate(game, dt) {
    for (const p of game.players) if (p.active) p.pos = V.add(p.pos, V.mul(p.vel, dt));
    const b = game.ball;
    b.pos = V.add(b.pos, V.mul(b.vel, dt));
    b.rot += V.len(b.vel) * dt / b.r;
  }

  return { encode, apply, extrapolate };
})();
