'use strict';
// Vetor de observação para a rede neural, do ponto de vista de um jogador.
// Tudo no "referencial do time": o ataque é sempre para +x (o time 1 é espelhado),
// posições relativas ao jogador e normalizadas. Tamanho fixo (vagas vazias = zeros).
const Features = (() => {
  const MAX_MATES = 4, MAX_OPPS = 5;
  const SELF = 22, BALL = 12, MATE = 7, OPP = 7, GOALS = 6, MISC = 6;
  const SIZE = SELF + BALL + MAX_MATES * MATE + MAX_OPPS * OPP + GOALS + MISC;
  const POS = 1 / (CFG.FIELD_W / 2);   // posições em [-1, 1]
  const VEL = 1 / 300;
  const DIST = 1 / 1000;

  function build(p, g, out) {
    const dir = p.team === 0 ? 1 : -1;   // espelha o time 1 para atacar em +x
    const b = g.ball;
    const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2;
    let k = 0;
    const put = (v) => { out[k++] = v; };
    const relPos = (q) => { put((q.x - p.pos.x) * dir * POS); put((q.y - p.pos.y) * POS); };
    const vel = (v) => { put(v.x * dir * VEL); put(v.y * VEL); };

    // ---- eu (22) ----
    put(p.pos.x * dir * POS); put(p.pos.y / H2);
    vel(p.vel);
    put(p.facing.x * dir); put(p.facing.y);
    put(p.stamina / CFG.STAMINA_MAX); put(p.exhausted ? 1 : 0);
    put(p.effortBar); put(p.effortT > 0 ? 1 : 0);
    put(p.isKeeper ? 1 : 0);
    put(g.hasBall(p) ? 1 : 0); put(p.held ? 1 : 0);
    put(g.inOwnBox(p) ? 1 : 0);
    put(p.stance === 'drib' ? 1 : 0); put(p.stance === 'def' ? 1 : 0);
    put(p.cd.tackle > 0 ? 1 : 0); put(p.cd.slide > 0 ? 1 : 0); put(p.cd.dribble > 0 ? 1 : 0);
    put(p.action ? 1 : 0);
    put((p.recover > 0 || p.fallen > 0 || p.getup > 0) ? 1 : 0);
    put(p.dribbleLag > 0 ? 1 : 0);

    // ---- bola (12) ----
    relPos(b.pos); put(V.dist(p.pos, b.pos) * DIST);
    vel(b.vel); put(V.len(b.vel) * DIST);
    const o = b.owner;
    put(!o ? 1 : 0); put(o === p ? 1 : 0); put(o && o !== p && o.team === p.team ? 1 : 0); put(o && o.team !== p.team ? 1 : 0);
    put(p.reach ? 1 : 0);                       // alvo disponível para ação de primeira
    put(b.lock && b.lock !== p ? 1 : 0);        // outro jogador tem a prioridade

    // ---- companheiros (4 × 7), por distância ----
    const mates = g.players.filter((q) => q.active && q.team === p.team && q !== p)
      .sort((a, c) => V.dist(a.pos, p.pos) - V.dist(c.pos, p.pos));
    for (let i = 0; i < MAX_MATES; i++) {
      const q = mates[i];
      if (!q) { for (let j = 0; j < MATE; j++) put(0); continue; }
      put(1); relPos(q.pos); vel(q.vel); put(b.owner === q ? 1 : 0); put(q.isKeeper ? 1 : 0);
    }
    // ---- adversários (5 × 7), por distância ----
    const opps = g.players.filter((q) => q.active && q.team !== p.team)
      .sort((a, c) => V.dist(a.pos, p.pos) - V.dist(c.pos, p.pos));
    for (let i = 0; i < MAX_OPPS; i++) {
      const q = opps[i];
      if (!q) { for (let j = 0; j < OPP; j++) put(0); continue; }
      put(1); relPos(q.pos); vel(q.vel); put(b.owner === q ? 1 : 0); put(q.isKeeper ? 1 : 0);
    }

    // ---- gols (6) ----
    const oppGoal = { x: dir * W2, y: 0 }, ownGoal = { x: -dir * W2, y: 0 };
    relPos(oppGoal); put(V.dist(p.pos, oppGoal) * DIST);
    relPos(ownGoal); put(V.dist(p.pos, ownGoal) * DIST);

    // ---- diversos (6) ----
    put(mates.some((q) => q.isKeeper) || p.isKeeper ? 1 : 0);
    put(opps.some((q) => q.isKeeper) ? 1 : 0);
    put(V.clamp((g.score[p.team] - g.score[1 - p.team]) / 3, -1, 1));
    put(g.time / CFG.MATCH_TIME);
    put((H2 - Math.abs(p.pos.y)) * DIST);        // distância à parede lateral mais próxima
    put((W2 - Math.abs(p.pos.x)) * DIST);        // distância à linha de fundo mais próxima
    return out;
  }

  return { build, SIZE, MAX_MATES, MAX_OPPS };
})();
