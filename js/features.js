'use strict';
// Vetor de observação para a rede neural, do ponto de vista de um jogador.
// Tudo no "referencial do time": o ataque é sempre para +x (o time 1 é espelhado),
// posições relativas ao jogador e normalizadas. Tamanho fixo (vagas vazias = zeros).
const Features = (() => {
  const MAX_MATES = 4, MAX_OPPS = 5;
  const SELF = 33, BALL = 12, MATE = 11, OPP = 10, GOALS = 6, MISC = 12;
  const SIZE = SELF + BALL + MAX_MATES * MATE + MAX_OPPS * OPP + GOALS + MISC;
  const POS = 1 / (CFG.FIELD_W / 2);   // posições em [-1, 1]
  const VEL = 1 / 300;
  const DIST = 1 / 1000;

  // Controle de campo: grade de células; cada célula pertence ao jogador ativo mais
  // próximo (Voronoi). Calculado uma vez por tick e guardado no jogo.
  const GX = 12, GY = 7;
  function pitchControl(g) {
    if (g._pc && g._pc.now === g.now && g._pc.n === g.players.length) return g._pc;
    const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2;
    const owner = new Int16Array(GX * GY);
    const cx = new Float32Array(GX), cy = new Float32Array(GY);
    for (let i = 0; i < GX; i++) cx[i] = -W2 + (i + 0.5) * CFG.FIELD_W / GX;
    for (let j = 0; j < GY; j++) cy[j] = -H2 + (j + 0.5) * CFG.FIELD_H / GY;
    const count = new Int16Array(g.players.length);
    const active = g.players.filter((q) => q.active);
    for (let j = 0; j < GY; j++) for (let i = 0; i < GX; i++) {
      let best = -1, bd = Infinity;
      for (const q of active) {
        const dx = q.pos.x - cx[i], dy = q.pos.y - cy[j];
        const d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = q.id; }
      }
      owner[j * GX + i] = best;
      if (best >= 0) count[best]++;
    }
    g._pc = { now: g.now, n: g.players.length, owner, count, cx, cy };
    return g._pc;
  }
  // fração das células de uma faixa de x (no referencial do time) que o time controla
  function teamShare(pc, g, team, dir, xlo, xhi) {
    let own = 0, tot = 0;
    for (let j = 0; j < GY; j++) for (let i = 0; i < GX; i++) {
      const x = pc.cx[i] * dir / (CFG.FIELD_W / 2);
      if (x < xlo || x > xhi) continue;
      tot++;
      const o = pc.owner[j * GX + i];
      if (o >= 0 && g.players[o].team === team) own++;
    }
    return tot ? own / tot : 0;
  }

  function build(p, g, out) {
    const pc = pitchControl(g);
    const cells = GX * GY;
    const dir = p.team === 0 ? 1 : -1;   // espelha o time 1 para atacar em +x
    const b = g.ball;
    const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2;
    let k = 0;
    const put = (v) => { out[k++] = v; };
    const relPos = (q) => { put((q.x - p.pos.x) * dir * POS); put((q.y - p.pos.y) * POS); };
    const vel = (v) => { put(v.x * dir * VEL); put(v.y * VEL); };

    // ---- eu (31) ----
    put(p.pos.x * dir * POS); put(p.pos.y / H2);
    relPos(p.home);                                   // posição-base da formação (papel tático)
    for (let i = 0; i < 5; i++) put(p.idx === i ? 1 : 0);   // índice na formação (0 = goleiro inicial)
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
    put(p.charge ? Math.min(1, p.charge.t / (p.charge.kind === 'shot' ? CFG.CHARGE_MAX : CFG.PASS_CHARGE)) : 0);   // progresso da carga
    put(p.queued ? Math.min(1, (p.queued.age || 0) / CFG.LOCK_MAX) : 0);   // idade da ação travada
    put(pc.count[p.id] / cells * 4);                    // minha fatia de campo (Voronoi), ~0..1
    {                                                    // espaço livre à frente: adversário mais próximo num cone de ataque
      let free = 600;
      for (const q of g.players) {
        if (!q.active || q.team === p.team) continue;
        const d = V.sub(q.pos, p.pos);
        if (d.x * dir > 0 && Math.abs(d.y) < d.x * dir * 0.8 + 40) free = Math.min(free, V.len(d));
      }
      put(free / 600);
    }

    // ---- bola (12) ----
    relPos(b.pos); put(V.dist(p.pos, b.pos) * DIST);
    vel(b.vel); put(V.len(b.vel) * DIST);
    const o = b.owner;
    put(!o ? 1 : 0); put(o === p ? 1 : 0); put(o && o !== p && o.team === p.team ? 1 : 0); put(o && o.team !== p.team ? 1 : 0);
    put(p.reach ? 1 : 0);                       // alvo disponível para ação de primeira
    put(b.lock && b.lock !== p ? 1 : 0);        // outro jogador tem a prioridade

    // linha livre entre dois pontos (nenhum adversário a menos de 26 px da linha, fora as pontas)
    const oppsAll = g.players.filter((q) => q.active && q.team !== p.team);
    const laneFree = (a, c2) => oppsAll.every((o) => V.dist(o.pos, a) <= 30 || V.dist(o.pos, c2) <= 30 || V.segDist(o.pos, a, c2) >= o.r + 26) ? 1 : 0;
    const nearestOppD = (pt) => oppsAll.length ? Math.min(...oppsAll.map((o) => V.dist(o.pos, pt))) : 1000;
    const oppsNear = (pt) => oppsAll.filter((o) => V.dist(o.pos, pt) < 200).length;
    // ---- companheiros (4 × 11), por distância ----
    const mates = g.players.filter((q) => q.active && q.team === p.team && q !== p)
      .sort((a, c) => V.dist(a.pos, p.pos) - V.dist(c.pos, p.pos));
    for (let i = 0; i < MAX_MATES; i++) {
      const q = mates[i];
      if (!q) { for (let j = 0; j < MATE; j++) put(0); continue; }
      put(1); relPos(q.pos); vel(q.vel); put(b.owner === q ? 1 : 0); put(q.isKeeper ? 1 : 0); put(pc.count[q.id] / cells * 4);
      put(Math.min(1, nearestOppD(q.pos) / 400));   // quão marcado ele está
      put(oppsNear(q.pos) / 5);                      // adversários perto dele
      put(laneFree(b.pos, q.pos));                   // linha de passe (da bola até ele) livre
    }
    // ---- adversários (5 × 10), por distância ----
    const opps = oppsAll.slice().sort((a, c) => V.dist(a.pos, p.pos) - V.dist(c.pos, p.pos));
    const oppGoalPt = { x: dir * W2, y: 0 };
    for (let i = 0; i < MAX_OPPS; i++) {
      const q = opps[i];
      if (!q) { for (let j = 0; j < OPP; j++) put(0); continue; }
      put(1); relPos(q.pos); vel(q.vel); put(b.owner === q ? 1 : 0); put(q.isKeeper ? 1 : 0); put(pc.count[q.id] / cells * 4);
      put(Math.min(1, V.dist(q.pos, b.pos) * DIST));                          // distância dele à bola
      put(V.segDist(q.pos, p.pos, oppGoalPt) < q.r + 30 ? 1 : 0);              // está na minha linha de chute
    }

    // ---- gols (6) ----
    const oppGoal = { x: dir * W2, y: 0 }, ownGoal = { x: -dir * W2, y: 0 };
    relPos(oppGoal); put(V.dist(p.pos, oppGoal) * DIST);
    relPos(ownGoal); put(V.dist(p.pos, ownGoal) * DIST);

    // ---- diversos (12) ----
    put(mates.some((q) => q.isKeeper) || p.isKeeper ? 1 : 0);
    put(opps.some((q) => q.isKeeper) ? 1 : 0);
    put(V.clamp((g.score[p.team] - g.score[1 - p.team]) / 3, -1, 1));
    put(g.time / CFG.MATCH_TIME);
    put((H2 - Math.abs(p.pos.y)) * DIST);        // distância à parede lateral mais próxima
    put((W2 - Math.abs(p.pos.x)) * DIST);        // distância à linha de fundo mais próxima
    // aglomeração e espaço: companheiros / adversários num raio de 220 px, e distância ao adversário mais próximo
    put(mates.filter((q) => V.dist(q.pos, p.pos) < 220).length / 4);
    put(opps.filter((q) => V.dist(q.pos, p.pos) < 220).length / 5);
    put(Math.min(1, (opps.length ? Math.min(...opps.map((q) => V.dist(q.pos, p.pos))) : 1000) / 400));
    // controle de campo do time: total, terço de ataque, terço de defesa (Voronoi)
    put(teamShare(pc, g, p.team, dir, -1, 1));
    put(teamShare(pc, g, p.team, dir, 1 / 3, 1));
    put(teamShare(pc, g, p.team, dir, -1, -1 / 3));
    return out;
  }

  return { build, SIZE, MAX_MATES, MAX_OPPS };
})();
