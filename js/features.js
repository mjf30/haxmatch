'use strict';
// Vetor de observação para a rede neural, do ponto de vista de um jogador.
// Tudo no "referencial do time": o ataque é sempre para +x (o time 1 é espelhado),
// posições relativas ao jogador e normalizadas. Tamanho fixo (vagas vazias = zeros).
const Features = (() => {
  const MAX_MATES = 4, MAX_OPPS = 5;
  const SELF = 38, BALL = 14, MATE = 12, OPP = 14, GOALS = 14, MISC = 23;
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

  // estrutura de um time (jogadores de linha): centro, largura (y), profundidade (x), área da caixa
  function teamShape(list) {
    if (!list.length) return { cx: 0, cy: 0, w: 0, d: 0, area: 0, minx: 0, maxx: 0, miny: 0, maxy: 0 };
    let sx = 0, sy = 0, minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    for (const q of list) { sx += q.pos.x; sy += q.pos.y; minx = Math.min(minx, q.pos.x); maxx = Math.max(maxx, q.pos.x); miny = Math.min(miny, q.pos.y); maxy = Math.max(maxy, q.pos.y); }
    return { cx: sx / list.length, cy: sy / list.length, w: maxy - miny, d: maxx - minx, area: (maxy - miny) * (maxx - minx), minx, maxx, miny, maxy };
  }
  // vértice do envoltório convexo: é o extremo do time em alguma de 16 direções
  function onHull(p, list) {
    if (list.length <= 2) return 1;
    for (let k = 0; k < 16; k++) {
      const a = k * Math.PI / 8, dx = Math.cos(a), dy = Math.sin(a);
      const me = p.pos.x * dx + p.pos.y * dy;
      if (list.every((q) => q === p || q.pos.x * dx + q.pos.y * dy <= me + 1e-6)) return 1;
    }
    return 0;
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
    put(p.cd.dash > 0 ? 1 : 0); put(p.cd.dive > 0 ? 1 : 0);   // cooldowns do dash e do mergulho (ações do controle total)
    // "linha" para a estrutura: quem não é goleiro, ou o goleiro fora da área (líbero)
    const isFieldQ = (q) => !q.isKeeper || !g.inOwnBox(q);
    const myField = g.players.filter((q) => q.active && q.team === p.team && isFieldQ(q));
    put(isFieldQ(p) ? onHull(p, myField) : 0);                    // sou vértice do envoltório convexo do time (borda) ou estou no meio
    {   // último homem: companheiros entre mim e o meu gol; adversários entre mim e o gol deles
      let behind = 0, ahead = 0;
      for (const q of g.players) {
        if (!q.active || q === p) continue;
        const dx = (q.pos.x - p.pos.x) * dir;
        if (q.team === p.team) { if (dx < 0) behind++; } else if (dx > 0) ahead++;
      }
      put(behind / 4); put(ahead / 5);
    }

    // ---- bola (12) ----
    relPos(b.pos); put(V.dist(p.pos, b.pos) * DIST);
    vel(b.vel); put(V.len(b.vel) * DIST);
    const o = b.owner;
    put(!o ? 1 : 0); put(o === p ? 1 : 0); put(o && o !== p && o.team === p.team ? 1 : 0); put(o && o.team !== p.team ? 1 : 0);
    put(p.reach ? 1 : 0);                       // alvo disponível para ação de primeira
    put(b.lock && b.lock !== p ? 1 : 0);        // outro jogador tem a prioridade
    {                                           // onde a bola vai parar (sem paredes), relativo a mim
      const k = CFG.BALL_DRAG, sp = V.len(b.vel);
      const t = Math.min(3, Math.max(0, Math.log(Math.max(1, sp / 40)) / k));
      const f = (1 - Math.exp(-k * t)) / k;
      relPos(g.clampField(V.add(b.pos, V.mul(b.vel, f)), 10));
    }

    // linha livre entre dois pontos (nenhum adversário a menos de 26 px da linha, fora as pontas)
    const oppsAll = g.players.filter((q) => q.active && q.team !== p.team);
    const laneFree = (a, c2) => oppsAll.every((o) => V.dist(o.pos, a) <= 30 || V.dist(o.pos, c2) <= 30 || V.segDist(o.pos, a, c2) >= o.r + 26) ? 1 : 0;
    const nearestOppD = (pt) => oppsAll.length ? Math.min(...oppsAll.map((o) => V.dist(o.pos, pt))) : 1000;
    const oppsNear = (pt) => oppsAll.filter((o) => V.dist(o.pos, pt) < 200).length;
    const freeAheadOf = (q) => {   // adversário mais próximo num cone de ataque à frente de q
      let free = 600;
      for (const o of oppsAll) { const d = V.sub(o.pos, q.pos); if (d.x * dir > 0 && Math.abs(d.y) < d.x * dir * 0.8 + 40) free = Math.min(free, V.len(d)); }
      return free / 600;
    };
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
      put(freeAheadOf(q));                           // espaço livre à frente dele (profundidade)
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
      put((q.fallen > 0 || q.getup > 0) ? 1 : 0);                              // caído / levantando
      put(q.action ? 1 : 0);                                                   // em tackle, carrinho, dash…
      put(q.stance === 'def' ? 1 : 0);                                         // postura defensiva
      put(Math.min(1, V.dist(q.pos, { x: -dir * W2, y: 0 }) * DIST));          // perigo: distância dele ao meu gol
    }

    // ---- gols (6) ----
    const oppGoal = { x: dir * W2, y: 0 }, ownGoal = { x: -dir * W2, y: 0 };
    relPos(oppGoal); put(V.dist(p.pos, oppGoal) * DIST);
    relPos(ownGoal); put(V.dist(p.pos, ownGoal) * DIST);
    {   // chute: linha livre para cada canto; goleiro adversário relativo ao gol dele
      const blockers = oppsAll.filter((o) => !o.isKeeper);
      for (const s of [-1, 1]) {
        const corner = { x: oppGoal.x, y: s * (CFG.GOAL_W / 2 - 30) };
        put(blockers.every((o) => V.segDist(o.pos, b.pos, corner) >= o.r + 30) ? 1 : 0);
      }
      const gk = oppsAll.find((o) => o.isKeeper);
      if (gk) { put(1); put((gk.pos.x - oppGoal.x) * dir * DIST); put(gk.pos.y / (CFG.GOAL_W / 2)); }
      else { put(0); put(0); put(0); }
      // espaço livre à esquerda e à direita (cone lateral)
      for (const s of [-1, 1]) {
        let free = 600;
        for (const o of oppsAll) { const d = V.sub(o.pos, p.pos); if (d.y * s > 0 && Math.abs(d.x) < d.y * s * 0.8 + 40) free = Math.min(free, V.len(d)); }
        put(free / 600);
      }
      put(0);   // reservado
    }

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
    {   // contra-ataque: adversários (de linha) mais perto do meu gol do que o meu último defensor de linha
      const myField = g.players.filter((q) => q.active && q.team === p.team && !q.isKeeper);
      const lastX = myField.length ? Math.min(...myField.map((q) => q.pos.x * dir)) : 0;
      put(oppsAll.filter((o) => !o.isKeeper && o.pos.x * dir < lastX).length / 5);
    }
    put(g.state === 'play' ? 1 : 0);
    {   // estrutura do meu time e do adversário (jogadores de linha)
      const mine = teamShape(myField);
      const theirs = teamShape(oppsAll.filter(isFieldQ));
      relPos({ x: mine.cx, y: mine.cy }); put(mine.w / CFG.FIELD_H); put(mine.d / CFG.FIELD_W); put(mine.area / (CFG.FIELD_W * CFG.FIELD_H) * 4);
      relPos({ x: theirs.cx, y: theirs.cy }); put(theirs.w / CFG.FIELD_H); put(theirs.d / CFG.FIELD_W);
    }
    return out;
  }

  // Controle de campo por tempo de chegada (Spearman / Fernández-Bornn), grade fina 40x23, com velocidade:
  // t_i(célula) = REACT + |célula - (pos_i + vel_i*REACT)| / SPRINT; controle do time 0 = logística((t1 - t0)/TAU).
  // Guarda, por célula e por time, o melhor e o segundo melhor tempo (e quem é o melhor), para avaliar
  // rapidamente "e se o jogador p estivesse em outro lugar". Independente da grade 12x7 da observação da rede.
  const TX = 40, TY = 23, REACT = 0.2, TAU = 0.3;
  function pitchControlTT(g) {
    if (g._pctt && g._pctt.now === g.now && g._pctt.n === g.players.length) return g._pctt;
    const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2, n = TX * TY;
    const cx = new Float32Array(TX), cy = new Float32Array(TY);
    for (let i = 0; i < TX; i++) cx[i] = -W2 + (i + 0.5) * CFG.FIELD_W / TX;
    for (let j = 0; j < TY; j++) cy[j] = -H2 + (j + 0.5) * CFG.FIELD_H / TY;
    const t1 = [new Float32Array(n).fill(1e9), new Float32Array(n).fill(1e9)];
    const t2 = [new Float32Array(n).fill(1e9), new Float32Array(n).fill(1e9)];
    const who = [new Int16Array(n).fill(-1), new Int16Array(n).fill(-1)];
    for (const q of g.players) {
      if (!q.active) continue;
      const px = q.pos.x + q.vel.x * REACT, py = q.pos.y + q.vel.y * REACT, T = t1[q.team], T2 = t2[q.team], Wq = who[q.team];
      for (let j = 0; j < TY; j++) for (let i = 0; i < TX; i++) {
        const k = j * TX + i;
        const t = REACT + Math.hypot(cx[i] - px, cy[j] - py) / CFG.SPRINT;
        if (t < T[k]) { T2[k] = T[k]; T[k] = t; Wq[k] = q.id; } else if (t < T2[k]) T2[k] = t;
      }
    }
    const p0 = new Float32Array(n);   // probabilidade de controle do time 0
    for (let k = 0; k < n; k++) p0[k] = 1 / (1 + Math.exp((t1[0][k] - t1[1][k]) / TAU));
    g._pctt = { now: g.now, n: g.players.length, TX, TY, cx, cy, t1, t2, who, p0, REACT, TAU };
    return g._pctt;
  }
  // Mapa de passe: P(a bola chega à célula), a partir da posição atual da bola, contra o time que não tem a posse.
  // A bola percorre a reta a V_BALL; em pontos a cada 60 px, cada adversário tem tempo de chegada (reação + distância
  // projetada / sprint); interceptação no ponto = logística(t_bola - t_adv); chegar = produto dos (1 - interceptação).
  // Global por tick (cache); a mesma grade fina do pitchControlTT. Sem dono da bola: todos 1.
  const V_BALL = 520;
  function passMap(g) {
    if (g._pmap && g._pmap.now === g.now && g._pmap.n === g.players.length) return g._pmap;
    const pc = pitchControlTT(g);
    const n = pc.TX * pc.TY, pm = new Float32Array(n).fill(1);
    const o = g.ball.owner;
    if (o) {
      const opps = g.players.filter((q) => q.active && q.team !== o.team).map((q) => ({ x: q.pos.x + q.vel.x * pc.REACT, y: q.pos.y + q.vel.y * pc.REACT }));
      const bx = g.ball.pos.x, by = g.ball.pos.y;
      for (let j = 0; j < pc.TY; j++) for (let i = 0; i < pc.TX; i++) {
        const k = j * pc.TX + i, dx = pc.cx[i] - bx, dy = pc.cy[j] - by, d = Math.hypot(dx, dy);
        if (d < 40) continue;
        let ok = 1;
        for (let s = 60; s <= d; s += 60) {
          const sx = bx + dx * (s / d), sy = by + dy * (s / d), tb = s / V_BALL;
          let to = Infinity;
          for (const q of opps) { const t = pc.REACT + Math.hypot(q.x - sx, q.y - sy) / CFG.SPRINT; if (t < to) to = t; }
          ok *= 1 - 1 / (1 + Math.exp((to - tb) / pc.TAU));
          if (ok < 0.01) break;
        }
        pm[k] = ok;
      }
    }
    g._pmap = { now: g.now, n: g.players.length, p: pm, owner: o ? o.id : -1 };
    return g._pmap;
  }
  return { build, SIZE, MAX_MATES, MAX_OPPS, pitchControl, pitchControlTT, passMap };
})();
