'use strict';
// Bots programados, em duas camadas:
//   chooseMacro(p, g, ctx)  -> decisão tática (uma de MACROS)
//   execute(p, g, dt, ctx, macro, inp) -> movimento, mira e botões que realizam a decisão
// A rede neural (js/macrobot.js) substitui só a decisão; a execução é a mesma.
const AI = (() => {
  // com bola: shoot, pass (melhor opção), passback (opção segura atrás), dribble, hold (proteger e esperar)
  // sem bola: chase (ir na bola solta), defend (pressionar o portador), cover (fechar o espaço
  //           entre a bola e o gol / marcar adversário livre), openfwd / openwide / openback
  //           (abrir espaço à frente / pelo lado / atrás do portador para receber), home
  // com bola: shoot, pass, passback, longpass, switch (inverter o lado), dribble, hold (segurar/esperar)
  // sem bola: chase, defend, cover, openfwd, openwide, overlap (ultrapassar por fora), runbox
  //           (desmarcar na área), openback, home
  //           runspace (atacar o espaço além da última linha), guardgoal (assumir o gol
  //           quando o goleiro saiu), through (passe em profundidade no ponto futuro)
  // goleiro sem bola: gk_angle (fechar o ângulo), gk_rush (sair pressionando/varrer),
  //           gk_line (ficar na linha), gk_up (subir como líbero)
  //           shootq (chute rápido, carga baixa), carryspace (conduzir para o espaço livre),
  //           openbest (abrir no ponto mais livre da grade), cutlane (cortar a linha de passe)
  const MACROS = ['shoot', 'shootq', 'pass', 'passback', 'longpass', 'through', 'switch', 'dribble', 'carryspace', 'hold',
    'chase', 'defend', 'cover', 'cutlane', 'openfwd', 'openwide', 'overlap', 'runbox', 'runspace', 'openback', 'openbest', 'guardgoal', 'home',
    'gk_angle', 'gk_press', 'gk_rush', 'gk_line', 'gk_up', 'cross'];
  // cross: cruzamento da lateral (chute forte com curva para o ponto futuro de um companheiro na área)
  // gk_press: sai fechando o ângulo agressivamente com a postura defensiva, sem tackle nem mergulho
  const GK_MACROS = ['gk_angle', 'gk_press', 'gk_rush', 'gk_line', 'gk_up'];
  const KICK_MACROS = ['shoot', 'shootq', 'pass', 'passback', 'longpass', 'through', 'switch', 'cross'];

  // ---------- estilos: pesos das notas e dos papéis ----------
  // prog/space/margin/share: pesos da nota de passe; through/long/switch/cross/passback: bônus por tipo;
  // carry: valor da condução; carrySpace: espaço mínimo para arrancar; hold: valor de proteger;
  // width: fator da largura (abertura/ultrapassagem); runners: corredores em profundidade (0..2);
  // compact: quanto o posicionamento sem bola acompanha a bola (compactação na defesa)
  const STYLES = {
    // A decisão com a bola é a mesma para todos (oportunidade clara vence sempre); o estilo só dá bônus pequenos e positivos.
    // O que define o estilo é a movimentação: vagas de papéis (support apoio, wide ponta, fwd opção curta à frente, runner profundidade),
    // distâncias (supBack/supSide, fwdDist, wideY/wideX), gatilho das corridas, ponto de abertura (gridRadius/gridProg),
    // contra-pressão ao perder a bola alta, compactação e forma base (homeWidth/homeAdvance).
    balanced: { prog: 0.5, space: 0.3, margin: 0.2, share: 0.15, through: 0.15, long: -0.03, switch: 0, cross: 0.1, passback: -0.2, carry: 0.55, carrySpace: 380, hold: 0.1,
      supBack: 240, supSide: 120, fwdDist: 280, wideY: 380, wideX: 60, slots: ['support', 'wide', 'runner', 'wide'], runTrigger: 'space', gridRadius: 520, gridProg: 0.5, counterpress: 1, compact: 1.0, homeWidth: 1.0, homeAdvance: 0 },
    short: { prog: 0.5, space: 0.3, margin: 0.2, share: 0.15, through: 0.15, long: -0.03, switch: 0, cross: 0.1, passback: -0.2, carry: 0.55, carrySpace: 380, hold: 0.1,
      supBack: 160, supSide: 160, fwdDist: 260, wideY: 270, wideX: 100, slots: ['support', 'fwd', 'wide', 'runner'], runTrigger: 'space', gridRadius: 450, gridProg: 0.5, counterpress: 2, compact: 1.0, homeWidth: 0.9, homeAdvance: 60 },
    long: { prog: 0.5, space: 0.3, margin: 0.2, share: 0.15, through: 0.15, long: 0.05, switch: 0.1, cross: 0.2, passback: -0.2, carry: 0.55, carrySpace: 380, hold: 0.1,
      supBack: 300, supSide: 220, fwdDist: 360, wideY: 480, wideX: 200, slots: ['wide', 'wide', 'runner', 'support'], runTrigger: 'space', gridRadius: 650, gridProg: 0.6, counterpress: 1, compact: 0.9, homeWidth: 1.5, homeAdvance: 60 },
    direct: { prog: 0.5, space: 0.3, margin: 0.2, share: 0.15, through: 0.25, long: -0.03, switch: 0, cross: 0.1, passback: -0.2, carry: 0.6, carrySpace: 380, hold: 0.1,
      supBack: 180, supSide: 140, fwdDist: 320, wideY: 330, wideX: 150, slots: ['runner', 'support', 'runner', 'wide'], runTrigger: 'always', gridRadius: 600, gridProg: 0.7, counterpress: 1, compact: 1.0, homeWidth: 1.0, homeAdvance: 120 },
  };
  function styleOf(p, g) { return (g.styles && STYLES[g.styles[p.team]]) || STYLES.balanced; }

  function nearest(list, pt) {
    let best = null, bd = Infinity;
    for (const q of list) { const d = V.dist(q.pos, pt); if (d < bd) { bd = d; best = q; } }
    return { p: best, d: bd };
  }

  function laneClear(a, b, blockers, w) {
    return !blockers.some((o) => V.dist(o.pos, a) > 30 && V.segDist(o.pos, a, b) < o.r + w);
  }

  function predictBall(g, t) {
    const b = g.ball;
    const k = CFG.BALL_DRAG;
    const f = (1 - Math.exp(-k * t)) / k;
    return g.clampField(V.add(b.pos, V.mul(b.vel, f)), 20);
  }

  function bestPass(g, p, mates, opps, dir, minDist) {
    let best = null, bs = -Infinity;
    for (const m of mates) {
      const d = V.dist(p.pos, m.pos);
      if (d < minDist || d > 900) continue;
      const on = nearest(opps, m.pos).d;
      if (on < 75) continue;
      if (m.isKeeper && on < 200) continue;   // goleiro só como opção quando bem livre
      if (!laneClear(g.ball.pos, m.pos, opps, 26)) continue;
      const progress = (m.pos.x - p.pos.x) * dir;
      const score = progress + on * 0.8 - d * 0.25 + (m.callT > 0 ? 250 : 0);
      if (score > bs) { bs = score; best = m; }
    }
    return best;
  }

  // contexto compartilhado por decisão e execução
  function context(p, g) {
    const dir = p.team === 0 ? 1 : -1;
    const W2 = CFG.FIELD_W / 2;
    const mates = g.players.filter((q) => q.active && q.team === p.team && q !== p);
    const opps = g.players.filter((q) => q.active && q.team !== p.team);
    return {
      ball: g.ball, dir, W2,
      ownGoal: { x: -dir * W2, y: 0 }, oppGoal: { x: dir * W2, y: 0 },
      mates, opps, hasBall: g.ball.owner === p,
      firstTouch: !g.ball.owner && p.reach !== null && p.reach !== undefined,   // bola solta no alvo: ação de primeira possível
      teamHasKeeper: mates.some((m) => m.isKeeper),
    };
  }

  function homePos(p, g, c) {
    const S = styleOf(p, g);
    const oppBall = c.ball.owner && c.ball.owner.team !== p.team;
    // forma base do estilo (largura e altura) só com a posse ou bola solta; na defesa, forma normal
    const h = oppBall ? { x: p.home.x, y: p.home.y } : { x: p.home.x + c.dir * (S.homeAdvance || 0), y: p.home.y * (S.homeWidth || 1) };
    h.x += V.clamp(c.ball.pos.x * 0.45 * S.compact, -c.W2 * 0.4, c.W2 * 0.4);
    h.y += c.ball.pos.y * 0.35 * S.compact;
    if (c.ball.owner && c.ball.owner.team !== p.team) h.x -= c.dir * 150 * S.compact;
    // não entra na própria área se já existe goleiro (evita trocar de luvas)
    if (c.teamHasKeeper && Math.abs(h.y) < CFG.BOX_H / 2 + 30) {
      const edge = -c.dir * (c.W2 - CFG.BOX_W - 35);
      if ((h.x - edge) * c.dir < 0) h.x = edge;
    }
    return g.clampField(h, 40);
  }

  // quão livre está um ponto para receber: distância ao adversário mais próximo,
  // linha de passe limpa a partir da bola, e longe de companheiros (não aglomerar)
  // ganho de controle de campo (Voronoi) se eu estivesse em pt: células a até 260 px que passariam a ser minhas;
  // tomar do adversário vale cheio, redistribuir de companheiro vale pouco (não adianta ficar em cima dele)
  function controlGain(pt, p, g) {
    const pc = (typeof Features !== 'undefined' && Features.pitchControl) ? Features.pitchControl(g) : null;
    if (!pc) return 0;
    const GX = pc.cx.length, GY = pc.cy.length;
    let gain = 0;
    for (let j = 0; j < GY; j++) for (let i = 0; i < GX; i++) {
      const dx = pc.cx[i] - pt.x, dy = pc.cy[j] - pt.y;
      const d2 = dx * dx + dy * dy;
      if (d2 > 260 * 260) continue;
      const own = pc.owner[j * GX + i];
      if (own < 0 || own === p.id) continue;
      const o = g.players[own];
      const ox = o.pos.x - pc.cx[i], oy = o.pos.y - pc.cy[j];
      if (d2 < ox * ox + oy * oy) gain += (o.team !== p.team) ? 1 : 0.3;
    }
    return gain;
  }
  function openness(pt, c, p, g) {
    const dOpp = nearest(c.opps, pt).d;
    const lane = laneClear(c.ball.pos, pt, c.opps, 24) ? 1 : 0;
    const dMate = nearest(c.mates, pt).d;
    const gain = g ? controlGain(pt, p, g) : 0;
    return Math.min(dOpp, 260) + 120 * lane + Math.min(dMate, 260) * 0.8 - (dMate < 120 ? 150 : 0) + 45 * gain;
  }
  // procura, numa grade ao redor de um alvo nominal, o ponto mais livre dentro do campo
  function bestOpenPoint(nominal, c, p, g, radius) {
    let best = null, bs = -Infinity;
    for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) {
      let pt = g.clampField({ x: nominal.x + i * radius / 2, y: nominal.y + j * radius / 2 }, 45);
      // não entra na própria área (deixa o goleiro em paz)
      if (c.teamHasKeeper && g.inBoxPt(pt, p.team)) continue;
      const s = openness(pt, c, p, g) - 0.6 * V.dist(pt, nominal);   // o alvo nominal (papel do estilo) pesa; a abertura só ajusta
      if (s > bs) { bs = s; best = pt; }
    }
    return best || g.clampField(nominal, 45);
  }
  // alvo nominal relativo ao portador (ou à bola) para cada macro de apoio
  // borda lateral do bloco defensivo adversário (jogadores de linha) do lado side: maior y*side
  function blockEdge(c, side) {
    let e = -Infinity;
    for (const o of c.opps) if (!o.isKeeper) e = Math.max(e, o.pos.y * side);
    return e;
  }
  function supportNominal(kind, p, g, c) {
    const anchor = c.ball.owner ? c.ball.owner.pos : c.ball.pos;
    const dir = c.dir;
    const S = styleOf(p, g);
    if (kind === 'openfwd') return { x: anchor.x + dir * S.fwdDist, y: anchor.y * 0.5 + (p.pos.y >= anchor.y ? 1 : -1) * 90 };
    if (kind === 'openwide') {   // ponta: largura absoluta do estilo (ou fora do bloco adversário), à altura wideX do portador
      const side = p.pos.y >= anchor.y ? 1 : -1;
      const y = Math.max(S.wideY, Math.abs(anchor.y) * 0.3 + S.wideY * 0.7, blockEdge(c, side) + 100) * side;
      return g.clampField({ x: anchor.x + dir * S.wideX, y: V.clamp(y, -CFG.FIELD_H / 2 + 60, CFG.FIELD_H / 2 - 60) }, 60);
    }
    if (kind === 'overlap') {   // ultrapassagem por fora: à frente do portador, por fora do bloco
      const side = p.pos.y >= anchor.y ? 1 : -1;
      const y = Math.max(anchor.y * side + 300, blockEdge(c, side) + 100) * side;
      return { x: anchor.x + dir * 320, y: V.clamp(y, -CFG.FIELD_H / 2 + 70, CFG.FIELD_H / 2 - 70) };
    }
    if (kind === 'runspace') {  // atacar o espaço: além da última linha de defensores, num ponto livre
      const defenders = c.opps.filter((o) => !o.isKeeper);
      const lastX = defenders.length ? Math.max(...defenders.map((o) => o.pos.x * dir)) : anchor.x * dir + 200;
      const W2 = CFG.FIELD_W / 2;
      const x = dir * Math.min(lastX + 70, W2 - CFG.BOX_W * 0.7);
      return { x, y: V.clamp(p.pos.y * 0.7, -CFG.FIELD_H / 2 * 0.6, CFG.FIELD_H / 2 * 0.6) };
    }
    if (kind === 'runbox') {    // desmarcar na área adversária
      const W2 = CFG.FIELD_W / 2;
      return { x: dir * (W2 - CFG.BOX_W * 0.55), y: V.clamp(p.pos.y * 0.6 + (p.pos.y >= 0 ? 1 : -1) * 60, -CFG.BOX_H / 2 * 0.7, CFG.BOX_H / 2 * 0.7) };
    }
    // openback: apoio atrás (distâncias do estilo); no nosso campo fica mais ao lado do que atrás (o time precisa subir)
    const own = anchor.x * dir < 0;
    return { x: anchor.x - dir * (own ? S.supBack * 0.5 : S.supBack), y: anchor.y * 0.4 + (p.pos.y >= anchor.y ? 1 : -1) * (own ? S.supSide * 1.6 : S.supSide) };
  }

  // passe em profundidade: companheiro correndo para a frente; mira no ponto futuro dele
  function throughTarget(g, p, c) {
    let best = null, bs = -Infinity;
    for (const m of c.mates) {
      if (m.isKeeper) continue;
      const fwd = V.dot(m.vel, { x: c.dir, y: 0 });
      if (fwd < 60) continue;                                     // tem que estar correndo para a frente
      const lead = g.clampField(V.add(m.pos, V.mul(m.vel, 0.9)), 40);
      const d = V.dist(p.pos, lead);
      if (d < 200 || d > 900) continue;
      if ((lead.x - p.pos.x) * c.dir < 100) continue;
      if (!laneClear(c.ball.pos, lead, c.opps, 26)) continue;
      const on = nearest(c.opps, lead).d;
      if (on < 90) continue;
      const s = fwd + on * 0.5 + (lead.x - p.pos.x) * c.dir * 0.3;
      if (s > bs) { bs = s; best = { m, lead, d }; }
    }
    return best;
  }

  // inversão: companheiro do outro lado do campo, com linha limpa
  function switchTarget(g, p, c) {
    let best = null, bs = -Infinity;
    for (const m of c.mates) {
      if (m.isKeeper) continue;
      const dy = Math.abs(m.pos.y - p.pos.y), d = V.dist(p.pos, m.pos);
      if (dy < 300 || d > 1100) continue;
      if ((m.pos.x - p.pos.x) * c.dir < -150) continue;
      if (!laneClear(c.ball.pos, m.pos, c.opps, 26)) continue;
      const on = nearest(c.opps, m.pos).d;
      if (on < 90) continue;
      const s = on + dy * 0.3;
      if (s > bs) { bs = s; best = m; }
    }
    return best;
  }

  // espaço livre ao longo de um raio: distância até o primeiro adversário perto da linha
  function rayFree(from, dirv, opps, maxD) {
    let best = maxD;
    for (const o of opps) {
      const rel = V.sub(o.pos, from);
      const along = V.dot(rel, dirv);
      if (along <= 0 || along > maxD) continue;
      const lateral = Math.abs(rel.x * dirv.y - rel.y * dirv.x);
      if (lateral < 70) best = Math.min(best, along);
    }
    return best;
  }
  // direção com mais espaço para conduzir (raios em leque à frente, prefere o gol)
  function spaceDir(p, c) {
    const toGoal = V.norm(V.sub(c.oppGoal, p.pos));
    let best = null, bs = -Infinity;
    for (let k = -4; k <= 4; k++) {
      const d = V.rot(toGoal, k * 0.35);            // até ±80°: lateral também
      const fr = rayFree(p.pos, d, c.opps, 500);
      const s = fr - Math.abs(k) * 30;
      if (s > bs) { bs = s; best = d; }
    }
    return { dir: best, free: bs };
  }
  // melhor ponto para abrir: células da grade num raio, livres, com linha de passe e à frente
  function bestGridPoint(p, g, c, radius) {
    const anchor = c.ball.owner ? c.ball.owner.pos : c.ball.pos;
    let best = null, bs = -Infinity;
    const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2;
    for (let i = 0; i < 12; i++) for (let j = 0; j < 7; j++) {
      const pt = { x: -W2 + (i + 0.5) * CFG.FIELD_W / 12, y: -H2 + (j + 0.5) * CFG.FIELD_H / 7 };
      const dp = V.dist(pt, p.pos);
      if (dp > radius) continue;
      if (c.teamHasKeeper && g.inBoxPt(pt, p.team)) continue;
      const da = V.dist(pt, anchor);
      if (da < 120 || da > 800) continue;
      const s = openness(pt, c, p, g) + styleOf(p, g).gridProg * (pt.x - anchor.x) * c.dir - 0.15 * dp;   // progressão pesa conforme o estilo
      if (s > bs) { bs = s; best = pt; }
    }
    return best || homePos(p, g, c);
  }
  // adversário mais livre (para cortar a linha de passe)
  function freestOpp(p, c) {
    let best = null, bs = -Infinity;
    for (const o of c.opps) {
      if (o === c.ball.owner || o.isKeeper) continue;
      const on = nearest(c.mates.concat([p]), o.pos).d;
      const s = on + ((o.pos.x - c.ball.pos.x) * -c.dir) * 0.3;   // livre e avançado no ataque deles
      if (s > bs) { bs = s; best = o; }
    }
    return best;
  }

  function shotTargetFor(p, c) {
    const gk = c.opps.find((o) => o.isKeeper);
    const cornerY = (CFG.GOAL_W / 2 - 30) * (gk && gk.pos.y > 0 ? -1 : 1);
    return { x: c.oppGoal.x, y: cornerY };
  }

  // ---------- decisão tática do script ----------
  function chooseMacro(p, g, c) {
    const ai = p.ai;
    const ball = c.ball;
    if (p.isKeeper) return chooseKeeperMacro(p, g, c);
    if (c.hasBall || c.firstTouch) return chooseCarrierMacro(p, g, c);
    return chooseOffBallMacro(p, g, c);
  }

  // com a bola (no pé ou de primeira): as mesmas opções para todo mundo, goleiro incluído.
  // Cada opção (chute, passes, condução, proteger) recebe uma nota a partir dos observáveis;
  // vence a maior. Sem sorteios: o comportamento é determinístico e explicável.
  function laneMargin(a, b, opps) {
    let m = Infinity;
    for (const o of opps) { if (V.dist(o.pos, a) <= 30) continue; m = Math.min(m, V.segDist(o.pos, a, b)); }
    return V.clamp((m - 30) / 80, 0, 1);   // 0 = adversário em cima da linha, 1 = linha folgada (110 px)
  }
  function voronoiShare(g, q) {
    const pc = (typeof Features !== 'undefined' && Features.pitchControl) ? Features.pitchControl(g) : null;
    if (!pc) return 0.5;
    return V.clamp(pc.count[q.id] / (pc.owner.length / g.players.filter((x) => x.active).length), 0, 2) / 2;   // 0.5 = fatia média
  }
  // dono (Voronoi) da célula de um ponto: 1 se é do meu time, 0 se é do adversário
  function voronoiOurs(g, pt, team) {
    const pc = (typeof Features !== 'undefined' && Features.pitchControl) ? Features.pitchControl(g) : null;
    if (!pc) return 0.5;
    const GX = pc.cx.length, GY = pc.cy.length;
    const i = V.clamp(Math.floor((pt.x + CFG.FIELD_W / 2) / (CFG.FIELD_W / GX)), 0, GX - 1);
    const j = V.clamp(Math.floor((pt.y + CFG.FIELD_H / 2) / (CFG.FIELD_H / GY)), 0, GY - 1);
    const o = pc.owner[j * GX + i];
    return o >= 0 && g.players[o].team === team ? 1 : 0;
  }
  // nota de um passe para `recv` no ponto `tgt`
  function passValue(p, g, c, recv, tgt, pressure) {
    const S = styleOf(p, g);
    const prog = V.clamp(((tgt.x - p.pos.x) * c.dir) / 500, -0.5, 1);
    const space = Math.min(nearest(c.opps, tgt).d, 260) / 260;
    const margin = laneMargin(c.ball.pos, tgt, c.opps);
    const share = voronoiShare(g, recv);
    let v = S.prog * prog + S.space * space + S.margin * margin + S.share * share;
    // alvo em célula controlada por nós (Voronoi): passe para o espaço nosso
    v += 0.1 * voronoiOurs(g, tgt, p.team);
    // receptor no nosso terço defensivo com adversário perto: risco de perder atrás
    if (tgt.x * c.dir < -c.W2 * 0.5 && nearest(c.opps, tgt).d < 200) v -= 0.3;
    if (recv.callT > 0) v += 0.3;
    return v;
  }
  // cruzamento: da lateral do terço final, para um companheiro na área que esteja à frente dos
  // defensores naquela trajetória e longe do goleiro (ele não fecha o ângulo direto)
  function crossTarget(g, p, c) {
    const dir = c.dir, W2 = c.W2;
    if (Math.abs(p.pos.y) < CFG.FIELD_H * 0.18 || p.pos.x * dir < W2 * 0.15) return null;
    const gk = c.opps.find((o) => o.isKeeper);
    let best = null, bs = -Infinity;
    for (const m of c.mates) {
      if (m.isKeeper) continue;
      const lead = g.clampField(V.add(m.pos, V.mul(m.vel, 0.5)), 30);
      if (lead.x * dir < W2 - CFG.BOX_W * 1.4 || Math.abs(lead.y) > CFG.BOX_H / 2 + 60) continue;   // na área (ou quase)
      const d = V.dist(c.ball.pos, lead);
      if (d < 200 || d > 950) continue;
      if (gk && V.dist(lead, gk.pos) < 150) continue;
      const dv = V.norm(V.sub(lead, c.ball.pos));
      let blocked = false;
      for (const o of c.opps) {   // defensor na trajetória, antes do receptor
        if (o.isKeeper) continue;
        const rel = V.sub(o.pos, c.ball.pos);
        const along = V.dot(rel, dv);
        if (along > 30 && along < d - 40 && Math.abs(rel.x * dv.y - rel.y * dv.x) < 36) { blocked = true; break; }
      }
      if (blocked) continue;
      const dGk = gk ? V.dist(lead, gk.pos) : 400;
      const v = 0.45 + 0.3 * Math.min(nearest(c.opps, lead).d, 150) / 150 + 0.25 * V.clamp((dGk - 150) / 200, 0, 1) + (m.vel.x * dir > 40 ? 0.1 : 0);
      if (v > bs) { bs = v; best = { m, lead, d, value: v }; }
    }
    return best;
  }
  function chooseCarrierMacro(p, g, c) {
    const ai = p.ai;
    const ball = c.ball;
    const dir = c.dir;
    const S = styleOf(p, g);
    if (ai.mode === 'shoot') return KICK_MACROS.includes(ai.macro) ? ai.macro : 'shoot';
    if (ai.mode === 'pass') return KICK_MACROS.includes(ai.macro) ? ai.macro : 'pass';
    const dGoal = V.dist(p.pos, c.oppGoal);
    const no = nearest(c.opps, p.pos);
    const dOpp = no.d;
    // de primeira é opcional: só compensa perto do gol (não dá tempo de dominar) ou sob pressão
    if (!c.hasBall && !(dGoal < 520 || dOpp < 130)) return 'chase';
    const pressure = V.clamp(1 - dOpp / 220, 0, 1);
    const opts = [];
    // ---- chute: distância, ângulo, linha livre, goleiro fora da linha ----
    {
      const target = shotTargetFor(p, c);
      const lane = laneClear(ball.pos, target, c.opps.filter((o) => !o.isKeeper), 30);
      const angle = 1 - Math.min(1, Math.abs(p.pos.y) / (CFG.FIELD_H * 0.42));
      const distF = dGoal < 300 ? 1 : dGoal < 550 ? 0.75 : dGoal < 800 ? 0.4 : 0;
      const gk = c.opps.find((o) => o.isKeeper);
      const gkOut = (gk ? V.clamp((Math.abs(gk.pos.x - c.oppGoal.x) - 120) / 200, 0, 1) : 1) * (dGoal < 900 ? 0.3 : 0);
      // de primeira só compensa bem perto (longe, domina e decide com a bola no pé)
      const shot = (distF * (0.5 + 0.5 * angle) * (lane ? 1 : 0.35) + gkOut) * (c.hasBall || dGoal < 300 ? 1 : 0.3);
      const gkD = gk ? V.dist(gk.pos, p.pos) : 9999;
      const quick = dOpp < 90 || gkD < 200 || !c.hasBall;   // pressionado, goleiro em cima ou de primeira: sai rápido; sozinho: carrega
      opts.push({ macro: quick ? 'shootq' : 'shoot', score: shot });
    }
    // ---- passes: enfiada, lançamento, inversão, passe, recuo ----
    const thr = throughTarget(g, p, c);
    if (thr) opts.push({ macro: 'through', score: passValue(p, g, c, thr.m, thr.lead, pressure) + S.through - (thr.d > 800 ? 0.1 : 0) });
    const lp = longPassTarget(g, p, c);
    if (lp) opts.push({ macro: 'longpass', score: passValue(p, g, c, lp, V.add(lp.pos, V.mul(lp.vel, 0.6)), pressure) + S.long });
    const cr = crossTarget(g, p, c);
    if (cr) opts.push({ macro: 'cross', score: cr.value + S.cross });
    const sw = switchTarget(g, p, c);
    if (sw) {
      const crowded = c.opps.filter((o) => V.dist(o.pos, p.pos) < 260).length >= 2 ? 0.15 : 0;
      opts.push({ macro: 'switch', score: passValue(p, g, c, sw, sw.pos, pressure) + crowded + S.switch });
    }
    const bp = bestPass(g, p, c.mates, c.opps, dir, 100);
    if (bp) opts.push({ macro: 'pass', score: passValue(p, g, c, bp, V.add(bp.pos, V.mul(bp.vel, 0.3)), pressure) });
    const sp = safePassTarget(g, p, c);
    if (sp) opts.push({ macro: 'passback', score: passValue(p, g, c, sp, sp.pos, pressure) + S.passback + 0.4 * Math.max(0, pressure - 0.3) });
    // ---- conduzir / proteger ----
    if (c.hasBall) {
      const sd = spaceDir(p, c);
      const ownHalf = p.pos.x * dir < 0 ? 0.12 : 0;   // no nosso campo, avançar vale mais
      const carry = V.clamp(sd.free / 500, 0, 1) * S.carry * (1 - pressure) * (p.isKeeper ? 0.3 : 1) + (dGoal < 900 ? 0.1 : 0) + ownHalf;
      // conduzir calmo por padrão; arrancar para o espaço com campo à frente e ninguém perto
      opts.push({ macro: (sd.free > S.carrySpace && dOpp > 150) ? 'carryspace' : 'dribble', score: carry });
      opts.push({ macro: 'hold', score: S.hold + 0.35 * pressure * (dOpp < 70 ? 1 : 0) });
    } else {
      opts.push({ macro: 'chase', score: 0.3 });   // bola no alvo mas sem chute/passe bom: domina
    }
    let best = opts[0];
    for (const o of opts) if (o.score > best.score) best = o;
    return best.macro;
  }

  function chooseOffBallMacro(p, g, c) {
    const ai = p.ai;
    const ball = c.ball;
    // goleiro fora da área (ou sem goleiro) e eu sou o mais perto do gol: assumo o gol
    const keeper = c.mates.find((m) => m.isKeeper);
    if ((!keeper || !g.inOwnBox(keeper)) && ball.pos.x * c.dir < c.W2 * 0.2) {
      const closest = c.mates.filter((m) => !m.isKeeper).concat([p]).sort((a, b) => V.dist(a.pos, c.ownGoal) - V.dist(b.pos, c.ownGoal))[0];
      if (closest === p && !(ball.owner && ball.owner === p)) return 'guardgoal';
    }
    // companheiro com a bola: papéis pela estrutura do time (determinístico, igual para todos):
    // apoio atrás (o mais perto do portador que não está à frente), largura em cada lado quando
    // falta, um atacante da profundidade / área, e os demais no ponto mais livre da grade.
    if (ball.owner && ball.owner.team === p.team) {
      const carrier = ball.owner;
      const dir = c.dir;
      const field = c.mates.filter((m) => !m.isKeeper && m !== carrier).concat([p]);
      const S = styleOf(p, g);
      const roles = new Map();
      const free = () => field.filter((m) => !roles.has(m));
      const attacking = carrier.pos.x * dir > c.W2 * 0.3;   // bola no terço de ataque
      const defenders = c.opps.filter((o) => !o.isKeeper);
      const lastX = defenders.length ? Math.max(...defenders.map((o) => o.pos.x * dir)) : c.W2;
      // portador aberto na lateral do campo de ataque: o corredor ataca a área para o cruzamento
      const wideCarrier = Math.abs(carrier.pos.y) >= CFG.FIELD_H * 0.18 && carrier.pos.x * dir >= c.W2 * 0.15;
      const gap = lastX - carrier.pos.x * dir;
      const canRun = S.runTrigger === 'always' ? gap > 60 : S.runTrigger === 'space' ? gap > 100 : false;   // 'final': só na área
      const runnerRole = (attacking || wideCarrier) ? 'runbox' : (canRun ? 'runspace' : 'openfwd');
      const wideTaken = new Set();
      // vagas na ordem do estilo; cada vaga escolhe o jogador livre mais adequado
      for (const slot of S.slots) {
        const fr = free();
        if (!fr.length) break;
        if (slot === 'support') {   // apoio: o mais perto do portador entre os que não estão à frente (senão o mais perto)
          const behind = fr.filter((m) => (m.pos.x - carrier.pos.x) * dir <= 40);
          const list = (behind.length ? behind : fr).slice().sort((a, b) => V.dist(a.pos, carrier.pos) - V.dist(b.pos, carrier.pos));
          roles.set(list[0], attacking ? 'overlap' : 'openback');
        } else if (slot === 'wide') {   // ponta: lado ainda sem ponta; o jogador livre mais lateral desse lado
          const sides = [1, -1].filter((sd) => !wideTaken.has(sd)).sort((a, b) => {
            const oa = field.some((m) => m.pos.y * a > S.wideY - 80), ob = field.some((m) => m.pos.y * b > S.wideY - 80);
            return (oa ? 1 : 0) - (ob ? 1 : 0);   // primeiro o lado sem ninguém aberto
          });
          if (!sides.length) continue;
          const side = sides[0];
          const cand = fr.slice().sort((a, b) => (b.pos.y - a.pos.y) * side);
          roles.set(cand[0], 'openwide'); wideTaken.add(side);
        } else if (slot === 'fwd') {   // opção curta à frente: o mais perto do portador entre os que estão à frente (senão o mais avançado)
          const ahead = fr.filter((m) => (m.pos.x - carrier.pos.x) * dir > 40);
          const list = (ahead.length ? ahead : fr).slice().sort((a, b) => V.dist(a.pos, carrier.pos) - V.dist(b.pos, carrier.pos));
          roles.set(list[0], 'openfwd');
        } else if (slot === 'runner') {   // profundidade: o mais avançado
          const adv = fr.slice().sort((a, b) => (b.pos.x - a.pos.x) * dir);
          roles.set(adv[0], runnerRole);
        }
      }
      let role = roles.get(p) || 'openbest';
      // oportunidade acima do papel: à frente da bola com espaço atrás da linha e ninguém já correndo -> ataca o espaço
      if (!['runspace', 'runbox', 'openback', 'overlap'].includes(role) && canRun && (p.pos.x - carrier.pos.x) * dir > 0 &&
          !Array.from(roles.values()).some((r) => r === 'runspace' || r === 'runbox')) role = 'runspace';
      // aglomeração: companheiro (que não é o portador) a menos de 110 px e não sou o apoio -> abrir no ponto mais livre
      if (role !== 'openback' && c.mates.some((m) => m !== carrier && V.dist(m.pos, p.pos) < 110)) role = 'openbest';
      return role;
    }
    if (ball.owner) {
      // defesa: o mais perto pressiona; o segundo corta a linha mais perigosa (ou cobre);
      // o último homem cobre; os outros compactam (home puxado para a bola)
      const chasers = c.mates.filter((m) => !m.isKeeper).concat([p]);
      const order = chasers.slice().sort((a, b) => V.dist(a.pos, ball.owner.pos) - V.dist(b.pos, ball.owner.pos)).indexOf(p);
      if (order === 0) return 'defend';
      if (order === 1 && styleOf(p, g).counterpress >= 2 && ball.owner.pos.x * c.dir > 0) return 'defend';   // contra-pressão: perdeu alta, dois vão
      const lastMan = chasers.slice().sort((a, b) => V.dist(a.pos, c.ownGoal) - V.dist(b.pos, c.ownGoal))[0] === p;
      if (order === 1) {
        const fo = freestOpp(p, c);
        const dangerous = fo && (V.dist(fo.pos, c.ownGoal) < V.dist(ball.pos, c.ownGoal) + 100 || V.dist(fo.pos, c.ownGoal) < 600);
        return dangerous && !lastMan ? 'cutlane' : 'cover';
      }
      return lastMan ? 'cover' : 'home';
    }
    // bola solta: só o mais próximo vai; o segundo cobre; o resto se posiciona
    const dBall = V.dist(p.pos, ball.pos);
    const pred = predictBall(g, V.clamp(dBall / 450, 0, 1.2));
    const rank = c.mates.filter((m) => !m.isKeeper && V.dist(m.pos, pred) < V.dist(p.pos, pred)).length;
    if (rank === 0) return 'chase';
    if (rank === 1) return (pred.x - p.pos.x) * c.dir > 0 ? 'cover' : 'openfwd';
    return 'home';
  }

  // passe longo (com chute carregado): companheiro bem à frente, livre, com linha limpa
  function longPassTarget(g, p, c) {
    let best = null, bs = -Infinity;
    for (const m of c.mates) {
      if (m.isKeeper) continue;
      const d = V.dist(p.pos, m.pos);
      if (d < 450 || d > 1500) continue;
      if ((m.pos.x - p.pos.x) * c.dir < 150) continue;        // tem que ser para a frente
      const lead = V.add(m.pos, V.mul(m.vel, 0.6));
      if (!laneClear(c.ball.pos, lead, c.opps, 24)) continue;
      const on = nearest(c.opps, lead).d;
      if (on < 85) continue;
      const s = (m.pos.x - p.pos.x) * c.dir + on * 0.5;
      if (s > bs) { bs = s; best = m; }
    }
    return best;
  }

  // passe seguro: companheiro atrás (ou ao lado) com linha limpa e sem marcador perto
  function safePassTarget(g, p, c) {
    let best = null, bs = -Infinity;
    for (const m of c.mates) {
      const d = V.dist(p.pos, m.pos);
      if (d < 80 || d > 700) continue;
      if ((m.pos.x - p.pos.x) * c.dir > 60) continue;          // não é "para trás"
      if (!laneClear(c.ball.pos, m.pos, c.opps, 24)) continue;
      const on = nearest(c.opps, m.pos).d;
      if (on < 90) continue;
      const s = on - d * 0.2;
      if (s > bs) { bs = s; best = m; }
    }
    return best;
  }

  // ---------- execução ----------
  function execute(p, g, dt, c, macro, inp) {
    const ai = p.ai;
    const ball = c.ball;
    const dir = c.dir;
    const moveTo = (pt, sprint) => {
      const d = V.sub(pt, p.pos), l = V.len(d);
      if (l > 8) { const n = V.norm(d); inp.mx = n.x; inp.my = n.y; }
      inp.sprint = !!sprint && l > 40;
    };
    inp.aim = { x: ball.pos.x, y: ball.pos.y };

    const first = !c.hasBall && c.firstTouch && KICK_MACROS.includes(macro);   // ação de primeira (trava no alvo)
    if (c.hasBall || first) {
      const chargeOf = () => p.charge || (p.queued && p.queued.kind === 'shot' ? p.queued : null);   // carga real ou da trava
      // sem a bola essas decisões não fazem sentido; com a bola as outras viram "conduzir"
      if (!['shoot', 'shootq', 'pass', 'passback', 'longpass', 'through', 'switch', 'cross', 'dribble', 'carryspace', 'hold'].includes(macro)) macro = 'dribble';
      if (ai.mode === 'shoot' && !['shoot', 'shootq', 'longpass', 'through', 'cross'].includes(macro)) ai.mode = 'none';
      if (ai.mode === 'pass' && macro !== 'pass' && macro !== 'passback' && macro !== 'switch') ai.mode = 'none';

      if (macro === 'longpass') {
        if (ai.mode !== 'shoot') {
          const t = longPassTarget(g, p, c);
          if (!t) macro = 'dribble';
          else {
            const lead = V.add(t.pos, V.mul(t.vel, 0.6));
            const d = V.dist(p.pos, lead);
            ai.mode = 'shoot'; ai.modeT = ai.t; ai.aim = lead;
            ai.chargeT = V.clamp((d - 300) / 1200, 0.3, 0.85) * CFG.CHARGE_MAX;   // força pela distância, sem estourar
          }
        }
        if (ai.mode === 'shoot') {
          inp.aim = ai.aim; inp.shoot = true;
          const ch = chargeOf();
          if (ch && ch.t >= ai.chargeT) { inp.shoot = false; ai.mode = 'none'; }
          else if (!ch && ai.t - ai.modeT > 0.2) ai.mode = 'none';
          return inp;
        }
      }

      if (macro === 'cross') {
        // cruzamento: chute forte para o ponto futuro do companheiro na área, com curva para dentro
        if (ai.mode !== 'shoot') {
          const t = crossTarget(g, p, c);
          if (!t) macro = 'dribble';
          else {
            ai.mode = 'shoot'; ai.modeT = ai.t; ai.aim = t.lead;
            ai.chargeT = V.clamp((t.d - 200) / 1000, 0.45, 0.9) * CFG.CHARGE_MAX;
            const d0 = V.norm(V.sub(t.lead, ball.pos));
            const toCenter = V.sub({ x: c.oppGoal.x, y: 0 }, t.lead);
            ai.crossSpin = V.dot(toCenter, V.perp(d0)) >= 0 ? 1 : -1;   // curva para o lado do gol
          }
        }
        if (ai.mode === 'shoot') {
          inp.aim = ai.aim; inp.shoot = true;
          const d0 = V.norm(V.sub(ai.aim, ball.pos));
          inp.drag = V.mul(V.perp(d0), 5 * ai.crossSpin);   // arrasto lateral constante = efeito acumulado durante a carga
          const ch = chargeOf();
          if (ch && ch.t >= ai.chargeT) { inp.shoot = false; ai.mode = 'none'; }
          else if (!ch && ai.t - ai.modeT > 0.2) ai.mode = 'none';
          return inp;
        }
      }

      if (macro === 'through') {
        // passe em profundidade: chute fraco/médio mirado no ponto futuro (o chute não tem assistência de mira)
        if (ai.mode !== 'shoot') {
          const t = throughTarget(g, p, c);
          if (!t) macro = 'dribble';
          else {
            ai.mode = 'shoot'; ai.modeT = ai.t; ai.aim = t.lead;
            ai.chargeT = V.clamp((t.d - 150) / 1400, 0.05, 0.6) * CFG.CHARGE_MAX;
          }
        }
        if (ai.mode === 'shoot') {
          inp.aim = ai.aim; inp.shoot = true;
          const ch = chargeOf();
          if (ch && ch.t >= ai.chargeT) { inp.shoot = false; ai.mode = 'none'; }
          else if (!ch && ai.t - ai.modeT > 0.2) ai.mode = 'none';
          return inp;
        }
      }

      if (macro === 'switch') {
        if (ai.mode !== 'pass') {
          const t = switchTarget(g, p, c);
          if (!t) macro = 'dribble';
          else { ai.mode = 'pass'; ai.aim = V.add(t.pos, V.mul(t.vel, 0.35)); ai.holdN = 3 + Math.floor(V.dist(p.pos, t.pos) / 220); }
        }
        if (ai.mode === 'pass') {
          inp.aim = ai.aim; inp.pass = ai.holdN-- > 0;
          if (ai.holdN < 0) ai.mode = 'none';
          return inp;
        }
      }

      if (macro === 'passback') {
        if (ai.mode !== 'pass') {
          const t = safePassTarget(g, p, c);
          if (!t) macro = 'hold';
          else { ai.mode = 'pass'; ai.aim = V.add(t.pos, V.mul(t.vel, 0.3)); ai.holdN = 2 + Math.floor(V.dist(p.pos, t.pos) / 250); }
        }
        if (ai.mode === 'pass') {
          inp.aim = ai.aim; inp.pass = ai.holdN-- > 0;
          if (ai.holdN < 0) ai.mode = 'none';
          return inp;
        }
      }
      if (macro === 'shoot' || macro === 'shootq') {
        if (ai.mode !== 'shoot') {
          const dGoal = V.dist(p.pos, c.oppGoal);
          ai.mode = 'shoot'; ai.modeT = ai.t; ai.aim = shotTargetFor(p, c);
          // carregado: forte por padrão (de 300 px em diante, carga cheia); rápido: sai logo
          ai.chargeT = (macro === 'shootq' ? 0.4 : V.clamp(0.5 + dGoal / 600, 0.6, 1)) * CFG.CHARGE_MAX;
        }
        inp.aim = ai.aim; inp.shoot = true;
        if (c.hasBall) moveTo(c.oppGoal, false);
        const ch = chargeOf();
        if (ch && ch.t >= ai.chargeT) { inp.shoot = false; ai.mode = 'none'; }
        else if (!ch && ai.t - ai.modeT > 0.2) ai.mode = 'none';
        return inp;
      }
      // de primeira sem alvo válido: domina a bola (ir na bola)
      if (!c.hasBall) { ai.mode = 'none'; return execute(p, g, dt, c, 'chase', inp); }

      if (macro === 'carryspace') {
        // conduz para a direção com mais espaço, correndo e empurrando a bola
        const sp = spaceDir(p, c);
        const d = sp.dir || V.norm(V.sub(c.oppGoal, p.pos));
        inp.mx = d.x; inp.my = d.y;
        inp.aim = V.add(p.pos, V.mul(d, 150));
        inp.sprint = p.stamina > 8 && sp.free > 200;
        if (p.cd.grab <= 0 && sp.free > 350 && V.len(p.vel) > 100 && ai.t - (ai.lastPush || -9) > 0.6) { inp.special = true; ai.lastPush = ai.t; }   // push ball só com muito espaço, sem repetir
        return inp;
      }

      if (macro === 'hold') {
        // protege a bola: postura de drible, de costas para o marcador, andando devagar para longe dele
        const no = nearest(c.opps, p.pos);
        const away = no.p ? V.norm(V.sub(p.pos, no.p.pos)) : { x: -dir, y: 0 };
        const toGoal = V.norm(V.sub(c.oppGoal, p.pos));
        const st = V.norm(V.add(V.mul(away, 0.7), V.mul(toGoal, 0.3)));
        inp.mx = st.x * 0.6; inp.my = st.y * 0.6;
        inp.stance = true;
        inp.aim = V.add(p.pos, V.mul(toGoal, 150));
        if (no.d < 60 && p.cd.dribble <= 0 && g.rng() < 0.08) inp.special = true;   // drible para escapar
        return inp;
      }

      if (macro === 'pass') {
        if (ai.mode !== 'pass') {
          // pedido de bola tem prioridade: passa para quem pediu, ou chutão se longe
          const caller = c.mates.find((m) => m.callT > 0 && !(ai.lastCall && ai.t - ai.lastCall < 2.5));
          if (caller) {
            ai.lastCall = ai.t;
            const target = V.add(caller.pos, V.mul(caller.vel, 0.3));
            const dCall = V.dist(p.pos, target);
            if (dCall < 750 && laneClear(ball.pos, target, c.opps, 22)) {
              ai.mode = 'pass'; ai.aim = target; ai.holdN = 2 + Math.floor(dCall / 250);
            } else {
              ai.mode = 'shoot'; ai.modeT = ai.t; ai.aim = target;
              ai.chargeT = V.clamp(dCall / 1500, 0.35, 1) * CFG.CHARGE_MAX;
              inp.aim = ai.aim; inp.shoot = true;
              return inp;
            }
          } else {
            const cand = bestPass(g, p, c.mates, c.opps, dir, 100);
            if (!cand) { macro = 'dribble'; }
            else {
              ai.mode = 'pass';
              ai.aim = V.add(cand.pos, V.mul(cand.vel, 0.3));
              ai.holdN = 2 + Math.floor(V.dist(p.pos, cand.pos) / 250);
            }
          }
        }
        if (ai.mode === 'pass') {
          inp.aim = ai.aim; inp.pass = ai.holdN-- > 0;
          if (ai.holdN < 0) ai.mode = 'none';
          return inp;
        }
      }

      // conduzir em direção ao gol, desviando do marcador
      const no = nearest(c.opps, p.pos);
      const dOpp = no.d;
      let steer = V.norm(V.sub({ x: c.oppGoal.x - dir * 100, y: p.pos.y * 0.4 }, p.pos));
      if (no.p && dOpp < 170) {
        const toOpp = V.sub(no.p.pos, p.pos);
        if (V.dot(V.norm(toOpp), steer) > 0.2) {
          const side = V.perp(steer);
          const s = V.dot(side, toOpp) > 0 ? -1 : 1;
          steer = V.norm(V.add(steer, V.mul(side, s * (dOpp < 90 ? 1.5 : 0.8))));
        }
      }
      inp.mx = steer.x; inp.my = steer.y;
      inp.aim = V.add(p.pos, V.mul(steer, 120));
      // condução calma: bola no pé; sem push (empurrar é a decisão "carryspace")
      const clearAhead = !c.opps.some((o) => {
        const d = V.sub(o.pos, p.pos), l = V.len(d);
        return l < 300 && V.dot(V.norm(d), steer) > 0.5;
      });
      if (clearAhead && p.stamina > 10 && V.dist(p.pos, c.oppGoal) > 600) inp.sprint = true;   // corre com a bola no pé
      if (dOpp < 80) {
        inp.stance = true;                                // postura de drible
        if (p.cd.dribble <= 0 && g.rng() < 0.06) inp.special = true;
      }
      return inp;
    }

    // ---------- sem a bola ----------
    ai.mode = 'none';
    if (['shoot', 'shootq', 'pass', 'passback', 'longpass', 'through', 'switch', 'cross', 'dribble', 'carryspace', 'hold'].includes(macro)) macro = 'chase';
    if (GK_MACROS.includes(macro)) macro = 'cover';   // decisões de goleiro num jogador de linha: cobrir
    // desmarcar na área só faz sentido com a bola no campo de ataque
    if (macro === 'runbox' && ball.pos.x * dir < 0) macro = 'openfwd';
    if (macro === 'defend' && !(ball.owner && ball.owner.team !== p.team)) macro = ball.owner ? 'openback' : 'chase';

    // abrir espaço para receber: ponto livre perto de um alvo nominal relativo ao portador/bola
    // abrir no melhor ponto da grade (Voronoi): livre, com linha de passe, à frente
    if (macro === 'openbest') {
      const target = bestGridPoint(p, g, c, styleOf(p, g).gridRadius);
      moveTo(target, V.dist(p.pos, target) > 220);
      inp.aim = ball.pos;
      return inp;
    }
    // cortar a linha de passe: entre o portador e o adversário mais livre
    if (macro === 'cutlane') {
      const fo = freestOpp(p, c);
      if (!fo || !ball.owner || ball.owner.team === p.team) return execute(p, g, dt, c, 'cover', inp);
      const target = V.lerp(ball.owner.pos, fo.pos, 0.55);
      moveTo(target, V.dist(p.pos, target) > 160);
      inp.aim = ball.pos;
      if (V.dist(p.pos, target) < 60) inp.stance = true;   // pronto para interceptar
      return inp;
    }
    // assumir o gol: corre para a frente do próprio gol (dentro da área, recebe as luvas)
    if (macro === 'guardgoal') {
      const target = { x: c.ownGoal.x + dir * 70, y: V.clamp(ball.pos.y * 0.3, -CFG.GOAL_W / 2 * 0.6, CFG.GOAL_W / 2 * 0.6) };
      moveTo(target, true);
      inp.aim = ball.pos;
      if (V.dist(p.pos, target) < 60) inp.stance = true;
      return inp;
    }

    if (['openfwd', 'openwide', 'openback', 'overlap', 'runbox', 'runspace'].includes(macro)) {
      const target = bestOpenPoint(supportNominal(macro, p, g, c), c, p, g, (macro === 'runbox' || macro === 'openwide') ? 100 : 140);
      moveTo(target, (macro === 'overlap' || macro === 'runspace') ? V.dist(p.pos, target) > 120 : (macro === 'openwide' ? V.dist(p.pos, target) > 150 : V.dist(p.pos, target) > 260));
      inp.aim = ball.pos;                       // de frente para a bola, pronto para o toque de primeira
      return inp;
    }

    // cobrir: fechar o espaço entre a bola e o próprio gol, marcando o adversário livre mais perigoso
    if (macro === 'cover') {
      // bola solta indo para o meu gol: interceptar a trajetória (bloquear o chute)
      const towardGoal = !ball.owner && ball.vel.x * dir < -150 && V.len(ball.vel) > 200;
      if (towardGoal) {
        const dirB = V.norm(ball.vel);
        const t = V.clamp(V.dot(V.sub(p.pos, ball.pos), dirB) / Math.max(1, V.len(ball.vel)), 0.05, 1.2);
        const pt = predictBall(g, t);
        moveTo(pt, true);
        inp.aim = ball.pos; inp.stance = true;
        return inp;
      }
      const danger = c.opps.filter((o) => !o.isKeeper && o !== ball.owner)
        .sort((a, b) => V.dist(a.pos, c.ownGoal) - V.dist(b.pos, c.ownGoal))[0];
      let target;
      if (danger && V.dist(danger.pos, c.ownGoal) < V.dist(ball.pos, c.ownGoal) + 150) {
        target = V.add(danger.pos, V.mul(V.norm(V.sub(c.ownGoal, danger.pos)), 45));   // entre ele e o gol
      } else {
        target = V.lerp(ball.pos, c.ownGoal, 0.35);                                      // linha bola-gol
      }
      target = g.clampField(target, 40);
      if (c.teamHasKeeper && g.inBoxPt(target, p.team)) target.x = -dir * (c.W2 - CFG.BOX_W - 40);
      moveTo(target, V.dist(p.pos, target) > 200);
      inp.aim = ball.pos;
      if (V.dist(p.pos, ball.pos) < 160) inp.stance = true;   // pronto para interceptar
      return inp;
    }

    if (macro === 'defend') {
      const carrier = ball.owner;
      const goalSide = V.norm(V.sub(c.ownGoal, carrier.pos));
      const intercept = V.add(carrier.pos, V.mul(goalSide, 26));
      const d = V.dist(p.pos, ball.pos);
      moveTo(intercept, d > 150);
      inp.aim = ball.pos;
      if (d < 150) inp.stance = true;
      const toBall = V.norm(V.sub(ball.pos, p.pos));
      if (d < 54 && p.cd.tackle <= 0 && V.dot(toBall, p.moveDir) > 0.2) {
        inp.tackle = true; inp.sprint = false;
      } else if (d > 70 && d < 135 && p.cd.slide <= 0 && p.stamina > 18 &&
        V.dot(carrier.vel, V.sub(carrier.pos, p.pos)) > 40 && V.len(carrier.vel) > 130 && g.rng() < 0.15) {
        inp.sprint = true; inp.tackle = true; inp.stance = false;
      }
      return inp;
    }

    if (macro === 'chase') {
      if (ball.owner) {   // bola com alguém: ir na bola = marcar/apoiar conforme o dono
        return execute(p, g, dt, c, ball.owner.team === p.team ? 'openback' : 'defend', inp);
      }
      const dBall = V.dist(p.pos, ball.pos);
      const pred = predictBall(g, V.clamp(dBall / 450, 0, 1.2));
      moveTo(pred, dBall > 90);
      inp.aim = ball.pos;
      const bs = V.len(ball.vel);
      if (bs > CFG.CONTROL_MAX * 0.85 && dBall < 150 && V.dot(ball.vel, V.sub(p.pos, ball.pos)) > 0) inp.stance = true;
      return inp;
    }

    // home
    moveTo(homePos(p, g, c), false);
    return inp;
  }

  // ---------- reflexos: regras fixas que sobrescrevem a rede em situações críticas ----------
  // A rede decide o jogo; aqui só entram situações que não pedem adaptação e nas quais errar custa caro.
  // Devolve a macro a executar (a da rede, ou a do reflexo).
  function reflex(p, g, c, macro) {
    const ball = c.ball, dir = c.dir;
    const hasBall = c.hasBall;
    if (p.isKeeper && !hasBall) {
      const inBox = g.inBoxPt(ball.pos, p.team);
      if (!ball.owner && inBox) {   // bola solta na minha área: se sou o mais perto (ou está muito perto), vou buscar
        const myD = V.dist(p.pos, ball.pos), oppD = nearest(c.opps, ball.pos).d;
        if (myD < oppD - 10 || myD < 90) return 'gk_rush';
      }
      if (ball.owner && ball.owner.team !== p.team && inBox && V.dist(p.pos, ball.pos) < 160) return 'gk_rush';   // atacante com a bola na área: sair
      return macro;
    }
    if (hasBall || (c.firstTouch && KICK_MACROS.includes(macro))) {
      if (p.held) return macro;
      const dGoal = V.dist(p.pos, c.oppGoal);
      const dOpp = nearest(c.opps, p.pos).d;
      const gk = c.opps.find((o) => o.isKeeper);
      const target = shotTargetFor(p, c);
      const lane = laneClear(ball.pos, target, c.opps.filter((o) => !o.isKeeper), 30);
      const angle = 1 - Math.min(1, Math.abs(p.pos.y) / (CFG.FIELD_H * 0.42));
      // chance clara: perto, ângulo aberto e linha livre (ou goleiro fora da linha) -> finaliza
      const gkOut = gk ? Math.abs(gk.pos.x - c.oppGoal.x) > 220 : true;
      if (lane && ((dGoal < 320 && angle > 0.4) || (dGoal < 520 && angle > 0.6 && gkOut)) && !['shoot', 'shootq', 'cross'].includes(macro)) {
        const gkD = gk ? V.dist(gk.pos, p.pos) : 9999;
        return (dOpp < 90 || gkD < 200 || !hasBall) ? 'shootq' : 'shoot';
      }
      // cruzamento muito claro para uma rede que não conhece a macro (28 saídas)
      if (hasBall && macro !== 'cross') { const cr = crossTarget(g, p, c); if (cr && cr.value >= 0.9) return 'cross'; }
      return macro;
    }
    // sem a bola
    if (!ball.owner) {   // bola solta: quem é claramente o mais perto do time vai na bola (alguém tem que ir)
      const dBall = V.dist(p.pos, ball.pos);
      const pred = predictBall(g, V.clamp(dBall / 450, 0, 1.2));
      const mine = V.dist(p.pos, pred);
      const others = c.mates.filter((m) => !m.isKeeper);
      if (!p.isKeeper && others.every((m) => V.dist(m.pos, pred) > mine + 60) && mine < 350 && !['chase', 'gk_rush'].includes(macro)) return 'chase';
    }
    // aglomeração: companheiro de linha (não portador) a menos de 120 px fazendo o mesmo -> o mais longe da bola abre no ponto mais livre
    if (!p.isKeeper && !['chase', 'defend', 'gk_rush', 'guardgoal'].includes(macro)) {
      const dB = V.dist(p.pos, ball.pos);
      const twin = c.mates.find((m) => !m.isKeeper && m !== ball.owner && V.dist(m.pos, p.pos) < 120 && (m.ai.macro === macro || V.dist(m.pos, ball.pos) < dB));
      if (twin && V.dist(twin.pos, ball.pos) <= dB) return 'openbest';
    }
    if (ball.owner && ball.owner.team !== p.team && !p.isKeeper) {   // último homem não sobe com o adversário atacando
      const field = c.mates.filter((m) => !m.isKeeper).concat([p]);
      const lastMan = field.slice().sort((a, b) => V.dist(a.pos, c.ownGoal) - V.dist(b.pos, c.ownGoal))[0] === p;
      if (lastMan && ['runspace', 'runbox', 'overlap', 'openfwd', 'openwide', 'openbest', 'openback', 'home'].includes(macro)) return 'cover';
    }
    return macro;
  }

  // ---------- tempo de reação + decisão ----------
  // macroFn (opcional): (p, g, ctx) -> macro. Sem ele, usa a decisão do script.
  // tempo de reação (todos os bots, rede incluída): após mudar o dono da bola, repete o último
  // input por BOT_REACTION s. Devolve o input repetido, ou null se é hora de decidir.
  function reactionHold(p, g, dt) {
    const ai = p.ai;
    const ownerId = g.ball.owner ? g.ball.owner.id : -1;
    if (ai.lastOwner === undefined) ai.lastOwner = ownerId;
    if (ownerId !== ai.lastOwner) {
      ai.lastOwner = ownerId;
      const base = p.isKeeper ? CFG.BOT_REACTION_GK : CFG.BOT_REACTION;
      ai.reactUntil = (ai.t || 0) + base * (0.7 + 0.6 * g.rng());
    }
    if (ai.reactUntil && (ai.t || 0) < ai.reactUntil && ai.lastInput) {
      ai.t = (ai.t || 0) + dt;
      const held = Object.assign(emptyInput(), ai.lastInput);
      held.aim = { x: ai.lastInput.aim.x, y: ai.lastInput.aim.y };
      held.special = false; held.tackle = false;   // não repete ações de um toque
      return held;
    }
    return null;
  }
  function think(p, g, dt, macroFn) {
    const held = reactionHold(p, g, dt);
    if (held) return held;
    const inp = decide(p, g, dt, macroFn);
    p.ai.lastInput = inp;
    return inp;
  }

  function decide(p, g, dt, macroFn) {
    const inp = emptyInput();
    const ai = p.ai;
    ai.t = (ai.t || 0) + dt;
    const c = context(p, g);
    if (!c.hasBall) ai.mode = 'none';
    const macro = macroFn ? macroFn(p, g, c) : chooseMacro(p, g, c);
    ai.macro = macro;
    if (p.isKeeper) return keeperExecute(p, g, inp, dt, c, macro);
    return execute(p, g, dt, c, macro, inp);
  }

  // ---------- goleiro: decisão ----------
  function chooseKeeperMacro(p, g, c) {
    const { ball, dir, mates, opps } = c;
    const ai = p.ai;
    if (c.hasBall) return chooseCarrierMacro(p, g, c);   // com a bola (no pé ou na mão): as mesmas opções de qualquer jogador
    const loose = !ball.owner;
    const inBox = g.inBoxPt(ball.pos, p.team);
    if (loose && inBox) {
      const myD = V.dist(p.pos, ball.pos), oppD = nearest(opps, ball.pos).d;
      if (myD < oppD - 10 || myD < 90) return 'gk_rush';
    }
    if (ball.owner && ball.owner.team !== p.team && inBox && V.dist(p.pos, ball.pos) < 160) return 'gk_rush';
    // pressionar: determinístico e com histerese (entra a 480 px, sai a 560), sem alternar a cada tick
    const dBG = V.dist(ball.pos, c.ownGoal);
    if (ball.owner && ball.owner.team !== p.team && dBG < (ai.gkPress ? 560 : 480)) { ai.gkPress = true; return 'gk_press'; }
    ai.gkPress = false;
    if (ball.pos.x * dir > c.W2 * 0.35 && (!ball.owner || ball.owner.team === p.team)) return 'gk_up';   // bola longe, no ataque: sobe
    const towardMe = loose && ball.vel.x * dir < 0 && V.len(ball.vel) > 400;
    if (towardMe && V.dist(ball.pos, c.ownGoal) < 500) return 'gk_line';
    return 'gk_angle';
  }

  // ---------- goleiro: execução ----------
  function keeperExecute(p, g, inp, dt, c, macro) {
    const { ball, dir, ownGoal, oppGoal, mates, opps } = c;
    const ai = p.ai;
    const team = p.team;
    const moveTo = (pt, sprint) => {
      const d = V.sub(pt, p.pos), l = V.len(d);
      if (l > 8) { const n = V.norm(d); inp.mx = n.x; inp.my = n.y; }
      inp.sprint = !!sprint && l > 40;
    };

    if (c.hasBall || (c.firstTouch && KICK_MACROS.includes(macro) && macro !== 'shoot')) {
      if (GK_MACROS.includes(macro)) macro = 'hold';
      // lançamento sem alvo: chutão para o ataque
      if (macro === 'longpass' && ai.mode !== 'shoot' && !longPassTarget(g, p, c)) {
        ai.mode = 'shoot'; ai.modeT = ai.t;
        ai.aim = { x: oppGoal.x * 0.55, y: (g.rng() - 0.5) * 500 };
        ai.chargeT = 0.75 * CFG.CHARGE_MAX;
      }
      if (p.held) {
        // bola na mão: chute e passe são iguais mecanicamente; "conduzir" solta a bola no pé,
        // "segurar" espera protegido pela zona de repulsão
        if (macro === 'dribble' || macro === 'carryspace') { inp.special = true; inp.aim = oppGoal; return inp; }
        if (macro === 'hold') { inp.aim = oppGoal; return inp; }
      }
      const out = execute(p, g, dt, c, macro, inp);
      // passe com a bola na mão e alvo longe: arremesso
      if (ai.mode === 'pass' && p.held && ai.aim && V.dist(p.pos, ai.aim) > 320 && inp.pass) { inp.throwBall = true; inp.pass = false; }
      return out;
    }
    if (!GK_MACROS.includes(macro)) macro = 'gk_angle';

    const toBall = V.sub(ball.pos, ownGoal), dBall = V.len(toBall);
    const speed = V.len(ball.vel);
    const loose = !ball.owner;
    const towardMe = ball.vel.x * dir < 0;
    const W2 = CFG.FIELD_W / 2;
    const xlo = team === 0 ? -W2 + 24 : W2 - 170;
    const xhi = team === 0 ? -W2 + 170 : W2 - 24;
    let target, sprint = false;

    if (macro === 'gk_rush') {
      // sair pressionando / varrer: vai na bola (prevista) ou no portador, com tackle quando chega
      if (loose) { const myD = V.dist(p.pos, ball.pos); target = predictBall(g, V.clamp(myD / 500, 0, 0.5)); }
      else { target = ball.pos; if (V.dist(p.pos, ball.pos) < 58 && p.cd.tackle <= 0) inp.tackle = true; }
      sprint = true;
    } else if (macro === 'gk_press') {
      // fecha o ângulo agressivamente: avança na linha bola-gol até perto do atacante,
      // com a postura defensiva (hitbox maior, agarra qualquer bola do adversário); sem tackle/mergulho
      const depth = Math.min(Math.max(60, dBall - 70), CFG.BOX_W - 25);
      target = V.add(ownGoal, V.mul(V.norm(toBall), depth));
      target.y = V.clamp(target.y, -CFG.BOX_H / 2 * 0.8, CFG.BOX_H / 2 * 0.8);
      sprint = V.dist(p.pos, target) > 120;
      moveTo(target, sprint);
      inp.aim = ball.pos;
      inp.stance = true;
      return inp;
    } else if (macro === 'gk_line') {
      // na linha: acompanha o y da bola (ou onde ela vai cruzar), fundo, pronto para mergulhar
      let y = ball.pos.y;
      if (loose && towardMe && speed > 180) { const tx = (xlo - ball.pos.x) / ball.vel.x; if (tx > 0 && tx < 1.5) y = ball.pos.y + ball.vel.y * tx; }
      target = { x: -dir * (W2 - 30), y: V.clamp(y, -CFG.GOAL_W / 2 * 0.85, CFG.GOAL_W / 2 * 0.85) };
    } else if (macro === 'gk_up') {
      // líbero: sobe até a borda da área, na linha bola-gol
      const depth = CFG.BOX_W - 30;
      target = V.add(ownGoal, V.mul(V.norm(toBall), depth));
      target.x = V.clamp(target.x, Math.min(xlo, -dir * (W2 - depth)), Math.max(xhi, -dir * (W2 - depth)));
      target.y = V.clamp(target.y, -CFG.BOX_H / 2 * 0.8, CFG.BOX_H / 2 * 0.8);
    } else {
      // fechar o ângulo: na linha bola-gol, um pouco à frente da linha
      const depth = V.clamp(50 + dBall * 0.06, 50, 115);
      target = V.add(ownGoal, V.mul(V.norm(toBall), depth));
      if (loose && towardMe && speed > 180) {
        const tx = (p.pos.x - ball.pos.x) / ball.vel.x;
        if (tx > 0 && tx < 1.2) target = { x: p.pos.x, y: ball.pos.y + ball.vel.y * tx };
      }
      target.x = V.clamp(target.x, xlo, xhi);
      target.y = V.clamp(target.y, -CFG.GOAL_W / 2 * 0.85, CFG.GOAL_W / 2 * 0.85);
    }
    moveTo(target, sprint);
    inp.aim = ball.pos;
    inp.stance = dBall < 300 && speed < 260 && !sprint && !inp.tackle;

    // mergulho: último recurso. Só quando é chute a gol que cruza a minha
    // linha num ponto que eu NÃO alcanço correndo a tempo.
    if (loose && p.cd.dive <= 0 && towardMe && speed > 260) {
      const tx = (p.pos.x - ball.pos.x) / ball.vel.x;
      if (tx > 0.02 && tx < 0.7) {
        const yAt = ball.pos.y + ball.vel.y * tx;
        const dy = yAt - p.pos.y;
        const onTarget = Math.abs(yAt) < CFG.GOAL_W / 2 + 12;
        const reachRunning = Math.abs(dy) < CFG.SPRINT * tx + 20;
        if (onTarget && !reachRunning && Math.abs(dy) < 190) {
          inp.special = true; inp.stance = false; inp.tackle = false;
          inp.mx = 0; inp.my = Math.sign(dy);
          inp.aim = { x: p.pos.x, y: yAt };
        }
      }
    }
    return inp;
  }

  return { think, reactionHold, reflex, MACROS, STYLES, chooseMacro, context, execute, keeperExecute, crossTarget };
})();
