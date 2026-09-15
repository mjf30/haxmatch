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
  const MACROS = ['shoot', 'pass', 'passback', 'longpass', 'switch', 'dribble', 'hold', 'chase', 'defend', 'cover', 'openfwd', 'openwide', 'overlap', 'runbox', 'openback', 'home'];

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
      if (m.isKeeper && !m.callT) continue;
      const on = nearest(opps, m.pos).d;
      if (on < 75) continue;
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
      teamHasKeeper: mates.some((m) => m.isKeeper),
    };
  }

  function homePos(p, g, c) {
    const h = { x: p.home.x, y: p.home.y };
    h.x += V.clamp(c.ball.pos.x * 0.45, -c.W2 * 0.4, c.W2 * 0.4);
    h.y += c.ball.pos.y * 0.35;
    if (c.ball.owner && c.ball.owner.team !== p.team) h.x -= c.dir * 150;
    // não entra na própria área se já existe goleiro (evita trocar de luvas)
    if (c.teamHasKeeper && Math.abs(h.y) < CFG.BOX_H / 2 + 30) {
      const edge = -c.dir * (c.W2 - CFG.BOX_W - 35);
      if ((h.x - edge) * c.dir < 0) h.x = edge;
    }
    return g.clampField(h, 40);
  }

  // quão livre está um ponto para receber: distância ao adversário mais próximo,
  // linha de passe limpa a partir da bola, e longe de companheiros (não aglomerar)
  function openness(pt, c, p) {
    const dOpp = nearest(c.opps, pt).d;
    const lane = laneClear(c.ball.pos, pt, c.opps, 24) ? 1 : 0;
    const dMate = nearest(c.mates, pt).d;
    return Math.min(dOpp, 260) + 120 * lane + Math.min(dMate, 200) * 0.6;
  }
  // procura, numa grade ao redor de um alvo nominal, o ponto mais livre dentro do campo
  function bestOpenPoint(nominal, c, p, g, radius) {
    let best = null, bs = -Infinity;
    for (let i = -2; i <= 2; i++) for (let j = -2; j <= 2; j++) {
      let pt = g.clampField({ x: nominal.x + i * radius / 2, y: nominal.y + j * radius / 2 }, 45);
      // não entra na própria área (deixa o goleiro em paz)
      if (c.teamHasKeeper && g.inBoxPt(pt, p.team)) continue;
      const s = openness(pt, c, p) - 0.15 * V.dist(pt, nominal);
      if (s > bs) { bs = s; best = pt; }
    }
    return best || g.clampField(nominal, 45);
  }
  // alvo nominal relativo ao portador (ou à bola) para cada macro de apoio
  function supportNominal(kind, p, g, c) {
    const anchor = c.ball.owner ? c.ball.owner.pos : c.ball.pos;
    const dir = c.dir;
    if (kind === 'openfwd') return { x: anchor.x + dir * 280, y: anchor.y * 0.5 + (p.pos.y >= anchor.y ? 1 : -1) * 90 };
    if (kind === 'openwide') { const side = p.pos.y >= anchor.y ? 1 : -1; return { x: anchor.x + dir * 60, y: V.clamp(anchor.y + side * 320, -CFG.FIELD_H / 2 + 80, CFG.FIELD_H / 2 - 80) }; }
    if (kind === 'overlap') {   // ultrapassagem por fora: à frente do portador, pela lateral
      const side = p.pos.y >= anchor.y ? 1 : -1;
      return { x: anchor.x + dir * 320, y: V.clamp(anchor.y + side * 300, -CFG.FIELD_H / 2 + 70, CFG.FIELD_H / 2 - 70) };
    }
    if (kind === 'runbox') {    // desmarcar na área adversária
      const W2 = CFG.FIELD_W / 2;
      return { x: dir * (W2 - CFG.BOX_W * 0.55), y: V.clamp(p.pos.y * 0.6 + (p.pos.y >= 0 ? 1 : -1) * 60, -CFG.BOX_H / 2 * 0.7, CFG.BOX_H / 2 * 0.7) };
    }
    return { x: anchor.x - dir * 240, y: anchor.y * 0.4 + (p.pos.y >= anchor.y ? 1 : -1) * 120 };   // openback
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

  function shotTargetFor(p, c) {
    const gk = c.opps.find((o) => o.isKeeper);
    const cornerY = (CFG.GOAL_W / 2 - 30) * (gk && gk.pos.y > 0 ? -1 : 1);
    return { x: c.oppGoal.x, y: cornerY };
  }

  // ---------- decisão tática do script ----------
  function chooseMacro(p, g, c) {
    const ai = p.ai;
    const ball = c.ball;
    if (c.hasBall) {
      if (ai.mode === 'shoot') return 'shoot';
      if (ai.mode === 'pass') return 'pass';
      const dGoal = V.dist(p.pos, c.oppGoal);
      const dOpp = nearest(c.opps, p.pos).d;
      const target = shotTargetFor(p, c);
      const lane = laneClear(ball.pos, target, c.opps.filter((o) => !o.isKeeper), 30);
      const angleOk = Math.abs(p.pos.y) < CFG.FIELD_H * 0.38 || dGoal < 260;
      if ((dGoal < 820 && lane && angleOk) || dGoal < 260 || (dGoal < 450 && angleOk && g.rng() < 0.03)) return 'shoot';
      const caller = c.mates.find((m) => m.callT > 0 && !(ai.lastCall && ai.t - ai.lastCall < 2.5));
      if (caller) return 'pass';
      if ((dOpp < 120 || g.rng() < 0.004) && bestPass(g, p, c.mates, c.opps, c.dir, 100)) return 'pass';
      // companheiro livre lá na frente: lançamento com chute carregado
      if ((dOpp < 160 || g.rng() < 0.01) && longPassTarget(g, p, c)) return 'longpass';
      // inversão de jogada quando o lado está congestionado
      if ((dOpp < 140 || g.rng() < 0.01) && switchTarget(g, p, c)) return 'switch';
      // pressionado, sem passe à frente: opção segura atrás, ou proteger a bola
      if (dOpp < 90 && safePassTarget(g, p, c)) return 'passback';
      if (dOpp < 60 || g.rng() < 0.01) return 'hold';
      return 'dribble';
    }
    // companheiro com a bola: quem está mais perto do portador apoia de perto (atrás),
    // os outros abrem espaço (à frente ou pelos lados), sem todos correrem para a bola
    if (ball.owner && ball.owner.team === p.team) {
      const carrier = ball.owner;
      const field = c.mates.filter((m) => !m.isKeeper && m !== carrier).concat([p]).sort((a, b) => V.dist(a.pos, carrier.pos) - V.dist(b.pos, carrier.pos));
      const order = field.indexOf(p);
      const ahead = (p.pos.x - carrier.pos.x) * c.dir > 0;
      const attacking = carrier.pos.x * c.dir > c.W2 * 0.3;   // bola no terço de ataque
      if (order === 0) return attacking ? (ahead ? 'runbox' : 'overlap') : (ahead ? 'openfwd' : 'openback');
      if (order === 1) return attacking ? (ahead ? 'runbox' : 'openwide') : 'openwide';
      return ahead ? 'openfwd' : 'home';
    }
    if (ball.owner) {
      const chasers = c.mates.filter((m) => !m.isKeeper).concat([p]);
      const order = chasers.sort((a, b) => V.dist(a.pos, ball.owner.pos) - V.dist(b.pos, ball.owner.pos)).indexOf(p);
      if (order === 0) return 'defend';
      if (order === 1) return 'cover';
      return 'home';
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
      if (d < 600 || d > 1500) continue;
      if ((m.pos.x - p.pos.x) * c.dir < 250) continue;        // tem que ser para a frente
      const lead = V.add(m.pos, V.mul(m.vel, 0.6));
      if (!laneClear(c.ball.pos, lead, c.opps, 30)) continue;
      const on = nearest(c.opps, lead).d;
      if (on < 110) continue;
      const s = (m.pos.x - p.pos.x) * c.dir + on * 0.5;
      if (s > bs) { bs = s; best = m; }
    }
    return best;
  }

  // passe seguro: companheiro atrás (ou ao lado) com linha limpa e sem marcador perto
  function safePassTarget(g, p, c) {
    let best = null, bs = -Infinity;
    for (const m of c.mates) {
      if (m.isKeeper && !m.callT) continue;
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

    if (c.hasBall) {
      // sem a bola essas decisões não fazem sentido; com a bola as outras viram "conduzir"
      if (!['shoot', 'pass', 'passback', 'longpass', 'switch', 'dribble', 'hold'].includes(macro)) macro = 'dribble';
      if (ai.mode === 'shoot' && macro !== 'shoot' && macro !== 'longpass') ai.mode = 'none';
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
          if (p.charge && p.charge.t >= ai.chargeT) { inp.shoot = false; ai.mode = 'none'; }
          else if (!p.charge && ai.t - ai.modeT > 0.2) ai.mode = 'none';
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

      if (macro === 'shoot') {
        if (ai.mode !== 'shoot') {
          const dGoal = V.dist(p.pos, c.oppGoal);
          ai.mode = 'shoot'; ai.modeT = ai.t; ai.aim = shotTargetFor(p, c);
          ai.chargeT = V.clamp(dGoal / 1000, 0.25, 1) * CFG.CHARGE_MAX;
        }
        inp.aim = ai.aim; inp.shoot = true;
        moveTo(c.oppGoal, false);
        if (p.charge && p.charge.t >= ai.chargeT) { inp.shoot = false; ai.mode = 'none'; }
        else if (!p.charge && ai.t - ai.modeT > 0.2) ai.mode = 'none';
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
      const clearAhead = !c.opps.some((o) => {
        const d = V.sub(o.pos, p.pos), l = V.len(d);
        return l < 230 && V.dot(V.norm(d), steer) > 0.55;
      });
      if (clearAhead && p.stamina > 10) {
        inp.sprint = true;
        if (p.cd.grab <= 0) inp.special = true;          // push ball
      } else if (dOpp < 80) {
        inp.stance = true;                                // postura de drible
        if (p.cd.dribble <= 0 && g.rng() < 0.06) inp.special = true;
      }
      return inp;
    }

    // ---------- sem a bola ----------
    ai.mode = 'none';
    if (['shoot', 'pass', 'passback', 'longpass', 'switch', 'dribble', 'hold'].includes(macro)) macro = 'chase';
    // desmarcar na área só faz sentido com a bola no campo de ataque
    if (macro === 'runbox' && ball.pos.x * dir < 0) macro = 'openfwd';
    if (macro === 'defend' && !(ball.owner && ball.owner.team !== p.team)) macro = ball.owner ? 'openback' : 'chase';

    // abrir espaço para receber: ponto livre perto de um alvo nominal relativo ao portador/bola
    if (macro === 'openfwd' || macro === 'openwide' || macro === 'openback' || macro === 'overlap' || macro === 'runbox') {
      const target = bestOpenPoint(supportNominal(macro, p, g, c), c, p, g, macro === 'runbox' ? 110 : 140);
      moveTo(target, macro === 'overlap' ? V.dist(p.pos, target) > 120 : V.dist(p.pos, target) > 260);
      inp.aim = ball.pos;                       // de frente para a bola, pronto para o toque de primeira
      // bola passada na minha direção: dominar / toque de primeira
      if (!ball.owner && g.ballInZone(p)) {
        const bs = V.len(ball.vel);
        if (bs > 180 && V.dist(p.pos, c.oppGoal) < 520 && g.rng() < 0.5) { inp.aim = { x: c.oppGoal.x, y: (g.rng() - 0.5) * (CFG.GOAL_W - 60) }; inp.shoot = true; }
      }
      return inp;
    }

    // cobrir: fechar o espaço entre a bola e o próprio gol, marcando o adversário livre mais perigoso
    if (macro === 'cover') {
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
      // toque de primeira em bola vindo forte na direção do gol adversário
      if (g.ballInZone(p) && bs > 180 && V.dist(p.pos, c.oppGoal) < 500 && g.rng() < 0.5) {
        inp.aim = { x: c.oppGoal.x, y: (g.rng() - 0.5) * (CFG.GOAL_W - 60) };
        inp.shoot = true; ai.firstT = ai.t;
      }
      return inp;
    }

    // home
    moveTo(homePos(p, g, c), false);
    return inp;
  }

  // ---------- tempo de reação + decisão ----------
  // macroFn (opcional): (p, g, ctx) -> macro. Sem ele, usa a decisão do script.
  function think(p, g, dt, macroFn) {
    const ai = p.ai;
    const ownerId = g.ball.owner ? g.ball.owner.id : -1;
    if (ai.lastOwner === undefined) ai.lastOwner = ownerId;
    if (ownerId !== ai.lastOwner) {
      ai.lastOwner = ownerId;
      const base = p.isKeeper ? CFG.BOT_REACTION_GK : CFG.BOT_REACTION;
      ai.reactUntil = (ai.t || 0) + base * (0.7 + 0.6 * g.rng());
    }
    if (!macroFn && ai.reactUntil && (ai.t || 0) < ai.reactUntil && ai.lastInput) {
      ai.t = (ai.t || 0) + dt;
      const held = Object.assign(emptyInput(), ai.lastInput);
      held.aim = { x: ai.lastInput.aim.x, y: ai.lastInput.aim.y };
      held.special = false; held.tackle = false;   // não repete ações de um toque
      return held;
    }
    const inp = decide(p, g, dt, macroFn);
    ai.lastInput = inp;
    return inp;
  }

  function decide(p, g, dt, macroFn) {
    const inp = emptyInput();
    const ai = p.ai;
    ai.t = (ai.t || 0) + dt;
    const c = context(p, g);
    if (!c.hasBall) ai.mode = 'none';
    if (p.isKeeper) return keeper(p, g, inp, dt, c);
    const macro = macroFn ? macroFn(p, g, c) : chooseMacro(p, g, c);
    ai.macro = macro;
    return execute(p, g, dt, c, macro, inp);
  }

  function keeper(p, g, inp, dt, c) {
    const { ball, dir, ownGoal, oppGoal, mates, opps } = c;
    const ai = p.ai;
    const hasBall = ball.owner === p;
    const team = p.team;
    const moveTo = (pt, sprint) => {
      const d = V.sub(pt, p.pos), l = V.len(d);
      if (l > 8) { const n = V.norm(d); inp.mx = n.x; inp.my = n.y; }
      inp.sprint = !!sprint && l > 40;
    };

    if (hasBall) {
      ai.holdT = (ai.holdT || 0) + dt;
      if (ai.mode === 'shoot') {
        inp.aim = ai.aim; inp.shoot = true;
        if (p.charge && p.charge.t >= ai.chargeT) { inp.shoot = false; ai.mode = 'none'; }
        else if (!p.charge && ai.t - ai.modeT > 0.2) ai.mode = 'none';
        return inp;
      }
      if (ai.mode === 'pass') {
        inp.aim = ai.aim;
        if (ai.useThrow) inp.throwBall = ai.holdN-- > 0; else inp.pass = ai.holdN-- > 0;
        if (ai.holdN < 0) ai.mode = 'none';
        return inp;
      }
      const caller = mates.find((m) => m.callT > 0);
      if (caller && ai.holdT > 0.3) {
        const target = V.add(caller.pos, V.mul(caller.vel, 0.3));
        const dCall = V.dist(p.pos, target);
        if (dCall < 800 && laneClear(ball.pos, target, opps, 22)) {
          ai.mode = 'pass'; ai.aim = target;
          ai.useThrow = p.held && dCall > 320;
          ai.holdN = ai.useThrow ? 1 : 2 + Math.floor(dCall / 300);
          if (ai.useThrow) inp.throwBall = true; else inp.pass = true;
        } else {
          ai.mode = 'shoot'; ai.modeT = ai.t; ai.aim = target;
          ai.chargeT = V.clamp(dCall / 1500, 0.4, 1) * CFG.CHARGE_MAX;
          inp.shoot = true;
        }
        inp.aim = ai.aim;
        return inp;
      }
      if (ai.holdT > 0.7) {
        const cand = bestPass(g, p, mates, opps, dir, 150);
        if (cand) {
          ai.mode = 'pass'; ai.aim = V.add(cand.pos, V.mul(cand.vel, 0.3));
          ai.useThrow = p.held && V.dist(p.pos, cand.pos) > 320;
          ai.holdN = ai.useThrow ? 1 : 2 + Math.floor(V.dist(p.pos, cand.pos) / 300);
          if (ai.useThrow) inp.throwBall = true; else inp.pass = true;
          inp.aim = ai.aim;
          return inp;
        }
        if (ai.holdT > 1.6) {
          ai.mode = 'shoot'; ai.modeT = ai.t;
          ai.aim = { x: oppGoal.x * 0.55, y: (g.rng() - 0.5) * 500 };
          ai.chargeT = 0.75 * CFG.CHARGE_MAX;
          inp.aim = ai.aim; inp.shoot = true;
          return inp;
        }
      }
      inp.aim = oppGoal;
      return inp;
    }
    ai.holdT = 0;

    const toBall = V.sub(ball.pos, ownGoal), dBall = V.len(toBall);
    const speed = V.len(ball.vel);
    const loose = !ball.owner;
    const towardMe = ball.vel.x * dir < 0;
    const W2 = CFG.FIELD_W / 2;
    const xlo = team === 0 ? -W2 + 24 : W2 - 170;
    const xhi = team === 0 ? -W2 + 170 : W2 - 24;

    // posição base: na linha bola-gol, um pouco à frente da linha
    const depth = V.clamp(50 + dBall * 0.06, 50, 115);
    let target = V.add(ownGoal, V.mul(V.norm(toBall), depth));
    // chute vindo: vai para o ponto onde a bola cruza a minha linha
    if (loose && towardMe && speed > 180) {
      const tx = (p.pos.x - ball.pos.x) / ball.vel.x;
      if (tx > 0 && tx < 1.2) target = { x: p.pos.x, y: ball.pos.y + ball.vel.y * tx };
    }
    target.x = V.clamp(target.x, xlo, xhi);
    target.y = V.clamp(target.y, -CFG.GOAL_W / 2 * 0.85, CFG.GOAL_W / 2 * 0.85);
    let sprint = false;

    if (loose && g.inBoxPt(ball.pos, team)) {
      const myD = V.dist(p.pos, ball.pos);
      const oppD = nearest(opps, ball.pos).d;
      if (myD < oppD - 10 || myD < 90) { target = predictBall(g, V.clamp(myD / 500, 0, 0.5)); sprint = true; }
    }
    if (ball.owner && ball.owner.team !== team && g.inBoxPt(ball.pos, team)) {
      const d = V.dist(p.pos, ball.pos);
      if (d < 58 && p.cd.tackle <= 0) inp.tackle = true;
      if (d < 160) { target = ball.pos; sprint = true; }
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

  return { think, MACROS, chooseMacro, context, execute };
})();
