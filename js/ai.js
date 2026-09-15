'use strict';
// Bots simples. Produzem o mesmo formato de input que o humano.
const AI = (() => {
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

  function think(p, g, dt) {
    const inp = emptyInput();
    const ball = g.ball;
    const dir = p.team === 0 ? 1 : -1;
    const W2 = CFG.FIELD_W / 2;
    const ownGoal = { x: -dir * W2, y: 0 }, oppGoal = { x: dir * W2, y: 0 };
    const mates = g.players.filter((q) => q.team === p.team && q !== p);
    const opps = g.players.filter((q) => q.team !== p.team);
    const ai = p.ai;
    ai.t = (ai.t || 0) + dt;
    inp.aim = { x: ball.pos.x, y: ball.pos.y };
    const hasBall = ball.owner === p;
    if (!hasBall) ai.mode = 'none';

    const moveTo = (pt, sprint) => {
      const d = V.sub(pt, p.pos), l = V.len(d);
      if (l > 8) { const n = V.norm(d); inp.mx = n.x; inp.my = n.y; }
      inp.sprint = !!sprint && l > 40;
    };
    const teamHasKeeper = mates.some((m) => m.isKeeper);
    const homePos = () => {
      const h = { x: p.home.x, y: p.home.y };
      h.x += V.clamp(ball.pos.x * 0.45, -W2 * 0.4, W2 * 0.4);
      h.y += ball.pos.y * 0.35;
      if (ball.owner && ball.owner.team !== p.team) h.x -= dir * 150;
      // não entra na própria área se já existe goleiro (evita trocar de luvas)
      if (teamHasKeeper && Math.abs(h.y) < CFG.BOX_H / 2 + 30) {
        const edge = -dir * (W2 - CFG.BOX_W - 35);
        if ((h.x - edge) * dir < 0) h.x = edge;
      }
      return g.clampField(h, 40);
    };

    if (p.isKeeper) return keeper(p, g, inp, dt, dir, ownGoal, oppGoal, mates, opps, moveTo);

    // ---------- com a bola ----------
    if (hasBall) {
      const dGoal = V.dist(p.pos, oppGoal);
      const no = nearest(opps, p.pos);
      const dOpp = no.d;

      if (ai.mode === 'shoot') {
        inp.aim = ai.aim; inp.shoot = true;
        moveTo(oppGoal, false);
        if (p.charge && p.charge.t >= ai.chargeT) { inp.shoot = false; ai.mode = 'none'; }
        else if (!p.charge && ai.t - ai.modeT > 0.2) ai.mode = 'none';
        return inp;
      }
      if (ai.mode === 'pass') {
        inp.aim = ai.aim; inp.pass = ai.holdN-- > 0;
        if (ai.holdN < 0) ai.mode = 'none';
        return inp;
      }

      // chutar?
      const gk = opps.find((o) => o.isKeeper);
      const cornerY = (CFG.GOAL_W / 2 - 30) * (gk && gk.pos.y > 0 ? -1 : 1);
      const shotTarget = { x: oppGoal.x, y: cornerY };
      const lane = laneClear(ball.pos, shotTarget, opps.filter((o) => !o.isKeeper), 30);
      const angleOk = Math.abs(p.pos.y) < CFG.FIELD_H * 0.38 || dGoal < 260;
      if ((dGoal < 820 && lane && angleOk) || dGoal < 260 || (dGoal < 450 && angleOk && g.rng() < 0.03)) {
        ai.mode = 'shoot'; ai.modeT = ai.t; ai.aim = shotTarget;
        ai.chargeT = V.clamp(dGoal / 1000, 0.25, 1) * CFG.CHARGE_MAX;
        inp.aim = shotTarget; inp.shoot = true;
        return inp;
      }
      // passar?
      if (dOpp < 120 || g.rng() < 0.004) {
        const cand = bestPass(g, p, mates, opps, dir, 100);
        if (cand) {
          ai.mode = 'pass';
          ai.aim = V.add(cand.pos, V.mul(cand.vel, 0.3));
          ai.holdN = 2 + Math.floor(V.dist(p.pos, cand.pos) / 250);
          inp.aim = ai.aim; inp.pass = true;
          return inp;
        }
      }
      // conduzir em direção ao gol, desviando do marcador
      let steer = V.norm(V.sub({ x: oppGoal.x - dir * 100, y: p.pos.y * 0.4 }, p.pos));
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
      const clearAhead = !opps.some((o) => {
        const d = V.sub(o.pos, p.pos), l = V.len(d);
        return l < 230 && V.dot(V.norm(d), steer) > 0.55;
      });
      if (clearAhead && p.stamina > 15) {
        inp.sprint = true;
        if (p.cd.grab <= 0) inp.special = true;          // push ball
      } else if (dOpp < 80) {
        inp.stance = true;                                // postura de drible
        if (p.cd.dribble <= 0 && g.rng() < 0.06) inp.special = true;
      }
      return inp;
    }

    // ---------- companheiro tem a bola ----------
    if (ball.owner && ball.owner.team === p.team) {
      const carrier = ball.owner;
      const h = homePos();
      h.x += dir * 140;
      if (V.dist(h, carrier.pos) < 170) h.y += (p.pos.y >= carrier.pos.y ? 1 : -1) * 170;
      const t = g.clampField(h, 40);
      moveTo(t, V.dist(p.pos, t) > 220);
      inp.aim = carrier.pos;
      return inp;
    }

    // ---------- adversário tem a bola ----------
    if (ball.owner) {
      const carrier = ball.owner;
      const chasers = mates.filter((m) => !m.isKeeper).concat([p]);
      const chaser = nearest(chasers, carrier.pos).p;
      if (chaser === p) {
        const goalSide = V.norm(V.sub(ownGoal, carrier.pos));
        const intercept = V.add(carrier.pos, V.mul(goalSide, 26));
        const d = V.dist(p.pos, ball.pos);
        moveTo(intercept, d > 150);
        inp.aim = ball.pos;
        if (d < 150) inp.stance = true;
        const toBall = V.norm(V.sub(ball.pos, p.pos));
        if (d < 54 && p.cd.tackle <= 0 && V.dot(toBall, p.moveDir) > 0.2) {
          inp.tackle = true; inp.sprint = false;
        } else if (d > 70 && d < 135 && p.cd.slide <= 0 && p.stamina > 30 &&
          V.dot(carrier.vel, V.sub(carrier.pos, p.pos)) > 40 && V.len(carrier.vel) > 130 && g.rng() < 0.15) {
          inp.sprint = true; inp.tackle = true; inp.stance = false;
        }
      } else {
        moveTo(homePos(), false);
      }
      return inp;
    }

    // ---------- bola solta ----------
    const dBall = V.dist(p.pos, ball.pos);
    const pred = predictBall(g, V.clamp(dBall / 450, 0, 1.2));
    const rank = mates.filter((m) => !m.isKeeper && V.dist(m.pos, pred) < V.dist(p.pos, pred)).length;
    if (rank < 2) {
      moveTo(pred, dBall > 90);
      inp.aim = ball.pos;
      const bs = V.len(ball.vel);
      if (bs > CFG.CONTROL_MAX * 0.85 && dBall < 150 && V.dot(ball.vel, V.sub(p.pos, ball.pos)) > 0) inp.stance = true;
      // toque de primeira em bola vindo forte na direção do gol adversário
      if (g.ballInZone(p) && bs > 180 && V.dist(p.pos, oppGoal) < 500 && g.rng() < 0.5) {
        inp.aim = { x: oppGoal.x, y: (g.rng() - 0.5) * (CFG.GOAL_W - 60) };
        inp.shoot = true; ai.firstT = ai.t;
      }
    } else {
      moveTo(homePos(), false);
    }
    return inp;
  }

  function keeper(p, g, inp, dt, dir, ownGoal, oppGoal, mates, opps, moveTo) {
    const ball = g.ball;
    const ai = p.ai;
    const hasBall = ball.owner === p;
    const team = p.team;

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
      if (d < 120) target = ball.pos;
    }
    moveTo(target, sprint);
    inp.aim = ball.pos;
    inp.stance = dBall < 300 && speed < 260 && !sprint && !inp.tackle;

    // mergulho: para o ponto de interceptação (chute cruzando minha linha ou
    // bola escapando para o gol mais rápido do que consigo correr)
    if (loose && p.cd.dive <= 0 && speed > 180) {
      let pt = null;
      if (towardMe) {
        const tx = (p.pos.x - ball.pos.x) / ball.vel.x;
        if (tx > 0.02 && tx < 0.6) {
          const yAt = ball.pos.y + ball.vel.y * tx;
          if (Math.abs(yAt) < CFG.GOAL_W / 2 + 40) pt = { x: p.pos.x, y: yAt };
        }
      }
      if (!pt) {
        const pr = predictBall(g, 0.3);
        const passingMe = (pr.x - p.pos.x) * dir < -10 && Math.abs(pr.y) < CFG.GOAL_W / 2 + 60;
        if (passingMe && V.dist(p.pos, pr) < 140) pt = pr;
      }
      if (pt) {
        const dv = V.sub(pt, p.pos), d = V.len(dv);
        if (d > 22 && d < 170) {
          const n = V.norm(dv);
          inp.special = true; inp.stance = false; inp.tackle = false;
          inp.mx = n.x; inp.my = n.y;
          inp.aim = pt;
        }
      }
    }
    return inp;
  }

  return { think };
})();
