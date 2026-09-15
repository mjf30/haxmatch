'use strict';
// Simulação determinística do jogo. Não depende de DOM: recebe inputs por
// jogador (setInput) e avança em passos fixos (step). Isso permite rodar a
// mesma simulação num servidor no futuro.

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makePlayer(id, team, idx, home, name) {
  const fx = team === 0 ? 1 : -1;
  return {
    id, team, idx, name,
    home: { x: home.x, y: home.y },
    pos: { x: home.x, y: home.y }, vel: { x: 0, y: 0 },
    moveDir: { x: fx, y: 0 }, facing: { x: fx, y: 0 }, moving: false,
    r: CFG.PLAYER_R, isKeeper: false, human: false, active: true,   // active=false: vaga vazia (fora do campo)
    input: emptyInput(), prevInput: emptyInput(),
    stance: 'none', sprinting: false, effortT: 0, effortBar: 1, lastSprintTap: -10,
    stamina: CFG.STAMINA_MAX,
    action: null,            // {type, t, dur, dir, hit, ...}
    recover: 0,              // tempo com controle reduzido (erro de tackle, pós-chute)
    fallen: 0,               // caído (carrinho)
    getup: 0,                // levantando (após carrinho/mergulho)
    cd: { tackle: 0, slide: 0, dash: 0, dribble: 0, dive: 0, grab: 0, gloves: 0, through: 0 },
    charge: null,            // {kind:'shot'|'pass', t, dir0, lastAngle, spin}
    queued: null,            // ação de primeira agendada na zona de ação: {kind:'shot'|'pass'|'push', t, charging, dir0, lastAngle, spin}
    armed: { shoot: false, pass: false },   // botão segurado e ainda não consumido (pré-carga de chute/passe)
    touchChain: 0,           // toques de primeira seguidos sem dominar (a partir do 2º não há passo acelerado)
    held: false, holdT: 0,   // goleiro com a bola nas mãos
    pushFlash: 0, callT: 0, fakeT: 0,
    stats: { goals: 0, assists: 0, steals: 0, saves: 0 },
    ping: 0,                 // ms (só faz sentido em rede)
    ai: {},
  };
}

class Game {
  constructor(opts = {}) {
    this.teamSize = opts.teamSize || CFG.TEAM_SIZE;
    this.rng = mulberry32(opts.seed || 1);
    this.events = [];
    this.reset();
  }

  // ---------- setup ----------
  reset() {
    this.players = [];
    this.time = CFG.MATCH_TIME;
    this.score = [0, 0];
    this.now = 0;
    this.msg = null;
    const form = FORMATIONS[this.teamSize];
    let id = 0;
    for (let team = 0; team < 2; team++) {
      const dir = team === 0 ? 1 : -1;
      for (let i = 0; i < this.teamSize; i++) {
        const name = (team === 0 ? 'V' : 'A') + (i + 1);
        const home = { x: form[i].x * dir * CFG.FIELD_W / 2, y: form[i].y * dir * CFG.FIELD_H / 2 };
        this.players.push(makePlayer(id++, team, i, home, name));
      }
    }
    this.ball = { pos: { x: 0, y: 0 }, vel: { x: 0, y: 0 }, spin: 0, r: CFG.BALL_R, owner: null, lastTouch: null, prevTouch: null, lastTeam: -1, rot: 0 };
    this.kickoffTeam = this.rng() < 0.5 ? 0 : 1;   // primeiro kickoff: time sorteado
    this.kickoff(true);
  }

  kickoff(first) {
    for (const p of this.players) {
      p.pos = p.active ? { x: p.home.x, y: p.home.y } : { x: 0, y: -CFG.FIELD_H };
      p.vel = { x: 0, y: 0 };
      p.action = null; p.charge = null; p.queued = null;
      p.fallen = 0; p.getup = 0; p.recover = 0;
      p.isKeeper = false; p.held = false; p.holdT = 0;
      p.stance = 'none'; p.sprinting = false; p.effortT = 0;
      for (const k in p.cd) p.cd[k] = 0;
      if (first) { p.stamina = CFG.STAMINA_MAX; p.effortBar = 1; }
      p.ai = {};
    }
    const b = this.ball;
    b.pos = { x: 0, y: 0 }; b.vel = { x: 0, y: 0 }; b.spin = 0; b.owner = null;
    // saída: um jogador do time que dá o kickoff fica no meio com a bola, de frente
    // para o próprio campo (de costas para o ataque)
    const team = this.kickoffTeam;
    const dir = team === 0 ? 1 : -1;
    const cands = this.players.filter((p) => p.team === team && p.active && p.idx !== 0);
    const kicker = cands.sort((a, c) => (c.human ? 1 : 0) - (a.human ? 1 : 0) || c.idx - a.idx)[0]
      || this.players.find((p) => p.team === team && p.active);
    if (kicker) {
      kicker.pos = { x: -dir * (kicker.r + 4), y: 0 };
      kicker.facing = { x: -dir, y: 0 }; kicker.moveDir = { x: -dir, y: 0 };
      b.owner = kicker; b.lastTouch = kicker; b.lastTeam = team; kicker.held = false;
      b.pos = { x: -dir * (kicker.r * 2 + b.r + CFG.CARRY_DIST + 4), y: 0 };
    }
    this.state = 'kickoff';
    this.stateT = CFG.KICKOFF_FREEZE;
    this.msg = null;
  }

  setInput(pid, input) { this.players[pid].input = input; }

  // ---------- geometria ----------
  goalX(team) { return (team === 0 ? -1 : 1) * CFG.FIELD_W / 2; }
  inBoxPt(pt, team) {
    const W2 = CFG.FIELD_W / 2;
    const xin = team === 0 ? (pt.x <= -W2 + CFG.BOX_W) : (pt.x >= W2 - CFG.BOX_W);
    return xin && Math.abs(pt.y) <= CFG.BOX_H / 2;
  }
  inOwnBox(p) { return this.inBoxPt(p.pos, p.team); }
  inOwnHalf(p) { return p.team === 0 ? p.pos.x <= 0 : p.pos.x >= 0; }
  clampField(pt, m = 0) {
    return { x: V.clamp(pt.x, -CFG.FIELD_W / 2 + m, CFG.FIELD_W / 2 - m), y: V.clamp(pt.y, -CFG.FIELD_H / 2 + m, CFG.FIELD_H / 2 - m) };
  }
  // limites da arena com a boca do gol aberta: dá para entrar no gol com a bola no pé
  static clampArena(pt, r) {
    const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2;
    const mouth = Math.abs(pt.y) < CFG.GOAL_W / 2 - r;
    const p = { x: pt.x, y: pt.y };
    if (Math.abs(p.x) > W2) {
      // dentro do gol: só se estiver na boca; senão volta para a linha
      if (!mouth) p.x = V.clamp(p.x, -W2 + r, W2 - r);
      else p.x = V.clamp(p.x, -W2 - CFG.GOAL_D + r, W2 + CFG.GOAL_D - r);
      if (Math.abs(p.x) > W2) p.y = V.clamp(p.y, -CFG.GOAL_W / 2 + r, CFG.GOAL_W / 2 - r);
    } else {
      p.y = V.clamp(p.y, -H2 + r, H2 - r);
    }
    return p;
  }
  hasBall(p) { return this.ball.owner === p && !p.held; }
  // raio do corpo em relação à bola: menor sem postura defensiva, maior com ela
  ballHitbox(p) { return p.stance === 'def' ? p.r + CFG.GRAB_MARGIN_DEF : p.r * CFG.HITBOX_MUL + CFG.GRAB_MARGIN; }
  // bola solta perto o bastante para agir de primeira (zona de ação)
  ballInZone(p) {
    const b = this.ball;
    return !b.owner && V.dist(p.pos, b.pos) < p.r + b.r + CFG.ACTION_RADIUS;
  }

  // ---------- passo ----------
  step(dt) {
    this.now += dt;
    this.events.length = 0;
    if (this.state === 'goal') {
      this.stateT -= dt;
      if (this.stateT <= 0) this.kickoff(false);
    } else if (this.state === 'kickoff') {
      this.stateT -= dt;
      if (this.stateT <= 0) { this.state = 'play'; this.msg = null; }
    } else if (this.state === 'play') {
      this.time -= dt;
      if (this.time <= 0) { this.time = 0; this.endMatch(); }
    }
    const frozen = this.state !== 'play';
    for (const p of this.players) if (p.active) this.updatePlayer(p, dt, frozen);
    this.collidePlayers();
    this.keeperRepel(dt);
    for (const p of this.players) if (p.active) this.clampPlayer(p);
    this.updateBall(dt);
    if (this.state === 'play') this.contacts();
    if (!this.ball.owner) Game.wallsFree(this.ball);   // contatos podem empurrar a bola para a parede
    if (this.state === 'play') this.updateGloves(dt);
    if (this.state === 'play') this.checkGoal();
    for (const p of this.players) p.prevInput = p.input;
    // estatísticas a partir dos eventos deste passo
    for (const e of this.events) {
      if ((e.type === 'steal' || e.type === 'tackle-win' || e.type === 'slide-hit') && e.p) e.p.stats.steals++;
      if ((e.type === 'save' || e.type === 'parry') && e.p) e.p.stats.saves++;
    }
    // quem tocou antes do último toque (para assistência)
    const b = this.ball;
    if (b.lastTouch !== this._seenTouch) { b.prevTouch = this._seenTouch || null; this._seenTouch = b.lastTouch; }
  }

  // ---------- jogador ----------
  updatePlayer(p, dt, frozen) {
    const inp = frozen ? emptyInput() : p.input;
    const prev = p.prevInput;
    const hasBall = this.hasBall(p);
    const held = this.ball.owner === p && p.held;

    for (const k in p.cd) p.cd[k] = Math.max(0, p.cd[k] - dt);
    p.recover = Math.max(0, p.recover - dt);
    p.fallen = Math.max(0, p.fallen - dt);
    p.getup = Math.max(0, p.getup - dt);
    p.pushFlash = Math.max(0, p.pushFlash - dt);
    p.callT = Math.max(0, p.callT - dt);
    p.fakeT = Math.max(0, p.fakeT - dt);
    p.effortT = Math.max(0, p.effortT - dt);
    p.effortBar = Math.min(1, p.effortBar + dt / CFG.EFFORT_RECHARGE);
    if (inp.call && !prev.call) { p.callT = 1.5; this.events.push({ type: 'call', p }); }

    // mira e direção de movimento
    const aimV = V.sub(inp.aim, p.pos);
    if (V.len(aimV) > 2) p.facing = V.norm(aimV);
    const ml = Math.hypot(inp.mx, inp.my);
    p.moving = ml > 0.1;
    if (p.moving) p.moveDir = V.norm({ x: inp.mx, y: inp.my });

    // postura (Ctrl): drible com bola, defensiva sem bola
    p.stance = inp.stance ? (hasBall ? 'drib' : 'def') : 'none';

    // sprint e extra effort (Shift duas vezes)
    if (inp.sprint && !prev.sprint) {
      if (this.now - p.lastSprintTap < CFG.DOUBLE_TAP && p.effortBar >= 1 && !held) {
        p.effortT = CFG.EFFORT_DUR; p.effortBar = 0;
        this.events.push({ type: 'effort', p });
      }
      p.lastSprintTap = this.now;
    }
    p.sprinting = inp.sprint && p.stamina > 0 && !held;

    const incapacitated = p.fallen > 0 || p.getup > 0;
    if (p.action) {
      this.runAction(p, dt);
    } else if (incapacitated) {
      p.vel = V.mul(p.vel, Math.max(0, 1 - 8 * dt));
      p.charge = null; p.queued = null;
    } else {
      if (!frozen) this.handleActions(p, inp, prev, dt, hasBall, held);
      this.move(p, inp, dt, this.hasBall(p), this.ball.owner === p && p.held);
    }

    // stamina
    const keeperFree = p.isKeeper && this.inOwnHalf(p);
    if (p.sprinting && p.moving && !keeperFree && !p.action) {
      p.stamina = Math.max(0, p.stamina - CFG.STAMINA_SPRINT * dt);
    } else {
      p.stamina = Math.min(CFG.STAMINA_MAX, p.stamina + CFG.STAMINA_REGEN * dt);
    }
    if (keeperFree) p.stamina = CFG.STAMINA_MAX;

    // goleiro com a bola nas mãos: limite de tempo e sair da área solta a bola
    if (this.ball.owner === p && p.held) {
      p.holdT += dt;
      if (p.holdT > CFG.GK_HOLD_MAX || !this.inOwnBox(p)) { p.held = false; this.events.push({ type: 'drop', p }); }
    }

    p.pos = V.add(p.pos, V.mul(p.vel, dt));
  }

  lungeDir(p) { return p.moving ? p.moveDir : p.facing; }

  kickDir(p) {
    const b = this.ball;
    const d = V.sub(p.input.aim, b.pos);
    return V.len(d) > 4 ? V.norm(d) : p.facing;
  }

  handleActions(p, inp, prev, dt, hasBall, held) {
    const pressed = (k) => inp[k] && !prev[k];
    const released = (k) => !inp[k] && prev[k];
    const ball = this.ball;
    const canKick = hasBall || held;
    const inZone = this.ballInZone(p);
    // botão "armado": apertou e ainda não usou; soltar desarma. Permite pré-carregar
    // antes da bola entrar na zona de ação.
    for (const k of ['shoot', 'pass']) {
      if (pressed(k)) p.armed[k] = true;
      if (!inp[k]) p.armed[k] = false;
    }

    // ---- ação de primeira agendada (zona de ação) ----
    // O jogador não age de imediato: acelera até a bola e executa no toque.
    if (p.queued) {
      const q = p.queued;
      if (!inZone || canKick) { p.queued = null; }
      else {
        if (q.charging) {
          q.t += dt;
          if (q.kind === 'shot') {
            const a = V.angle(p.facing);
            q.spin = V.clamp(q.spin + V.angleDiff(q.lastAngle, a) * CFG.SPIN_GAIN, -CFG.SPIN_MAX, CFG.SPIN_MAX);
            q.lastAngle = a;
            if (released('shoot')) q.charging = false;
          } else if (q.kind === 'pass' && released('pass')) q.charging = false;
        }
        if (pressed('tackle')) p.queued = null;   // cancela para tacklear
        return;
      }
    }
    if (!canKick && inZone) {
      if (p.armed.shoot) { p.armed.shoot = false; p.queued = { kind: 'shot', t: 0, charging: true, dir0: this.kickDir(p), lastAngle: V.angle(p.facing), spin: 0 }; return; }
      if (p.armed.pass) { p.armed.pass = false; p.queued = { kind: 'pass', t: 0, charging: true }; return; }
      if (pressed('special')) { p.queued = { kind: 'push', t: 0, charging: false }; return; }   // push é toque, não segura
    }

    // ---- carga de chute / passe com a bola dominada ----
    if (p.charge) {
      const c = p.charge;
      c.t += dt;
      if (!canKick) { p.charge = null; return; }
      if (c.kind === 'shot') {
        // efeito: deslocamento angular do mouse durante a carga
        const a = V.angle(p.facing);
        c.spin = V.clamp(c.spin + V.angleDiff(c.lastAngle, a) * CFG.SPIN_GAIN, -CFG.SPIN_MAX, CFG.SPIN_MAX);
        c.lastAngle = a;
        if (released('shoot')) { this.shoot(p, c); p.charge = null; }
        else if (pressed('special') || pressed('pass')) { p.charge = null; p.fakeT = 0.3; this.events.push({ type: 'fake', p }); }
      } else if (released('pass')) {
        this.pass(p, c); p.charge = null;
      }
      return;
    }

    if (canKick) {
      if (pressed('shoot')) {
        p.armed.shoot = false;
        p.charge = { kind: 'shot', t: 0, dir0: this.kickDir(p), lastAngle: V.angle(p.facing), spin: 0 };
        return;
      }
      if (pressed('pass')) { p.armed.pass = false; p.charge = { kind: 'pass', t: 0 }; return; }
      // extra effort com a bola nos pés: empurra sozinho enquanto a arrancada dura
      if (hasBall && p.effortT > 0 && p.moving) { this.push(p, true); return; }
      if (held && pressed('throwBall')) { this.throwBall(p); return; }
      if (pressed('special')) {
        if (held) { p.held = false; p.holdT = 0; return; }                     // solta e conduz
        if (p.stance === 'drib') { if (p.cd.dribble <= 0 && p.stamina >= CFG.COST_DRIBBLE * 0.5) this.startDribble(p); return; }
        this.push(p, p.sprinting);
      }
      return;
    }

    // ---- sem a bola nos pés ----
    if (pressed('tackle')) {
      if (p.sprinting && p.moving) { if (p.cd.slide <= 0 && p.stamina >= 6) this.startSlide(p); }
      else if (p.cd.tackle <= 0) this.startTackle(p);
      return;
    }
    if (pressed('special')) {
      if (p.stance === 'def') { if (p.cd.dash <= 0) this.startDash(p); }
      else if (p.isKeeper && this.inOwnBox(p) && p.cd.dive <= 0) {
        this.startAction(p, 'gkdive', this.lungeDir(p), CFG.GK_DIVE_DUR);
      }
      // sem bola e sem postura, Espaço não faz nada (o mergulho de linha do Rematch é um pulo, sem sentido em 2D)
    }
  }

  move(p, inp, dt, hasBall, held) {
    let base;
    if (held) base = CFG.KEEPER_HOLD_SPEED;
    else if (hasBall) {
      base = p.sprinting && p.moving && p.stance !== 'drib' ? CFG.SPRINT_BALL : CFG.SPEED_BALL;
      if (p.stance === 'drib') base *= CFG.MUL_DRIBBLE;
    }
    else if (p.stance === 'def') base = CFG.SPEED * CFG.MUL_DEF;
    else if (p.sprinting && p.moving) base = CFG.SPRINT * (p.effortT > 0 ? CFG.EXTRA_EFFORT : 1);
    else base = CFG.SPEED;
    if (p.charge) base *= CFG.MUL_CHARGE;
    if (p.recover > 0) base *= CFG.MUL_RECOVER;
    let desired = { x: inp.mx * base, y: inp.my * base };
    if (p.queued) {
      // com ação de primeira agendada o jogador vai em direção à bola, na
      // velocidade normal do estado dele (sem boost algum)
      const toBall = V.norm(V.sub(this.ball.pos, p.pos));
      desired = V.mul(toBall, base);
    }
    const diff = V.sub(desired, p.vel);
    const dl = V.len(diff);
    const acc = (V.len(desired) > V.len(p.vel) ? CFG.ACCEL : CFG.DECEL) * dt;
    p.vel = dl <= acc ? desired : V.add(p.vel, V.mul(diff, acc / dl));
  }

  // ---------- ações ----------
  startAction(p, type, dir, dur) {
    p.action = { type, t: 0, dur, dir, hit: false, body: false, rolled: false, bodyHit: [] };
    p.charge = null;
    this.events.push({ type: 'action', p, action: type });
  }
  startTackle(p) {
    this.startAction(p, 'tackle', this.lungeDir(p), CFG.TACKLE_DUR);
    p.cd.tackle = CFG.TACKLE_CD;
    p.stamina = Math.max(0, p.stamina - CFG.COST_TACKLE);
  }
  startSlide(p) {
    this.startAction(p, 'slide', p.moveDir, CFG.SLIDE_DUR);
    p.cd.slide = CFG.SLIDE_CD;
    p.stamina = Math.max(0, p.stamina - CFG.COST_SLIDE);
  }
  startDash(p) {
    this.startAction(p, 'dash', this.lungeDir(p), CFG.DASH_DUR);
    p.cd.dash = CFG.DASH_CD;
    p.stamina = Math.max(0, p.stamina - CFG.COST_DASH);
  }
  startDribble(p) {
    this.startAction(p, 'dribble', this.lungeDir(p), CFG.DRIBBLE_DUR);
    p.cd.dribble = CFG.DRIBBLE_CD;
    p.stamina = Math.max(0, p.stamina - CFG.COST_DRIBBLE);
  }

  runAction(p, dt) {
    const a = p.action;
    a.t += dt;
    const k = Math.max(0, 1 - a.t / a.dur);
    switch (a.type) {
      case 'tackle': p.vel = V.mul(a.dir, CFG.TACKLE_SPEED * k); this.tackleHit(p, a); break;
      case 'slide': p.vel = V.mul(a.dir, CFG.SLIDE_SPEED * k); this.slideHit(p, a); break;
      case 'dash': p.vel = V.mul(a.dir, CFG.SPEED * CFG.MUL_DEF * CFG.DASH_MUL); break;
      case 'dribble': p.vel = V.mul(a.dir, CFG.SPEED_BALL * CFG.DRIBBLE_MUL); break;
      case 'gkdive': p.vel = V.mul(a.dir, CFG.GK_DIVE_SPEED * k); this.diveHit(p, a); break;
    }
    if (a.t >= a.dur) this.endAction(p);
  }

  endAction(p) {
    const a = p.action;
    p.action = null;
    switch (a.type) {
      case 'tackle': if (!a.hit) p.recover = CFG.TACKLE_MISS_RECOVER; break;
      case 'slide':
        if (a.hit || a.body) p.getup = CFG.SLIDE_RECOVER;
        else { p.getup = CFG.SLIDE_MISS_RECOVER; p.stamina = Math.max(0, p.stamina - CFG.COST_SLIDE_MISS); }
        break;
      case 'gkdive': p.getup = CFG.GK_DIVE_RECOVER; p.cd.dive = CFG.GK_DIVE_CD; break;
    }
  }

  tackleHit(p, a) {
    if (a.rolled) return;
    const ball = this.ball;
    const center = V.add(p.pos, V.mul(a.dir, p.r * 0.5));
    if (V.dist(center, ball.pos) >= p.r + CFG.TACKLE_REACH + ball.r) return;
    const owner = ball.owner;
    if (!owner) { this.take(p, false); a.hit = true; a.rolled = true; this.events.push({ type: 'tackle-win', p }); return; }
    if (owner.team === p.team || owner.held) return;
    a.rolled = true;
    const immune = owner.action && (owner.action.type === 'dribble' || owner.action.type === 'dash');
    const chance = owner.stance === 'drib' ? CFG.TACKLE_CHANCE_DRIBBLE : CFG.TACKLE_CHANCE;
    if (!immune && this.rng() < chance) {
      owner.cd.grab = 0.45; owner.recover = 0.35;
      this.take(p, false);
      a.hit = true;
      this.events.push({ type: 'steal', p, victim: owner });
    } else {
      this.events.push({ type: 'tackle-fail', p, victim: owner });
    }
  }

  slideHit(p, a) {
    const ball = this.ball;
    const center = V.add(p.pos, V.mul(a.dir, p.r * 0.5));
    if (!a.hit) {
      const owner = ball.owner;
      const stealable = !owner || (owner.team !== p.team && !owner.held);
      if (stealable && V.dist(center, ball.pos) < p.r + CFG.SLIDE_REACH + ball.r) {
        if (owner) owner.cd.grab = 0.35;
        ball.owner = null;
        ball.vel = V.add(V.mul(a.dir, CFG.SLIDE_KNOCK), V.mul(p.vel, 0.2));
        ball.spin = 0; ball.lastTouch = p; ball.lastTeam = p.team;
        p.cd.grab = 0.3; p.cd.through = CFG.PASS_THROUGH;
        a.hit = true;
        this.events.push({ type: 'slide-hit', p });
      }
    }
    for (const o of this.players) {
      // só derruba o portador da bola (bola dominada no pé); sem bola o corpo só empurra
      if (!o.active || o.team === p.team || o.fallen > 0 || o.held || a.bodyHit.includes(o.id)) continue;
      if (ball.owner !== o) continue;
      if (V.dist(p.pos, o.pos) >= p.r + o.r + 2) continue;
      a.bodyHit.push(o.id); a.body = true;
      o.fallen = CFG.FALL_DUR; o.action = null; o.charge = null;
      o.vel = V.mul(a.dir, 150);
      if (ball.owner === o) {
        ball.owner = null; o.cd.grab = 0.5;
        ball.vel = V.add(V.mul(a.dir, 250), V.mul(o.vel, 0.5));
      }
      this.events.push({ type: 'knockdown', p, victim: o });
    }
  }

  // mergulho do goleiro: agarra (ou espalma se muito forte) dentro da área
  diveHit(p, a) {
    if (a.hit) return;
    const ball = this.ball;
    const owner = ball.owner;
    if (owner && (owner.team === p.team || owner.held)) return;
    if (V.dist(p.pos, ball.pos) >= p.r + CFG.GK_DIVE_REACH + ball.r) return;
    a.hit = true;
    const speed = V.len(ball.vel);
    if (owner) owner.cd.grab = 0.4;
    if (this.inOwnBox(p) && speed < CFG.GK_PARRY_SPEED) {
      this.take(p, true);
      this.events.push({ type: 'save', p });
    } else {
      ball.owner = null;
      const n = V.norm(V.sub(ball.pos, p.pos));
      ball.vel = V.add(V.mul(V.reflect(ball.vel, n), 0.45), V.mul(a.dir, 260));
      ball.spin = 0; ball.lastTouch = p; ball.lastTeam = p.team;
      p.cd.grab = 0.3;
      this.events.push({ type: 'parry', p });
    }
  }

  // ---------- posse / chutes ----------
  take(p, hands) {
    const b = this.ball;
    b.owner = p; b.spin = 0; b.lastTouch = p; b.lastTeam = p.team;
    p.held = !!hands; p.holdT = 0;
    p.touchChain = 0;
  }

  kick(p, dir, speed, spin) {
    const b = this.ball;
    b.owner = null; p.held = false;
    b.vel = V.add(V.mul(dir, speed), V.mul(p.vel, 0.1));
    b.spin = spin; b.lastTouch = p; b.lastTeam = p.team;
    p.cd.grab = CFG.KICK_COOLDOWN; p.cd.through = CFG.PASS_THROUGH;
  }

  shoot(p, c) {
    const power = Math.min(1, c.t / CFG.CHARGE_MAX);
    const speed = CFG.SHOT_MIN + (CFG.SHOT_MAX - CFG.SHOT_MIN) * power;
    const spin = c.spin * (1 - CFG.SPIN_POWER_FADE * power);
    this.kick(p, c.dir0, speed, spin);
    p.recover = 0.1 + 0.2 * power;
    this.events.push({ type: 'shot', p, power, spin });
  }

  passTarget(p, dir, deg) {
    let best = null, bestAng = (deg * Math.PI) / 180;
    for (const m of this.players) {
      if (!m.active || m.team !== p.team || m === p) continue;
      const d = V.sub(m.pos, this.ball.pos);
      if (V.len(d) < 60) continue;
      const ang = Math.abs(V.angleDiff(V.angle(dir), V.angle(d)));
      if (ang < bestAng) { bestAng = ang; best = m; }
    }
    return best;
  }

  pass(p, c) {
    let dir = this.kickDir(p);
    const power = Math.min(1, c.t / CFG.PASS_CHARGE);
    let speed = CFG.PASS_MIN + (CFG.PASS_MAX - CFG.PASS_MIN) * power;
    const target = this.passTarget(p, dir, CFG.PASS_ASSIST_DEG);
    if (target) {
      const lead = V.add(target.pos, V.mul(target.vel, 0.3));
      dir = V.norm(V.sub(lead, this.ball.pos));
      speed = V.clamp(this.speedForDistance(V.dist(this.ball.pos, lead) + 30), CFG.PASS_MIN * 0.8, CFG.PASS_MAX);
    }
    this.kick(p, dir, speed, 0);
    this.events.push({ type: 'pass', p, target });
  }

  throwBall(p) {
    let dir = this.kickDir(p);
    let speed = CFG.GK_THROW;
    const target = this.passTarget(p, dir, CFG.GK_THROW_ASSIST_DEG);
    if (target) {
      const lead = V.add(target.pos, V.mul(target.vel, 0.35));
      dir = V.norm(V.sub(lead, this.ball.pos));
      speed = V.clamp(this.speedForDistance(V.dist(this.ball.pos, lead) + 40), 400, CFG.GK_THROW);
    }
    this.kick(p, dir, speed, 0);
    this.events.push({ type: 'throw', p, target });
  }

  push(p, strong) {
    const b = this.ball;
    const dir = this.lungeDir(p);
    const speed = Math.max(V.len(p.vel) + 60, strong ? CFG.PUSH_SPEED : CFG.PUSH_WALK_SPEED);
    b.owner = null; p.held = false;
    b.vel = V.mul(dir, speed); b.spin = 0; b.lastTouch = p; b.lastTeam = p.team;
    p.cd.grab = CFG.PUSH_COOLDOWN; p.cd.through = CFG.PASS_THROUGH;
    p.pushFlash = 0.35;
    this.events.push({ type: 'push', p, strong });
  }

  // executa a ação agendada na zona de ação no instante do toque
  fireQueued(p) {
    const q = p.queued;
    p.queued = null;
    p.touchChain++;
    if (q.kind === 'shot') {
      this.shoot(p, q);
    } else if (q.kind === 'pass') {
      this.pass(p, q);
    } else {
      this.push(p, p.sprinting);
    }
    this.events.push({ type: 'first-touch', p, kind: q.kind });
  }

  // distância percorrida por uma bola solta com velocidade inicial v
  static travel(v) {
    let s = v, x = 0;
    const dt = 1 / 30;
    for (let t = 0; t < 5 && s > 1; t += dt) {
      s *= Math.exp(-CFG.BALL_DRAG * dt);
      s = Math.max(0, s - CFG.BALL_DECEL * dt);
      x += s * dt;
    }
    return x;
  }
  speedForDistance(d) {
    let lo = 150, hi = CFG.BALL_MAX;
    for (let i = 0; i < 14; i++) {
      const mid = (lo + hi) / 2;
      if (Game.travel(mid) < d) lo = mid; else hi = mid;
    }
    return hi;
  }

  // ---------- física da bola ----------
  // Integra um estado {pos, vel, spin} livre (usado pela bola e pelo preview).
  static integrateFree(b, dt) {
    if (b.spin !== 0) {
      const s = V.len(b.vel);
      b.vel = V.rot(b.vel, b.spin * dt * Math.min(1, s / 500));
      b.spin *= Math.exp(-CFG.SPIN_DECAY * dt);
      if (Math.abs(b.spin) < 0.01) b.spin = 0;
    }
    let s = V.len(b.vel);
    if (s > CFG.BALL_MAX) { b.vel = V.mul(b.vel, CFG.BALL_MAX / s); s = CFG.BALL_MAX; }
    if (s > 0) {
      const s2 = Math.max(0, s * Math.exp(-CFG.BALL_DRAG * dt) - CFG.BALL_DECEL * dt);
      b.vel = s2 > 0 ? V.mul(b.vel, s2 / s) : { x: 0, y: 0 };
    }
    b.pos = V.add(b.pos, V.mul(b.vel, dt));
    Game.wallsFree(b);
  }

  static wallsFree(b) {
    const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2, r = b.r, B = CFG.BALL_BOUNCE;
    const mouth = Math.abs(b.pos.y) < CFG.GOAL_W / 2 - r;
    // paredes laterais (x)
    if (b.pos.x - r < -W2) {
      if (!mouth && b.pos.x > -W2) { b.pos.x = -W2 + r; if (b.vel.x < 0) b.vel.x = -b.vel.x * B; }
      else if (b.pos.x - r < -W2 - CFG.GOAL_D) { b.pos.x = -W2 - CFG.GOAL_D + r; if (b.vel.x < 0) b.vel.x = -b.vel.x * B; }
    }
    if (b.pos.x + r > W2) {
      if (!mouth && b.pos.x < W2) { b.pos.x = W2 - r; if (b.vel.x > 0) b.vel.x = -b.vel.x * B; }
      else if (b.pos.x + r > W2 + CFG.GOAL_D) { b.pos.x = W2 + CFG.GOAL_D - r; if (b.vel.x > 0) b.vel.x = -b.vel.x * B; }
    }
    // dentro do gol: laterais da rede
    if (Math.abs(b.pos.x) > W2) {
      const lim = CFG.GOAL_W / 2 - r;
      if (b.pos.y > lim) { b.pos.y = lim; if (b.vel.y > 0) b.vel.y = -b.vel.y * B; }
      if (b.pos.y < -lim) { b.pos.y = -lim; if (b.vel.y < 0) b.vel.y = -b.vel.y * B; }
    } else {
      if (b.pos.y + r > H2) { b.pos.y = H2 - r; if (b.vel.y > 0) b.vel.y = -b.vel.y * B; }
      if (b.pos.y - r < -H2) { b.pos.y = -H2 + r; if (b.vel.y < 0) b.vel.y = -b.vel.y * B; }
    }
    // traves
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      const post = { x: sx * W2, y: sy * CFG.GOAL_W / 2 };
      const d = V.dist(b.pos, post), min = r + CFG.POST_R;
      if (d < min && d > 1e-6) {
        const n = V.norm(V.sub(b.pos, post));
        b.pos = V.add(post, V.mul(n, min));
        if (V.dot(b.vel, n) < 0) b.vel = V.mul(V.reflect(b.vel, n), B);
      }
    }
  }

  updateBall(dt) {
    const b = this.ball;
    const o = b.owner;
    if (o) {
      if (o.held) {
        b.pos = V.add(o.pos, V.mul(o.facing, o.r * 0.45));
        b.vel = o.vel;
      } else {
        let dir = o.moving ? o.moveDir : o.facing;
        let dist = o.r + b.r + CFG.CARRY_DIST;
        if (o.stance === 'drib') {
          // protege a bola do adversário mais próximo
          let near = null, nd = 140;
          for (const q of this.players) {
            if (!q.active || q.team === o.team) continue;
            const d = V.dist(q.pos, o.pos);
            if (d < nd) { nd = d; near = q; }
          }
          if (near) dir = V.norm(V.add(V.mul(dir, 0.4), V.mul(V.norm(V.sub(o.pos, near.pos)), 0.6)));
          dist -= 2;
        }
        // a bola é conduzida fisicamente: velocidade relativa ao dono limitada,
        // sem teleporte, e nunca atravessa o corpo dele
        const target = V.add(o.pos, V.mul(dir, dist));
        let rel = V.mul(V.sub(target, b.pos), CFG.CARRY_K);
        const rl = V.len(rel);
        if (rl > CFG.CARRY_MAX_REL) rel = V.mul(rel, CFG.CARRY_MAX_REL / rl);
        b.vel = V.lerp(b.vel, V.add(o.vel, rel), Math.min(1, CFG.CARRY_LERP * dt));
        b.pos = V.add(b.pos, V.mul(b.vel, dt));
        const dd = V.dist(b.pos, o.pos), minD = o.r + b.r;
        if (dd < minD) {
          const n = dd > 1e-6 ? V.norm(V.sub(b.pos, o.pos)) : dir;
          b.pos = V.add(o.pos, V.mul(n, minD));
        }
        b.pos = Game.clampArena(b.pos, b.r);
        // perdeu o controle (bola ficou longe demais do pé)
        if (V.dist(b.pos, o.pos) > o.r + b.r + CFG.CARRY_LOSE) { b.owner = null; o.cd.grab = 0.1; }
      }
      b.spin = 0;
    } else {
      Game.integrateFree(b, dt);
    }
    b.rot += V.len(b.vel) * dt / b.r;
  }

  // preview de trajetória (para o HUD)
  simulatePath(pos, vel, spin, steps, dt) {
    const s = { pos: { x: pos.x, y: pos.y }, vel: { x: vel.x, y: vel.y }, spin, r: this.ball.r };
    const pts = [];
    for (let i = 0; i < steps; i++) { Game.integrateFree(s, dt); pts.push({ x: s.pos.x, y: s.pos.y }); if (V.len(s.vel) < 20) break; }
    return pts;
  }

  // ---------- contatos jogador x bola ----------
  contacts() {
    const b = this.ball;
    for (const p of this.players) {
      if (!p.active) continue;
      if (b.owner === p) continue;
      if (b.owner && b.owner.held) continue;
      const d = V.dist(p.pos, b.pos);
      const free = p.fallen <= 0 && p.getup <= 0 && p.cd.grab <= 0;
      if (b.owner) {
        // bola dominada por outro: encostar nunca rouba (só tackle ou carrinho).
        // Sem postura: o corpo só desvia a bola. Com postura defensiva: se a bola
        // for passar por dentro de você, ela sai do domínio do adversário (fica solta).
        const hb = this.ballHitbox(p);
        if (d < hb + b.r) {
          const o = b.owner;
          const n = d > 1e-6 ? V.norm(V.sub(b.pos, p.pos)) : { x: 1, y: 0 };
          b.pos = V.add(p.pos, V.mul(n, hb + b.r + 0.5));
          if (p.stance === 'def' && o.team !== p.team && free && !p.action) {
            b.owner = null; o.cd.grab = 0.35;
            b.vel = V.add(V.mul(n, 60), V.mul(o.vel, 0.5));
            b.lastTouch = p; b.lastTeam = p.team;
            this.events.push({ type: 'block', p, victim: o });
          }
        }
        continue;
      }
      // bola acabou de sair do pé deste jogador: passa entre as pernas
      if (b.lastTouch === p && p.cd.through > 0) continue;
      const canGrab = free && (!p.action || p.action.type === 'dash');
      const hb = this.ballHitbox(p);
      const R = hb + b.r;
      if (d >= R) continue;
      if (canGrab && p.queued) { this.fireQueued(p); continue; }   // toque de primeira
      const speed = V.len(b.vel);
      const hands = p.isKeeper && this.inOwnBox(p);
      const limit = hands ? (p.stance === 'def' ? CFG.GK_PARRY_SPEED * 1.3 : CFG.GK_PARRY_SPEED)
        : (p.stance === 'def' ? CFG.CONTROL_MAX_DEF : CFG.CONTROL_MAX);
      if (canGrab && speed <= limit) {
        this.take(p, hands);
        this.events.push({ type: hands ? 'catch' : 'control', p });
      } else {
        // rebate no corpo
        const n = d > 1e-6 ? V.norm(V.sub(b.pos, p.pos)) : { x: 1, y: 0 };
        b.pos = V.add(p.pos, V.mul(n, hb + b.r + 0.5));
        const rel = V.sub(b.vel, p.vel);
        if (V.dot(rel, n) < 0) b.vel = V.add(V.mul(V.reflect(rel, n), CFG.DEFLECT_BOUNCE), p.vel);
        b.spin *= 0.3; b.lastTouch = p; b.lastTeam = p.team;
        if (canGrab) { p.cd.grab = CFG.DEFLECT_COOLDOWN; this.events.push({ type: 'deflect', p }); }
      }
    }
  }

  // ---------- luvas (goleiro) ----------
  updateGloves(dt) {
    for (let team = 0; team < 2; team++) {
      const keeper = this.players.find((p) => p.team === team && p.isKeeper);
      const inBox = this.players
        .filter((p) => p.active && p.team === team && !p.isKeeper && p.fallen <= 0 && this.inOwnBox(p))
        .sort((a, b) => Math.abs(a.pos.x - this.goalX(team)) - Math.abs(b.pos.x - this.goalX(team)));
      if (!keeper) {
        if (inBox.length) { inBox[0].isKeeper = true; inBox[0].cd.gloves = 1.5; this.events.push({ type: 'gloves', p: inBox[0] }); }
      } else if (!this.inOwnBox(keeper) && keeper.cd.gloves <= 0 && inBox.length) {
        keeper.isKeeper = false; keeper.held = false;
        inBox[0].isKeeper = true; inBox[0].cd.gloves = 1.5;
        this.events.push({ type: 'gloves', p: inBox[0] });
      }
    }
  }

  // ---------- jogadores ----------
  collidePlayers() {
    const ps = this.players;
    for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++) {
      const a = ps[i], c = ps[j];
      if (!a.active || !c.active) continue;
      const d = V.dist(a.pos, c.pos), min = a.r + c.r;
      if (d >= min || d < 1e-6) continue;
      const n = V.norm(V.sub(c.pos, a.pos));
      const push = (min - d) / 2;
      a.pos = V.sub(a.pos, V.mul(n, push));
      c.pos = V.add(c.pos, V.mul(n, push));
      const rel = V.dot(V.sub(c.vel, a.vel), n);
      if (rel < 0) { a.vel = V.add(a.vel, V.mul(n, rel * 0.5)); c.vel = V.sub(c.vel, V.mul(n, rel * 0.5)); }
    }
  }

  // goleiro com a bola nas mãos: adversários são empurrados para fora de um raio.
  // Não vale com a bola no pé (recuo), só quando agarrou com as luvas.
  keeperRepel(dt) {
    for (const k of this.players) {
      if (!k.active || !k.isKeeper || !k.held || this.ball.owner !== k) continue;
      const R = k.r + CFG.GK_REPEL;
      for (const o of this.players) {
        if (!o.active || o.team === k.team) continue;
        const d = V.dist(o.pos, k.pos);
        if (d >= R + o.r) continue;
        const n = d > 1e-6 ? V.norm(V.sub(o.pos, k.pos)) : { x: k.team === 0 ? 1 : -1, y: 0 };
        const gap = R + o.r - d;
        o.pos = V.add(o.pos, V.mul(n, Math.min(gap, CFG.GK_REPEL_PUSH * dt)));
        const inward = V.dot(o.vel, n);
        if (inward < 0) o.vel = V.sub(o.vel, V.mul(n, inward));
        if (o.action && (o.action.type === 'tackle' || o.action.type === 'slide')) { o.action = null; o.getup = 0.3; }
      }
    }
  }

  clampPlayer(p) {
    p.pos = Game.clampArena(p.pos, p.r);
    if (p.held && this.ball.owner === p) {
      const W2 = CFG.FIELD_W / 2;
      const lo = p.team === 0 ? -W2 + p.r : W2 - CFG.BOX_W + p.r;
      const hi = p.team === 0 ? -W2 + CFG.BOX_W - p.r : W2 - p.r;
      p.pos.x = V.clamp(p.pos.x, lo, hi);
      p.pos.y = V.clamp(p.pos.y, -CFG.BOX_H / 2 + p.r, CFG.BOX_H / 2 - p.r);
    }
  }

  // ---------- gol / fim ----------
  checkGoal() {
    const b = this.ball, W2 = CFG.FIELD_W / 2;
    if (Math.abs(b.pos.y) >= CFG.GOAL_W / 2) return;
    if (b.pos.x < -W2 - b.r) this.goal(1);
    else if (b.pos.x > W2 + b.r) this.goal(0);
  }
  goal(team) {
    this.score[team]++;
    this.kickoffTeam = 1 - team;
    this.state = 'goal';
    this.stateT = CFG.GOAL_PAUSE;
    const scorer = this.ball.owner || this.ball.lastTouch;
    if (this.ball.owner) { this.ball.lastTouch = this.ball.owner; this.ball.owner = null; this.ball.vel = { x: 0, y: 0 }; }
    const own = scorer && scorer.team !== team;
    if (scorer && !own) {
      scorer.stats.goals++;
      const pv = this.ball.prevTouch;
      if (pv && pv !== scorer && pv.team === team) pv.stats.assists++;
    }
    this.msg = `GOL ${CFG.TEAM_NAMES[team].toUpperCase()}!` + (scorer ? (own ? ` (contra, ${scorer.name})` : ` (${scorer.name})`) : '');
    for (const p of this.players) { p.charge = null; }
    this.events.push({ type: 'goal', team, scorer });
    if (Math.abs(this.score[0] - this.score[1]) >= CFG.MERCY) this.endMatch();
  }
  endMatch() {
    this.state = 'end';
    const s = this.score;
    this.msg = s[0] === s[1] ? 'FIM: EMPATE' : `FIM: ${CFG.TEAM_NAMES[s[0] > s[1] ? 0 : 1].toUpperCase()} VENCEU`;
    this.events.push({ type: 'end' });
  }
}
