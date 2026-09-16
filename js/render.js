'use strict';
// Desenho em Canvas 2D: campo, jogadores, bola, HUD.
class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.cam = { x: 0, y: 0, zoom: 1 };
    this.showHelp = true;
    this.flash = [];       // textos flutuantes {text, pos, t}
    this.w = 1; this.h = 1; this.dpr = 1;
    this.resize();
  }

  resize() {
    this.dpr = window.devicePixelRatio || 1;
    this.w = window.innerWidth; this.h = window.innerHeight;
    this.canvas.width = Math.floor(this.w * this.dpr);
    this.canvas.height = Math.floor(this.h * this.dpr);
    this.canvas.style.width = this.w + 'px';
    this.canvas.style.height = this.h + 'px';
  }

  updateCamera(game, human, dt) {
    const focus = human ? V.lerp(human.pos, game.ball.pos, 0.4) : game.ball.pos;
    const fitZoom = Math.min(this.w / (CFG.FIELD_W + 2 * CFG.GOAL_D + 60), this.h / (CFG.FIELD_H + 60));
    const zoom = Math.max(fitZoom, this.h / 900);
    this.cam.zoom += (zoom - this.cam.zoom) * Math.min(1, 4 * dt);
    const halfW = this.w / this.cam.zoom / 2, halfH = this.h / this.cam.zoom / 2;
    const limX = Math.max(0, CFG.FIELD_W / 2 + CFG.GOAL_D + 30 - halfW);
    const limY = Math.max(0, CFG.FIELD_H / 2 + 30 - halfH);
    const tx = V.clamp(focus.x, -limX, limX), ty = V.clamp(focus.y, -limY, limY);
    const k = Math.min(1, 5 * dt);
    this.cam.x += (tx - this.cam.x) * k;
    this.cam.y += (ty - this.cam.y) * k;
  }

  screenToWorld(sx, sy) {
    return { x: (sx - this.w / 2) / this.cam.zoom + this.cam.x, y: (sy - this.h / 2) / this.cam.zoom + this.cam.y };
  }

  addFlash(text, pos, color) { this.flash.push({ text, pos: { x: pos.x, y: pos.y }, t: 0, color: color || '#fff' }); }

  draw(game, human, dt) {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = this.showValue ? '#ffffff' : '#1b2a1c';   // na camada de análise, fundo branco e linhas pretas
    ctx.fillRect(0, 0, this.w, this.h);

    ctx.save();
    ctx.translate(this.w / 2, this.h / 2);
    ctx.scale(this.cam.zoom, this.cam.zoom);
    ctx.translate(-this.cam.x, -this.cam.y);
    this.drawPitch();
    if (this.showValue) this.drawValueOverlay(game);
    for (const p of game.players) if (p.active && (p.fallen > 0 || p.getup > 0)) this.drawPlayer(game, p, human);
    for (const p of game.players) if (p.active && !(p.fallen > 0 || p.getup > 0)) this.drawPlayer(game, p, human);
    if (human) this.drawAim(game, human);
    this.drawBall(game, human);
    this.drawFlashes(dt);
    ctx.restore();

    this.drawHUD(game, human);
  }

  // ---------- campo ----------
  drawPitch() {
    const ctx = this.ctx;
    const W2 = CFG.FIELD_W / 2, H2 = CFG.FIELD_H / 2;
    const analysis = !!this.showValue;
    const line = analysis ? 'rgba(0,0,0,0.9)' : 'rgba(255,255,255,0.85)';
    // grama com listras (ou branco na camada de análise)
    ctx.fillStyle = analysis ? '#ffffff' : '#3f8f3f';
    ctx.fillRect(-W2 - CFG.GOAL_D - 40, -H2 - 40, CFG.FIELD_W + 2 * CFG.GOAL_D + 80, CFG.FIELD_H + 80);
    if (!analysis) {
      const stripes = 14, sw = CFG.FIELD_W / stripes;
      for (let i = 0; i < stripes; i++) {
        ctx.fillStyle = i % 2 ? '#3d883d' : '#459945';
        ctx.fillRect(-W2 + i * sw, -H2, sw, CFG.FIELD_H);
      }
    }
    // linhas
    ctx.strokeStyle = line;
    ctx.lineWidth = 3;
    ctx.strokeRect(-W2, -H2, CFG.FIELD_W, CFG.FIELD_H);
    ctx.beginPath(); ctx.moveTo(0, -H2); ctx.lineTo(0, H2); ctx.stroke();
    ctx.beginPath(); ctx.arc(0, 0, CFG.FIELD_H * 0.11, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(0, 0, 4, 0, Math.PI * 2); ctx.fillStyle = analysis ? '#000' : '#fff'; ctx.fill();
    for (const s of [-1, 1]) {
      const x0 = s * W2, x1 = s * (W2 - CFG.BOX_W);
      ctx.beginPath();
      ctx.moveTo(x0, -CFG.BOX_H / 2); ctx.lineTo(x1, -CFG.BOX_H / 2); ctx.lineTo(x1, CFG.BOX_H / 2); ctx.lineTo(x0, CFG.BOX_H / 2);
      ctx.stroke();
      ctx.beginPath(); ctx.arc(s * (W2 - CFG.BOX_W + 60), 0, 4, 0, Math.PI * 2); ctx.fill();
      // gol: rede
      const gx = s > 0 ? W2 : -W2 - CFG.GOAL_D;
      ctx.fillStyle = analysis ? 'rgba(0,0,0,0.08)' : 'rgba(0,0,0,0.35)';
      ctx.fillRect(gx, -CFG.GOAL_W / 2, CFG.GOAL_D, CFG.GOAL_W);
      ctx.strokeStyle = analysis ? 'rgba(0,0,0,0.35)' : 'rgba(255,255,255,0.35)';
      ctx.lineWidth = 1;
      for (let y = -CFG.GOAL_W / 2; y <= CFG.GOAL_W / 2; y += 12) { ctx.beginPath(); ctx.moveTo(gx, y); ctx.lineTo(gx + CFG.GOAL_D, y); ctx.stroke(); }
      for (let x = gx; x <= gx + CFG.GOAL_D; x += 12) { ctx.beginPath(); ctx.moveTo(x, -CFG.GOAL_W / 2); ctx.lineTo(x, CFG.GOAL_W / 2); ctx.stroke(); }
      ctx.strokeStyle = line; ctx.lineWidth = 3;
      ctx.strokeRect(gx, -CFG.GOAL_W / 2, CFG.GOAL_D, CFG.GOAL_W);
      // traves
      ctx.fillStyle = analysis ? '#000' : '#eee';
      for (const sy of [-1, 1]) { ctx.beginPath(); ctx.arc(s * W2, sy * CFG.GOAL_W / 2, CFG.POST_R, 0, Math.PI * 2); ctx.fill(); }
    }
    // paredes
    ctx.strokeStyle = 'rgba(20,40,60,0.9)';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(-W2, -H2); ctx.lineTo(W2, -H2); ctx.lineTo(W2, -CFG.GOAL_W / 2);
    ctx.moveTo(W2, CFG.GOAL_W / 2); ctx.lineTo(W2, H2); ctx.lineTo(-W2, H2); ctx.lineTo(-W2, CFG.GOAL_W / 2);
    ctx.moveTo(-W2, -CFG.GOAL_W / 2); ctx.lineTo(-W2, -H2);
    ctx.stroke();
    ctx.strokeStyle = analysis ? 'rgba(0,0,0,0.6)' : 'rgba(120,200,255,0.25)';
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  // ---------- depuração: controle de campo x valor, e candidatos de movimento dos bots ----------
  drawValueOverlay(game) {
    const ctx = this.ctx;
    if (typeof Features === 'undefined' || !Features.pitchControlTT) return;
    const pc = Features.pitchControlTT(game);
    const GX = pc.TX, GY = pc.TY, cw = CFG.FIELD_W / GX, ch = CFG.FIELD_H / GY;
    const VM = (typeof VALUE_MAP !== 'undefined') ? VALUE_MAP : null;
    const valueAt = (x, y, dir) => {
      if (!VM) return 0.1;
      const i = Math.min(VM.GX - 1, Math.max(0, Math.floor((x * dir + CFG.FIELD_W / 2) / (CFG.FIELD_W / VM.GX))));
      const j = Math.min(VM.GY - 1, Math.max(0, Math.floor((y + CFG.FIELD_H / 2) / (CFG.FIELD_H / VM.GY))));
      return VM.v[j * VM.GX + i];
    };
    let vmax = 0.01; if (VM) for (const v of VM.v) vmax = Math.max(vmax, v);
    const mode = this.showValue;   // 1 = território (controle + alcance do passe), 2 = perigo (controle x valor x passe)
    const ownerTeam = game.ball.owner ? game.ball.owner.team : -1;
    const total = [0, 0];
    for (let j = 0; j < GY; j++) for (let i = 0; i < GX; i++) {
      const k = j * GX + i;
      const p0 = pc.p0[k];   // território: tempo de chegada relativo à bola
      const v0 = valueAt(pc.cx[i], pc.cy[j], 1), v1 = valueAt(pc.cx[i], pc.cy[j], -1);
      total[0] += p0 * v0; total[1] += (1 - p0) * v1;
      const blue = p0 >= 0.5, conf = Math.abs(p0 - 0.5) * 2, v = blue ? v0 : v1;
      const col = blue ? '80,140,255' : '255,110,90';
      let a;
      if (mode === 1) a = 0.06 + 0.55 * conf;
      else a = 0.04 + 0.65 * conf * (v / vmax);
      ctx.fillStyle = `rgba(${col},${a})`;
      ctx.fillRect(pc.cx[i] - cw / 2, pc.cy[j] - ch / 2, cw - 0.5, ch - 0.5);
    }
    // candidatos avaliados por cada bot (estilo controle de campo): pontos e ganho de valor
    ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
    for (const p of game.players) {
      if (!p.active || !p.ai || !p.ai.cands || !p.ai.cands.length) continue;
      const cands = p.ai.cands.slice().sort((a, b) => b.s - a.s);
      const top = cands.slice(0, 6);
      for (let k = 0; k < top.length; k++) {
        const cnd = top[k];
        ctx.fillStyle = k === 0 ? 'rgba(255,200,0,0.95)' : 'rgba(0,0,0,0.35)';
        ctx.beginPath(); ctx.arc(cnd.x, cnd.y, k === 0 ? 7 : 4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = k === 0 ? '#a06000' : 'rgba(0,0,0,0.7)';
        ctx.fillText((cnd.v >= 0 ? '+' : '') + (cnd.v * 100).toFixed(1), cnd.x, cnd.y - 9);
      }
      if (top.length) { ctx.strokeStyle = 'rgba(255,255,0,0.5)'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(p.pos.x, p.pos.y); ctx.lineTo(top[0].x, top[0].y); ctx.stroke(); }
    }
    ctx.fillStyle = '#000'; ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'left';
    const label = mode === 1 ? 'TERRITÓRIO: quem chega antes, relativo ao tempo mínimo da bola (com velocidade)' : 'PERIGO: território x valor (chance de gol em 8 s) = potencial da jogada';
    ctx.fillText(`${label}  ·  valor alcançável  azul ${(total[0] * 100).toFixed(1)}  vermelho ${(total[1] * 100).toFixed(1)}   (V alterna / desliga)`, -CFG.FIELD_W / 2 + 10, -CFG.FIELD_H / 2 - 14);
  }

  // ---------- jogadores ----------
  drawPlayer(game, p, human) {
    const ctx = this.ctx;
    const color = CFG.TEAM_COLORS[p.team];
    const down = p.fallen > 0 || p.getup > 0;
    const sliding = p.action && (p.action.type === 'slide' || p.action.type === 'gkdive');
    ctx.save();
    ctx.translate(p.pos.x, p.pos.y);
    // sombra
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.beginPath(); ctx.ellipse(3, 4, p.r, p.r * 0.85, 0, 0, Math.PI * 2); ctx.fill();

    if (sliding || down) {
      // corpo deitado/deslizando: elipse alongada
      const dir = p.action ? p.action.dir : p.moveDir;
      ctx.rotate(V.angle(dir));
      ctx.globalAlpha = down && !sliding ? 0.7 : 1;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.ellipse(0, 0, p.r * 1.5, p.r * 0.75, 0, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.lineWidth = 2; ctx.stroke();
      if (sliding) {
        const isSlide = p.action.type === 'slide';
        ctx.strokeStyle = isSlide ? 'rgba(255,255,255,0.35)' : 'rgba(255,220,120,0.5)'; ctx.lineWidth = isSlide ? 6 : 3;
        ctx.beginPath(); ctx.moveTo(-p.r * 1.5, 0); ctx.lineTo(-p.r * (isSlide ? 3.5 : 2.4), 0); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      ctx.rotate(-V.angle(dir));
    } else {
      // postura
      if (p.stance === 'def') { ctx.strokeStyle = 'rgba(90,200,255,0.9)'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(0, 0, game.ballHitbox(p), 0, Math.PI * 2); ctx.stroke(); }
      if (p.dribbleLag > 0) { ctx.strokeStyle = 'rgba(200,200,200,0.7)'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(0, 0, p.r + 6, 0, Math.PI * 2 * (p.dribbleLag / CFG.DRIBBLE_LAG)); ctx.stroke(); }
      else if (p.stance === 'drib') { ctx.strokeStyle = 'rgba(255,190,60,0.95)'; ctx.lineWidth = 3; ctx.setLineDash([6, 5]); ctx.beginPath(); ctx.arc(0, 0, p.r + 6, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); }
      // tackle / dash: rastro
      if (p.action) {
        ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 5;
        ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-p.action.dir.x * p.r * 2.2, -p.action.dir.y * p.r * 2.2); ctx.stroke();
      }
      // corpo
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(0, 0, p.r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = p.exhausted ? 'rgba(220,50,50,0.9)' : (p.recover > 0 ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.55)');
      ctx.lineWidth = 2; ctx.stroke();
      // goleiro com a bola nas mãos: zona de repulsão
      if (p.isKeeper && p.held && game.ball.owner === p) {
        ctx.strokeStyle = 'rgba(255,216,74,0.35)'; ctx.lineWidth = 2; ctx.setLineDash([5, 6]);
        ctx.beginPath(); ctx.arc(0, 0, p.r + CFG.GK_REPEL, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
      }
      // goleiro: luvas
      if (p.isKeeper) {
        ctx.fillStyle = '#ffd84a';
        ctx.beginPath(); ctx.arc(0, 0, p.r * 0.5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#222'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('G', 0, 0.5);
      }
      // direção de mira
      const f = p.facing;
      ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(f.x * (p.r - 6), f.y * (p.r - 6)); ctx.lineTo(f.x * (p.r + 3), f.y * (p.r + 3)); ctx.stroke();
      // carga de chute / passe
      const ch = p.charge || (p.queued && p.queued.kind !== 'push' ? p.queued : null);
      if (ch) {
        const max = ch.kind === 'shot' ? CFG.CHARGE_MAX : CFG.PASS_CHARGE;
        const k = Math.min(1, ch.t / max);
        ctx.strokeStyle = ch.kind === 'shot' ? `rgba(255,${Math.floor(220 - 180 * k)},60,0.95)` : 'rgba(120,255,140,0.95)';
        ctx.lineWidth = 4;
        ctx.beginPath(); ctx.arc(0, 0, p.r + 11, -Math.PI / 2, -Math.PI / 2 + k * Math.PI * 2); ctx.stroke();
      }
      // extra effort
      if (p.effortT > 0) { ctx.strokeStyle = 'rgba(255,255,120,0.8)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(0, 0, p.r + 14, 0, Math.PI * 2); ctx.stroke(); }
    }
    // pedindo bola
    if (p.callT > 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('!', 0, -p.r - 26);
    }
    // controlado pelo humano
    if (p === human) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(0, 0, p.r + 3, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.moveTo(0, -p.r - 22); ctx.lineTo(-6, -p.r - 32); ctx.lineTo(6, -p.r - 32); ctx.closePath(); ctx.fill();
    }
    // nome
    ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(p.name, 0, p.r + 4);
    ctx.restore();
  }

  // ---------- mira, preview, zona de ação ----------
  drawAim(game, p) {
    const ctx = this.ctx;
    const b = game.ball;
    const hasBall = game.hasBall(p);
    const held = b.owner === p && p.held;
    const inZone = game.ballInZone(p);
    if (!(hasBall || held || inZone)) return;
    // linha de mira
    const aim = p.input.aim;
    ctx.strokeStyle = this.showValue ? 'rgba(0,0,0,0.45)' : 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 6]);
    ctx.beginPath(); ctx.moveTo(b.pos.x, b.pos.y); ctx.lineTo(aim.x, aim.y); ctx.stroke();
    ctx.setLineDash([]);
    // preview de chute com efeito
    const shotC = (p.charge && p.charge.kind === 'shot') ? p.charge : (p.queued && p.queued.kind === 'shot' ? p.queued : null);
    if (shotC) {
      const c = shotC;
      const power = Math.min(1, c.t / CFG.CHARGE_MAX);
      const speed = Game.shotSpeed(power, !!(p.queued && p.queued.kind === 'shot'));
      const spin = c.spin * (1 - CFG.SPIN_POWER_FADE * power);
      const pts = game.simulatePath(b.pos, V.mul(c.dir0, speed), spin, 40, 1 / 30);
      ctx.strokeStyle = 'rgba(255,230,120,0.7)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(b.pos.x, b.pos.y);
      for (const q of pts) ctx.lineTo(q.x, q.y);
      ctx.stroke();
    }
    // alvo do passe assistido
    if ((p.charge && p.charge.kind === 'pass') || (p.queued && p.queued.kind === 'pass')) {
      const t = game.passTarget(p, game.kickDir(p), CFG.PASS_ASSIST_DEG);
      if (t) { ctx.strokeStyle = 'rgba(120,255,140,0.9)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(t.pos.x, t.pos.y, t.r + 8, 0, Math.PI * 2); ctx.stroke(); }
    }
  }

  drawBall(game, human) {
    const ctx = this.ctx;
    const b = game.ball;
    const speed = V.len(b.vel);
    ctx.save();
    // alvo (Rematch): losango azul quando a bola solta está ao alcance; verde quando a
    // sua ação está travada nela; vermelho se outro jogador tem a prioridade
    if (human && !b.owner && (human.reach || game.ballInZone(human))) {
      const mine = human.queued && b.lock === human;
      const taken = b.lock && b.lock !== human;
      const glow = ctx.createRadialGradient(b.pos.x, b.pos.y, b.r, b.pos.x, b.pos.y, b.r + 18);
      glow.addColorStop(0, mine ? 'rgba(120,255,140,0.6)' : taken ? 'rgba(255,90,90,0.5)' : 'rgba(120,190,255,0.55)');
      glow.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(b.pos.x, b.pos.y, b.r + 18, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = mine ? 'rgba(120,255,140,0.95)' : taken ? 'rgba(255,90,90,0.9)' : 'rgba(90,170,255,0.9)';
      ctx.lineWidth = mine ? 3 : 2;
      const s = b.r + 10 + (mine ? 2 * Math.sin(performance.now() / 70) : 0);
      ctx.beginPath();
      ctx.moveTo(b.pos.x, b.pos.y - s); ctx.lineTo(b.pos.x + s, b.pos.y); ctx.lineTo(b.pos.x, b.pos.y + s); ctx.lineTo(b.pos.x - s, b.pos.y); ctx.closePath();
      ctx.stroke();
    }
    // diamante do push ball (bola nos pés + Shift), verde logo após empurrar
    if (human && ((game.hasBall(human) && human.sprinting) || human.pushFlash > 0)) {
      ctx.strokeStyle = human.pushFlash > 0 ? 'rgba(90,255,120,0.95)' : 'rgba(90,170,255,0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      const s = b.r + 9;
      ctx.moveTo(b.pos.x, b.pos.y - s); ctx.lineTo(b.pos.x + s, b.pos.y); ctx.lineTo(b.pos.x, b.pos.y + s); ctx.lineTo(b.pos.x - s, b.pos.y); ctx.closePath();
      ctx.stroke();
    }
    // sombra e bola
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.beginPath(); ctx.arc(b.pos.x + 2, b.pos.y + 3, b.r, 0, Math.PI * 2); ctx.fill();
    if (speed > 600 && !b.owner) {
      ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = b.r * 1.2;
      const t = V.mul(V.norm(b.vel), -Math.min(40, speed / 25));
      ctx.beginPath(); ctx.moveTo(b.pos.x, b.pos.y); ctx.lineTo(b.pos.x + t.x, b.pos.y + t.y); ctx.stroke();
    }
    ctx.fillStyle = '#f7f7f7';
    ctx.beginPath(); ctx.arc(b.pos.x, b.pos.y, b.r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#333'; ctx.lineWidth = 1.5; ctx.stroke();
    // marca girando (mostra rotação e efeito)
    ctx.fillStyle = '#333';
    const a = b.rot + (b.spin !== 0 ? b.spin * 3 : 0);
    ctx.beginPath(); ctx.arc(b.pos.x + Math.cos(a) * b.r * 0.5, b.pos.y + Math.sin(a) * b.r * 0.5, b.r * 0.3, 0, Math.PI * 2); ctx.fill();
    if (b.owner && b.owner.held) {
      ctx.strokeStyle = '#ffd84a'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(b.pos.x, b.pos.y, b.r + 3, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.restore();
  }

  drawFlashes(dt) {
    const ctx = this.ctx;
    ctx.font = 'bold 16px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const f of this.flash) {
      f.t += dt;
      ctx.globalAlpha = Math.max(0, 1 - f.t / 1.2);
      ctx.fillStyle = f.color;
      ctx.fillText(f.text, f.pos.x, f.pos.y - 30 - f.t * 30);
    }
    ctx.globalAlpha = 1;
    this.flash = this.flash.filter((f) => f.t < 1.2);
  }

  drawScoreboard(game, human) {
    const ctx = this.ctx;
    const w = this.w, h = this.h;
    const cols = [['Jogador', 190], ['G', 40], ['A', 40], ['Roubos', 70], ['Defesas', 70], ['Ping', 60]];
    const tw = cols.reduce((s, c) => s + c[1], 0) + 40;
    const rows = game.players.filter((p) => p.active);
    const th = 60 + rows.length * 24 + 40 + 44;
    const x0 = w / 2 - tw / 2, y0 = h / 2 - th / 2;
    ctx.fillStyle = 'rgba(0,0,0,0.78)';
    ctx.fillRect(x0, y0, tw, th);
    ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, tw, th);
    ctx.font = 'bold 20px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = CFG.TEAM_COLORS[0]; ctx.fillText(`${CFG.TEAM_NAMES[0]} ${game.score[0]}`, x0 + tw / 2 - 80, y0 + 22);
    ctx.fillStyle = '#fff'; ctx.fillText('x', x0 + tw / 2, y0 + 22);
    ctx.fillStyle = CFG.TEAM_COLORS[1]; ctx.fillText(`${game.score[1]} ${CFG.TEAM_NAMES[1]}`, x0 + tw / 2 + 80, y0 + 22);
    let y = y0 + 52;
    ctx.font = 'bold 12px sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.7)';
    let x = x0 + 20;
    for (const [name, cw] of cols) { ctx.textAlign = name === 'Jogador' ? 'left' : 'center'; ctx.fillText(name, name === 'Jogador' ? x : x + cw / 2, y); x += cw; }
    y += 18;
    ctx.font = '13px sans-serif';
    for (let team = 0; team < 2; team++) {
      for (const p of rows.filter((q) => q.team === team)) {
        if (p === human) { ctx.fillStyle = 'rgba(255,255,255,0.12)'; ctx.fillRect(x0 + 8, y - 11, tw - 16, 22); }
        x = x0 + 20;
        ctx.fillStyle = CFG.TEAM_COLORS[team]; ctx.beginPath(); ctx.arc(x + 6, y, 5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.textAlign = 'left';
        ctx.fillText(`${p.name}${p.isKeeper ? ' (G)' : ''}${p.human ? '' : ' · bot'}`, x + 18, y);
        x += cols[0][1];
        const ping = p.human ? (p.ping > 0 ? `${Math.round(p.ping)} ms` : (p === human && this.roomCode ? '—' : 'host')) : '—';
        const vals = [p.stats.goals, p.stats.assists, p.stats.steals, p.stats.saves, p.human ? ping : '—'];
        ctx.textAlign = 'center';
        vals.forEach((v, i) => { const cw = cols[i + 1][1]; ctx.fillText(String(v), x + cw / 2, y); x += cw; });
        y += 24;
      }
    }
    // botão de trocar de time
    const other = human ? 1 - human.team : 1;
    const bw = 220, bh = 30, bx = x0 + tw / 2 - bw / 2, by = y0 + th - 70;
    this.switchBtn = { x: bx, y: by, w: bw, h: bh };
    ctx.fillStyle = CFG.TEAM_COLORS[other]; ctx.fillRect(bx, by, bw, bh);
    ctx.fillStyle = '#fff'; ctx.font = 'bold 13px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(`Trocar para ${CFG.TEAM_NAMES[other]}  (T)`, bx + bw / 2, by + bh / 2);
    ctx.font = '11px sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText('G gols · A assistências · Roubos: tackles e carrinhos certos · Defesas: mergulhos do goleiro', x0 + tw / 2, y0 + th - 16);
  }

  // ---------- HUD ----------
  drawHUD(game, human) {
    const ctx = this.ctx;
    const w = this.w, h = this.h;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    // placar
    const m = Math.floor(game.time / 60), s = Math.floor(game.time % 60);
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(w / 2 - 150, 10, 300, 44);
    ctx.font = 'bold 22px sans-serif'; ctx.textBaseline = 'middle';
    ctx.textAlign = 'right'; ctx.fillStyle = CFG.TEAM_COLORS[0]; ctx.fillText(`${CFG.TEAM_NAMES[0]}  ${game.score[0]}`, w / 2 - 40, 32);
    ctx.textAlign = 'left'; ctx.fillStyle = CFG.TEAM_COLORS[1]; ctx.fillText(`${game.score[1]}  ${CFG.TEAM_NAMES[1]}`, w / 2 + 40, 32);
    ctx.textAlign = 'center'; ctx.fillStyle = '#fff'; ctx.font = 'bold 18px monospace';
    ctx.fillText(`${m}:${s.toString().padStart(2, '0')}`, w / 2, 32);

    // stamina (barra grande) e extra effort (barra pequena)
    if (human) {
      const bx = 20, by = h - 40, bw = 260, bh = 14;
      ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(bx - 4, by - 4, bw + 8 + 70, bh + 8);
      ctx.fillStyle = '#333'; ctx.fillRect(bx, by, bw, bh);
      const st = human.stamina / CFG.STAMINA_MAX;
      ctx.fillStyle = human.exhausted ? '#d33' : (st > 0.3 ? '#6fd36f' : '#e0703c'); ctx.fillRect(bx, by, bw * st, bh);
      ctx.fillStyle = '#333'; ctx.fillRect(bx + bw + 8, by, 58, bh);
      ctx.fillStyle = human.effortBar >= 1 ? '#ffe66d' : '#8d8a4a'; ctx.fillRect(bx + bw + 8, by, 58 * human.effortBar, bh);
      ctx.fillStyle = '#fff'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      let status = human.isKeeper ? 'GOLEIRO' : 'LINHA';
      if (human.stance === 'def') status += ' · postura defensiva';
      if (human.stance === 'drib') status += ' · postura de drible';
      if (human.held) status += ` · bola nas mãos ${(CFG.GK_HOLD_MAX - human.holdT).toFixed(1)}s`;
      if (human.exhausted) status += ' · EXAUSTO';
      if (human.dribbleLag > 0) status += ' · lag do drible (chute/passe/Espaço cancela)';
      if (human.recover > 0) status += ' · recuperando';
      if (human.fallen > 0) status += ' · caído';
      if (human.getup > 0) status += ' · levantando';
      ctx.fillText(`${human.name} · ${status}`, bx, by - 8);
    }

    // mensagem central
    if (game.msg) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(0, h / 2 - 40, w, 80);
      ctx.fillStyle = '#fff'; ctx.font = 'bold 40px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(game.msg, w / 2, h / 2);
      if (game.state === 'end') { ctx.font = '16px sans-serif'; ctx.fillText('R para reiniciar', w / 2, h / 2 + 30); }
    } else if (game.state === 'kickoff') {
      ctx.fillStyle = '#fff'; ctx.font = 'bold 28px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('KICKOFF', w / 2, h / 2 - 60);
    }

    // código da sala (multiplayer)
    if (this.roomCode) {
      ctx.font = 'bold 14px monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(10, 10, 150, 24);
      ctx.fillStyle = '#ffe66d'; ctx.fillText(`SALA ${this.roomCode}`, 18, 15);
    }
    // aviso de tela cheia
    if (!this.fullscreen) {
      const txt = this.keyboardLock
        ? 'Aperte Enter para tela cheia: só assim Ctrl+W não fecha a aba'
        : 'Este navegador não suporta bloqueio de teclado: use C no lugar de Ctrl';
      ctx.font = '13px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(w / 2 - 260, 60, 520, 22);
      ctx.fillStyle = '#ffe66d'; ctx.fillText(txt, w / 2, 64);
    }
    // mira virtual (o cursor do sistema fica escondido no jogo)
    if (this.cursor && !this.showScoreboard) {
      const c = this.cursor;
      const dir0 = human && ((human.charge && human.charge.kind === 'shot' && human.charge.dir0) || (human.queued && human.queued.kind === 'shot' && human.queued.dir0));
      const spin = human ? ((human.charge && human.charge.kind === 'shot') ? human.charge.spin : (human.queued && human.queued.kind === 'shot') ? human.queued.spin : 0) : 0;
      ctx.lineWidth = 1.5;
      if (this.cursorFrozen && dir0) {
        // mira travada: anel + seta lateral proporcional ao efeito
        ctx.strokeStyle = 'rgba(255,200,80,0.95)';
        ctx.beginPath(); ctx.arc(c.x, c.y, 9, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath(); ctx.arc(c.x, c.y, 2, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,200,80,0.95)'; ctx.fill();
        const k = spin / CFG.SPIN_MAX;
        if (Math.abs(k) > 0.03) {
          const perp = V.perp(dir0);
          const len = 14 + 40 * Math.abs(k), sgn = Math.sign(k);
          const ex = c.x + perp.x * len * sgn, ey = c.y + perp.y * len * sgn;
          ctx.strokeStyle = 'rgba(255,230,120,0.95)'; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(c.x + perp.x * 11 * sgn, c.y + perp.y * 11 * sgn); ctx.lineTo(ex, ey); ctx.stroke();
          ctx.beginPath(); ctx.arc(ex, ey, 3.5, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,230,120,0.95)'; ctx.fill();
        }
      } else {
        ctx.strokeStyle = this.showValue ? 'rgba(0,0,0,0.9)' : 'rgba(255,255,255,0.9)';
        ctx.beginPath(); ctx.moveTo(c.x - 10, c.y); ctx.lineTo(c.x - 3, c.y); ctx.moveTo(c.x + 3, c.y); ctx.lineTo(c.x + 10, c.y);
        ctx.moveTo(c.x, c.y - 10); ctx.lineTo(c.x, c.y - 3); ctx.moveTo(c.x, c.y + 3); ctx.lineTo(c.x, c.y + 10); ctx.stroke();
        ctx.beginPath(); ctx.arc(c.x, c.y, 1.5, 0, Math.PI * 2); ctx.fillStyle = this.showValue ? '#000' : '#fff'; ctx.fill();
      }
    }
    // placar detalhado (Tab)
    if (this.showScoreboard) this.drawScoreboard(game, human);
    // ajuda
    if (this.showHelp) {
      const lines = [
        'WASD mover · mouse mira · Shift correr (2x = arrancada)',
        'LMB chute (segurar = força; a mira trava e arrastar o mouse = efeito) · RMB passe',
        'Espaço: push ball (com bola) / drible (Ctrl+bola; 2x seguidas = roleta, com lag) / dash (Ctrl sem bola) / mergulho GK',
        'E tackle · Shift+E carrinho · Ctrl (ou C) postura · F arremesso do goleiro · botão do meio pede a bola',
        'Losango na bola = alvo: LMB/RMB/Espaço travam a ação (verde) e ela sai no toque; vermelho = outro tem prioridade',
        'Tab placar/ping (T ou clique = trocar de time) · Q troca jogador (solo) · R reinicia · H ajuda',
        'Enter: tela cheia (bloqueia Ctrl+W e outros atalhos do navegador)',
      ];
      ctx.font = '12px sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(w - 470, 10, 460, 16 * lines.length + 12);
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      lines.forEach((l, i) => ctx.fillText(l, w - 18, 16 + i * 16));
    }
  }
}
