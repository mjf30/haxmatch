'use strict';
// Estado de input por jogador. Humanos e bots produzem o mesmo formato;
// a simulação detecta "apertou"/"soltou" comparando com o tick anterior.
function emptyInput() {
  return {
    mx: 0, my: 0,                 // eixo de movimento (-1..1)
    aim: { x: 0, y: 0 },          // ponto de mira no mundo
    shoot: false,                 // LMB
    pass: false,                  // RMB
    sprint: false,                // Shift
    stance: false,                // Ctrl (drible com bola / defensiva sem bola)
    special: false,               // Espaço (push, drible, dash, mergulho)
    tackle: false,                // E (tackle / carrinho se correndo)
    throwBall: false,             // F (arremesso do goleiro)
    call: false,                  // botão do meio (pedir bola)
    drag: { x: 0, y: 0 },         // movimento do mouse (px) com a mira travada: vira efeito no chute
  };
}

class HumanInput {
  constructor(canvas, hooks) {
    this.keys = new Set();
    this.buttons = new Set();
    this.canvas = canvas;
    this.mouse = { x: 0, y: 0 };   // mira virtual (tela)
    this.raw = { x: 0, y: 0 };     // posição real do mouse (sem pointer lock)
    this.frozen = false;           // mira travada (segurando o chute): arrasto vira efeito
    this.drag = { x: 0, y: 0 };
    this.hooks = hooks || {};
    this.enabled = false;        // só captura teclado/mouse na tela de jogo
    const gameKeys = new Set([
      'KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
      'Space', 'ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight',
      'KeyE', 'KeyF', 'KeyC', 'Tab', 'KeyR', 'KeyH', 'KeyP', 'KeyQ', 'KeyT', 'Enter', 'Escape',
    ]);
    const typing = (e) => {
      const t = e.target;
      return t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
    };
    window.addEventListener('keydown', (e) => {
      if (!this.enabled || typing(e)) return;      // menu ou campo de texto: o navegador cuida
      if (gameKeys.has(e.code)) e.preventDefault();
      if (e.repeat) return;
      this.keys.add(e.code);
      if (this.hooks.onKey) this.hooks.onKey(e.code);
    });
    window.addEventListener('keyup', (e) => { if (this.enabled) this.keys.delete(e.code); });
    window.addEventListener('blur', () => { this.keys.clear(); this.buttons.clear(); });
    canvas.addEventListener('mousemove', (e) => {
      const r = canvas.getBoundingClientRect();
      const locked = document.pointerLockElement === canvas;
      let dx, dy;
      if (locked) { dx = e.movementX; dy = e.movementY; }
      else { const nx = e.clientX - r.left, ny = e.clientY - r.top; dx = nx - this.raw.x; dy = ny - this.raw.y; this.raw.x = nx; this.raw.y = ny; }
      if (this.frozen) { this.drag.x += dx; this.drag.y += dy; return; }
      if (locked) { this.mouse.x = Math.max(0, Math.min(r.width, this.mouse.x + dx)); this.mouse.y = Math.max(0, Math.min(r.height, this.mouse.y + dy)); }
      else { this.mouse.x = this.raw.x; this.mouse.y = this.raw.y; }
    });
    canvas.addEventListener('mousedown', (e) => { if (!this.enabled) return; e.preventDefault(); this.buttons.add(e.button); });
    window.addEventListener('mouseup', (e) => this.buttons.delete(e.button));
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  down(code) { return this.keys.has(code); }
  setEnabled(on) {
    this.enabled = on; this.keys.clear(); this.buttons.clear(); this.frozen = false;
    if (!on && document.pointerLockElement === this.canvas) document.exitPointerLock();
  }
  // trava/destrava a mira (segurando o chute). Ao destravar sem pointer lock, a mira volta para o mouse real.
  setFrozen(on) {
    if (on === this.frozen) return;
    this.frozen = on;
    if (!on) { this.drag.x = 0; this.drag.y = 0; if (document.pointerLockElement !== this.canvas) { this.mouse.x = this.raw.x; this.mouse.y = this.raw.y; } }
  }
  lockPointer() {
    if (document.pointerLockElement === this.canvas || !this.canvas.requestPointerLock) return;
    try { const p = this.canvas.requestPointerLock(); if (p && p.catch) p.catch(() => {}); } catch (e) { /* sem pointer lock */ }
  }

  sample(screenToWorld) {
    const i = emptyInput();
    let mx = 0, my = 0;
    if (this.down('KeyW') || this.down('ArrowUp')) my -= 1;
    if (this.down('KeyS') || this.down('ArrowDown')) my += 1;
    if (this.down('KeyA') || this.down('ArrowLeft')) mx -= 1;
    if (this.down('KeyD') || this.down('ArrowRight')) mx += 1;
    const l = Math.hypot(mx, my);
    if (l > 1) { mx /= l; my /= l; }
    i.mx = mx; i.my = my;
    i.aim = screenToWorld(this.mouse.x, this.mouse.y);
    i.shoot = this.buttons.has(0);
    i.pass = this.buttons.has(2);
    i.call = this.buttons.has(1);
    i.sprint = this.down('ShiftLeft') || this.down('ShiftRight');
    i.stance = this.down('ControlLeft') || this.down('ControlRight') || this.down('KeyC');
    i.special = this.down('Space');
    i.tackle = this.down('KeyE');
    i.throwBall = this.down('KeyF');
    i.drag = { x: this.drag.x, y: this.drag.y };
    this.drag.x = 0; this.drag.y = 0;
    return i;
  }
}
