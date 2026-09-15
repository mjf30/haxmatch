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
  };
}

class HumanInput {
  constructor(canvas, hooks) {
    this.keys = new Set();
    this.buttons = new Set();
    this.mouse = { x: 0, y: 0 };
    this.hooks = hooks || {};
    const gameKeys = new Set([
      'KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
      'Space', 'ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight',
      'KeyE', 'KeyF', 'KeyC', 'Tab', 'KeyR', 'KeyH', 'KeyP', 'KeyQ', 'Enter', 'Escape',
    ]);
    const typing = (e) => {
      const t = e.target;
      return t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
    };
    window.addEventListener('keydown', (e) => {
      if (typing(e)) return;                       // digitando no lobby: deixa o navegador cuidar
      if (gameKeys.has(e.code)) e.preventDefault();
      if (e.repeat) return;
      this.keys.add(e.code);
      if (this.hooks.onKey) this.hooks.onKey(e.code);
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => { this.keys.clear(); this.buttons.clear(); });
    canvas.addEventListener('mousemove', (e) => {
      const r = canvas.getBoundingClientRect();
      this.mouse.x = e.clientX - r.left;
      this.mouse.y = e.clientY - r.top;
    });
    canvas.addEventListener('mousedown', (e) => { e.preventDefault(); this.buttons.add(e.button); });
    window.addEventListener('mouseup', (e) => this.buttons.delete(e.button));
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  down(code) { return this.keys.has(code); }

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
    return i;
  }
}
