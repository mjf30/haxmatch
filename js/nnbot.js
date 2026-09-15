'use strict';
// Bot controlado pela rede neural: observação (Features) -> MLP -> input do jogo.
// Saídas (12): mover x,y · mira x,y · chute · passe · sprint · postura · especial · tackle · arremesso · pedir bola
const NNBot = (() => {
  const OUT = 12;
  const SIZES_DEFAULT = [Features.SIZE, 128, 128, OUT];
  const sig = (v) => 1 / (1 + Math.exp(-v));

  // policy = { sizes, w }
  function think(p, g, dt, policy) {
    const inp = emptyInput();
    if (!policy) return inp;
    const dir = p.team === 0 ? 1 : -1;
    const x = Features.build(p, g, new Float32Array(Features.SIZE));
    const y = NN.forward(policy.sizes, policy.w, x);
    // movimento (referencial do time -> mundo)
    let mx = Math.tanh(y[0]), my = Math.tanh(y[1]);
    const ml = Math.hypot(mx, my);
    if (ml > 1) { mx /= ml; my /= ml; }
    if (ml < 0.15) { mx = 0; my = 0; }             // zona morta
    inp.mx = mx * dir; inp.my = my;
    // mira: direção relativa ao jogador, alcance de 600 px
    let ax = Math.tanh(y[2]), ay = Math.tanh(y[3]);
    const al = Math.hypot(ax, ay);
    if (al < 0.05) { ax = 1; ay = 0; }
    inp.aim = { x: p.pos.x + ax * dir * 600, y: p.pos.y + ay * 600 };
    inp.shoot = sig(y[4]) > 0.5;
    inp.pass = sig(y[5]) > 0.5;
    inp.sprint = sig(y[6]) > 0.5;
    inp.stance = sig(y[7]) > 0.5;
    inp.special = sig(y[8]) > 0.5;
    inp.tackle = sig(y[9]) > 0.5;
    inp.throwBall = sig(y[10]) > 0.5;
    inp.call = sig(y[11]) > 0.5;
    return inp;
  }

  function fromExport(obj) {
    if (!obj || !obj.w) return null;
    return { sizes: obj.sizes, w: Float32Array.from(obj.w) };
  }

  return { think, fromExport, OUT, SIZES_DEFAULT };
})();
