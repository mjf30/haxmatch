'use strict';
// Bot híbrido: a rede neural escolhe a decisão tática (uma das AI.MACROS) a
// partir da observação; a execução (movimento, mira, botões) é a do script.
// O goleiro também decide pela rede (decisões gk_*); a execução é do script.
const MacroBot = (() => {
  const SIZES_DEFAULT = [Features.SIZE, 64, AI.MACROS.length];
  const SET = (names) => new Set(names.map((n) => AI.MACROS.indexOf(n)));
  const CARRIER = SET(['shoot', 'shootq', 'pass', 'passback', 'longpass', 'through', 'switch', 'dribble', 'carryspace', 'hold']);
  const OFFBALL = SET(['chase', 'defend', 'cover', 'cutlane', 'openfwd', 'openwide', 'overlap', 'runbox', 'runspace', 'openback', 'openbest', 'guardgoal', 'home']);
  const GK = SET(['gk_angle', 'gk_press', 'gk_rush', 'gk_line', 'gk_up']);
  const KICK = SET(['shoot', 'shootq', 'pass', 'passback', 'longpass', 'through', 'switch']);
  // macros válidas na situação (igual a train_gpu/ppo.py macro_mask)
  function allowed(p, g, i) {
    const ball = g.ball.owner === p;
    if (ball) return CARRIER.has(i);
    const reach = p.reach !== null && p.reach !== undefined;
    if (p.isKeeper) return GK.has(i) || (reach && KICK.has(i) && AI.MACROS[i] !== 'shoot');
    return OFFBALL.has(i) || (reach && KICK.has(i));
  }

  // policy = { sizes, w }; devolve o input do jogo
  function think(p, g, dt, policy) {
    if (!policy) return AI.think(p, g, dt);
    const skip = policy.skip || 1;   // treinada com frame-skip: decide a cada `skip` ticks e mantém a macro
    return AI.think(p, g, dt, (pp, gg) => {
      if (skip > 1 && pp._mac && pp._mac.n < skip) { pp._mac.n++; return pp._mac.macro; }
      const x = Features.build(pp, gg, new Float32Array(Features.SIZE));
      const y = NN.forward(policy.sizes, policy.w, x);
      let best = -1;
      for (let i = 0; i < y.length; i++) { if (policy.mask && !allowed(pp, gg, i)) continue; if (best < 0 || y[i] > y[best]) best = i; }
      const macro = AI.MACROS[best];
      if (skip > 1) pp._mac = { macro, n: 1 };
      return macro;
    });
  }

  function fromExport(obj) {
    if (!obj || !obj.w || obj.kind !== 'macro') return null;
    return { sizes: obj.sizes, skip: obj.skip || 1, mask: !!obj.mask, w: Float32Array.from(obj.w) };
  }

  return { think, fromExport, SIZES_DEFAULT };
})();
