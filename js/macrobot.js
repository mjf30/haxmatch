'use strict';
// Bot híbrido: a rede neural escolhe a decisão tática (uma das AI.MACROS) a
// partir da observação; a execução (movimento, mira, botões) é a do script.
// O goleiro também decide pela rede (decisões gk_*); a execução é do script.
const MacroBot = (() => {
  const SIZES_DEFAULT = [Features.SIZE, 64, AI.MACROS.length];

  // policy = { sizes, w }; devolve o input do jogo
  function think(p, g, dt, policy) {
    if (!policy) return AI.think(p, g, dt);
    const skip = policy.skip || 1;   // treinada com frame-skip: decide a cada `skip` ticks e mantém a macro
    return AI.think(p, g, dt, (pp, gg) => {
      if (skip > 1 && pp._mac && pp._mac.n < skip) { pp._mac.n++; return pp._mac.macro; }
      const x = Features.build(pp, gg, new Float32Array(Features.SIZE));
      const y = NN.forward(policy.sizes, policy.w, x);
      let best = 0;
      for (let i = 1; i < y.length; i++) if (y[i] > y[best]) best = i;
      const macro = AI.MACROS[best];
      if (skip > 1) pp._mac = { macro, n: 1 };
      return macro;
    });
  }

  function fromExport(obj) {
    if (!obj || !obj.w || obj.kind !== 'macro') return null;
    return { sizes: obj.sizes, skip: obj.skip || 1, w: Float32Array.from(obj.w) };
  }

  return { think, fromExport, SIZES_DEFAULT };
})();
