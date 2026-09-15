'use strict';
// MLP mínimo em JS puro (sem dependências): pesos num Float32Array plano,
// ativação tanh nas camadas ocultas, saída linear. Usado no navegador e no treino.
const NN = (() => {
  function paramCount(sizes) {
    let n = 0;
    for (let i = 1; i < sizes.length; i++) n += sizes[i - 1] * sizes[i] + sizes[i];
    return n;
  }
  // inicialização (Xavier) com gerador determinístico
  function init(sizes, rng) {
    const w = new Float32Array(paramCount(sizes));
    let k = 0;
    for (let i = 1; i < sizes.length; i++) {
      const fanIn = sizes[i - 1], fanOut = sizes[i];
      const s = Math.sqrt(2 / (fanIn + fanOut));
      for (let j = 0; j < fanIn * fanOut; j++) w[k++] = (rng() * 2 - 1) * s * 1.7;
      for (let j = 0; j < fanOut; j++) w[k++] = 0;
    }
    return w;
  }
  function forward(sizes, w, x) {
    let a = x;
    let k = 0;
    for (let i = 1; i < sizes.length; i++) {
      const nIn = sizes[i - 1], nOut = sizes[i];
      const out = new Float32Array(nOut);
      for (let o = 0; o < nOut; o++) {
        let s = 0;
        const base = k + o * nIn;
        for (let j = 0; j < nIn; j++) s += w[base + j] * a[j];
        out[o] = s;
      }
      k += nIn * nOut;
      for (let o = 0; o < nOut; o++) out[o] += w[k + o];
      k += nOut;
      if (i < sizes.length - 1) for (let o = 0; o < nOut; o++) out[o] = Math.tanh(out[o]);
      a = out;
    }
    return a;
  }
  return { paramCount, init, forward };
})();
