'use strict';
// Worker: avalia candidatos (pesos) em partidas headless.
const { parentPort } = require('worker_threads');
const { loadSim, playMatch, fitnessOf } = require('./bundle');
const sim = loadSim();

parentPort.on('message', (msg) => {
  const { id, sizes, w, opps, seeds, scenarios, opts, kind } = msg;
  const policy = { sizes, w: new Float32Array(w), kind };
  let total = 0;
  const ms = [];
  for (let i = 0; i < seeds.length; i++) {
    const o = opps[i % opps.length];
    const oppPolicy = o === 'script' ? 'script' : { sizes, w: new Float32Array(o), kind };
    const m = playMatch(sim, policy, oppPolicy, Object.assign({}, opts, { seed: seeds[i], nnTeam: i % 2, scenario: scenarios ? scenarios[i % scenarios.length] : 'match' }));
    total += fitnessOf(m);
    ms.push(m);
  }
  parentPort.postMessage({ id, fitness: total / seeds.length, ms });
});
