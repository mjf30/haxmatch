'use strict';
// Bot de controle total treinado por PPO (train_gpu/ppo.py): a rede decide
// movimento, mira, efeito e botões diretamente. Ações discretas por cabeça
// (argmax), no referencial do time (ataque em +x), como em sim_torch.act_from_discrete.
const RawBot = (() => {
  const MOVE_DIRS = [[0, 0]];
  for (let i = 0; i < 8; i++) MOVE_DIRS.push([Math.cos(i * Math.PI / 4), Math.sin(i * Math.PI / 4)]);
  const AIM_SECTORS = 16;

  function argmax(arr, off, n) {
    let b = 0;
    for (let i = 1; i < n; i++) if (arr[off + i] > arr[off + b]) b = i;
    return b;
  }

  // policy = { sizes, heads, w, curve } — curve: efeito acumulado por jogador (estado local)
  function think(p, g, dt, policy) {
    const inp = emptyInput();
    if (!policy) return inp;
    const dir = p.team === 0 ? 1 : -1;
    const x = Features.build(p, g, new Float32Array(Features.SIZE));
    const y = NN.forward(policy.sizes, policy.w, x);
    const heads = policy.heads;
    const a = [];
    let off = 0;
    for (const n of heads) { a.push(argmax(y, off, n)); off += n; }
    const [move, aim, curve, shoot, pas, sprint, stance, special, tackle] = a;
    inp.mx = MOVE_DIRS[move][0] * dir; inp.my = MOVE_DIRS[move][1];
    const ang = aim * (2 * Math.PI / AIM_SECTORS);
    inp.aim = { x: p.pos.x + Math.cos(ang) * dir * 600, y: p.pos.y + Math.sin(ang) * 600 };
    // efeito: a simulação de treino soma curve*SPIN_MAX*0.08 por tick durante a carga;
    // aqui traduzimos para o arrasto do mouse equivalente (drag lateral em px)
    const c = curve - 1;
    if (c !== 0) {
      const d0 = (p.charge && p.charge.dir0) || (p.queued && p.queued.dir0) || { x: Math.cos(ang) * dir, y: Math.sin(ang) };
      const perp = { x: -d0.y, y: d0.x };
      const px = c * CFG.SPIN_MAX * 0.08 / CFG.SPIN_GAIN_PX;
      inp.drag = { x: perp.x * px, y: perp.y * px };
    }
    inp.shoot = !!shoot; inp.pass = !!pas; inp.sprint = !!sprint; inp.stance = !!stance; inp.special = !!special; inp.tackle = !!tackle;
    return inp;
  }

  function fromExport(obj) {
    if (!obj || !obj.w || obj.kind !== 'raw2') return null;
    return { sizes: obj.sizes, heads: obj.heads, w: Float32Array.from(obj.w) };
  }

  return { think, fromExport };
})();
