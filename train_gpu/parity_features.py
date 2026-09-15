"""Compara Features.build (JS) com features_torch.build (torch) num estado real:
o JS joga alguns segundos com os bots script, exporta o estado, o torch carrega."""
import json
import os
import subprocess
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from sim_torch import TorchSim, CFG
import features_torch as FT

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JS = r"""
const {loadSim}=require(%(bundle)s);const sim=loadSim();const {Game,AI,CFG,Features}=sim;
const g=new Game({teamSize:4,seed:%(seed)d});
for(let i=0;i<60*%(secs)d;i++){for(const p of g.players)g.setInput(p.id,AI.think(p,g,CFG.DT));g.step(CFG.DT);}
const st={players:g.players.map(p=>({pos:p.pos,vel:p.vel,facing:p.facing,stamina:p.stamina,exhausted:p.exhausted,effortBar:p.effortBar,effortT:p.effortT,
  isKeeper:p.isKeeper,held:p.held,stance:p.stance,cd:p.cd,action:p.action?p.action.type:null,recover:p.recover,fallen:p.fallen,getup:p.getup,dribbleLag:p.dribbleLag,
  charge:p.charge?{kind:p.charge.kind,t:p.charge.t}:null,queued:p.queued?{age:p.queued.age||0}:null,reach:!!p.reach})),
  ball:{pos:g.ball.pos,vel:g.ball.vel,owner:g.ball.owner?g.ball.owner.id:-1,lock:g.ball.lock?g.ball.lock.id:-1},score:g.score,time:g.time,
  feats:g.players.map(p=>Array.from(Features.build(p,g,new Float32Array(Features.SIZE))))};
console.log(JSON.stringify(st));
"""


def run(seed, secs):
    bundle = json.dumps(os.path.join(ROOT, 'train', 'bundle.js'))
    r = subprocess.run(['node', '-e', JS % {'bundle': bundle, 'seed': seed, 'secs': secs}], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)
    st = json.loads(r.stdout.strip().splitlines()[-1])
    sim = TorchSim(B=1, team_size=4, device='cuda', seconds=360, seed=1)
    ST = {'none': 0, 'drib': 1, 'def': 2}
    ACT = {None: 0, 'tackle': 1, 'slide': 2, 'dash': 3, 'dribble': 4, 'gkdive': 5}
    for i, p in enumerate(st['players']):
        sim.pos[0, i] = torch.tensor([p['pos']['x'], p['pos']['y']]); sim.vel[0, i] = torch.tensor([p['vel']['x'], p['vel']['y']])
        sim.facing[0, i] = torch.tensor([p['facing']['x'], p['facing']['y']])
        sim.stamina[0, i] = p['stamina']; sim.exhausted[0, i] = p['exhausted']; sim.effortBar[0, i] = p['effortBar']; sim.effortT[0, i] = p['effortT']
        sim.isKeeper[0, i] = p['isKeeper']; sim.held[0, i] = p['held']; sim.stance[0, i] = ST[p['stance']]
        for k in ('tackle', 'slide', 'dribble'): sim.cd[k][0, i] = p['cd'][k]
        sim.act[0, i] = ACT[p['action']]; sim.recover[0, i] = p['recover']; sim.fallen[0, i] = p['fallen']; sim.getup[0, i] = p['getup']
        sim.dribbleLag[0, i] = p['dribbleLag']
        sim.chKind[0, i] = 0 if not p['charge'] else (1 if p['charge']['kind'] == 'shot' else 2); sim.chT[0, i] = p['charge']['t'] if p['charge'] else 0
        sim.qKind[0, i] = 1 if p['queued'] else 0; sim.qAge[0, i] = p['queued']['age'] if p['queued'] else 0
        sim.reach[0, i] = p['reach']
    b = st['ball']
    sim.bpos[0] = torch.tensor([b['pos']['x'], b['pos']['y']]); sim.bvel[0] = torch.tensor([b['vel']['x'], b['vel']['y']])
    sim.owner[0] = b['owner']; sim.lock[0] = b['lock']
    sim.score[0] = torch.tensor(st['score']); sim.time[0] = st['time']
    x = FT.build(sim)[0].cpu()
    ref = torch.tensor(st['feats'])
    diff = (x - ref).abs()
    worst = diff.max(dim=1).values
    print(f'seed {seed} t={secs}s: diferença máx {float(diff.max()):.4f} · média {float(diff.mean()):.5f} · por jogador máx {[round(float(v), 3) for v in worst]}')
    if float(diff.max()) > 1e-3:
        bad = (diff > 1e-3).nonzero()
        for pi, fi in bad[:12].tolist():
            print(f'  jogador {pi} feature {fi}: js {float(ref[pi, fi]):.4f} torch {float(x[pi, fi]):.4f}')


if __name__ == '__main__':
    for seed, secs in ((5, 7), (9, 21), (3, 40)):
        run(seed, secs)
