"""Paridade entre a simulação JS (js/game.js) e a vetorizada (sim_torch.py) em
cenários controlados. Uso: python train_gpu/parity.py"""
import json
import subprocess
import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(__file__))
from sim_torch import TorchSim, CFG, DT

ROOT = os.path.join(os.path.dirname(__file__), '..')

JS = r"""
const fs=require('fs'),path=require('path'),vm=require('vm');
const code=['config.js','vec.js','input.js','game.js'].map(f=>fs.readFileSync(path.join(%(root)s,'js',f),'utf8').replace(/^'use strict';/,'')).join('\n')+'\nthis.__e={Game,CFG,V,emptyInput};';
const sb={console,Math,Infinity,Date};vm.createContext(sb);vm.runInContext(code,sb);const {Game,CFG,V,emptyInput}=sb.__e;
function setup(withBall){ const g=new Game({teamSize:3,seed:1}); g.state='play'; g.stateT=0;
  for(const p of g.players){ p.active=false; p.pos={x:0,y:-2000}; }
  const p=g.players[2]; p.active=true; p.pos={x:-300,y:0}; p.facing={x:1,y:0}; p.moveDir={x:1,y:0};
  if(withBall){ g.ball.owner=p; g.ball.pos={x:-300+p.r+8+3,y:0}; } else { g.ball.owner=null; g.ball.pos={x:0,y:-2000}; }
  g.ball.vel={x:0,y:0}; g.ball.lock=null; return [g,p]; }
const out={};
// 1) corrida 6 s sem bola
{ const [g,p]=setup(false); for(let i=0;i<360;i++){ const inp=emptyInput(); inp.mx=1; inp.sprint=true; inp.aim={x:5000,y:0}; g.setInput(2,inp); g.step(CFG.DT);} out.run=p.pos.x+300; out.staminaAfterRun=p.stamina; }
// 2) andar 6 s
{ const [g,p]=setup(false); for(let i=0;i<360;i++){ const inp=emptyInput(); inp.mx=1; inp.aim={x:5000,y:0}; g.setInput(2,inp); g.step(CFG.DT);} out.walk=p.pos.x+300; }
// 3) correr com bola 6 s
{ const [g,p]=setup(true); for(let i=0;i<360;i++){ const inp=emptyInput(); inp.mx=1; inp.sprint=true; inp.aim={x:5000,y:0}; g.setInput(2,inp); g.step(CFG.DT);} out.runBall=p.pos.x+300; out.ballOwned=g.ball.owner===p; }
// 4) chute carga cheia: velocidade inicial e distância até parar (sem paredes: mira para +x com espaço)
{ const [g,p]=setup(true); p.pos={x:-900,y:0}; g.ball.pos={x:-900+p.r+11,y:0}; let v0=0, t=0;
  for(let i=0;i<600;i++){ const inp=emptyInput(); inp.aim={x:5000,y:0}; inp.shoot=(i<100); g.setInput(2,inp); g.step(CFG.DT); for(const e of g.events) if(e.type==='shot'){ v0=V.len(g.ball.vel); t=i; } }
  out.shotV0=v0; out.shotTick=t; out.shotX=g.ball.pos.x; }
// 5) chute toque rápido
{ const [g,p]=setup(true); p.pos={x:-900,y:0}; g.ball.pos={x:-900+p.r+11,y:0}; let v0=0;
  for(let i=0;i<200;i++){ const inp=emptyInput(); inp.aim={x:5000,y:0}; inp.shoot=(i<1); g.setInput(2,inp); g.step(CFG.DT); for(const e of g.events) if(e.type==='shot') v0=V.len(g.ball.vel); }
  out.tapV0=v0; }
// 6) passe toque rápido (sem companheiro)
{ const [g,p]=setup(true); let v0=0; for(let i=0;i<200;i++){ const inp=emptyInput(); inp.aim={x:5000,y:0}; inp.pass=(i<1); g.setInput(2,inp); g.step(CFG.DT); for(const e of g.events) if(e.type==='pass') v0=V.len(g.ball.vel);} out.passV0=v0; }
// 7) push correndo: velocidade da bola e ticks até recuperar
{ const [g,p]=setup(true); let v0=0, regain=-1; for(let i=0;i<300;i++){ const inp=emptyInput(); inp.mx=1; inp.sprint=true; inp.aim={x:5000,y:0}; inp.special=(i===30); g.setInput(2,inp); g.step(CFG.DT); for(const e of g.events) if(e.type==='push') v0=V.len(g.ball.vel); if(v0>0&&regain<0&&g.ball.owner===p) regain=i-30; } out.pushV0=v0; out.pushRegain=regain; }
// 8) bola vindo a 500 px/s: domina? (limiar 600)
{ const [g,p]=setup(false); g.ball.pos={x:0,y:0}; g.ball.vel={x:-500,y:0}; let owned=false; for(let i=0;i<120;i++){ const inp=emptyInput(); inp.aim={x:5000,y:0}; g.setInput(2,inp); g.step(CFG.DT); if(g.ball.owner===p) owned=true;} out.control500=owned; }
{ const [g,p]=setup(false); g.ball.pos={x:0,y:0}; g.ball.vel={x:-900,y:0}; let owned=false; for(let i=0;i<120;i++){ const inp=emptyInput(); inp.aim={x:5000,y:0}; g.setInput(2,inp); g.step(CFG.DT); if(g.ball.owner===p) owned=true;} out.control900=owned; }
// 9) de primeira com o botão segurado: bola lenta chega, domina e continua carregando; sai na carga cheia
{ const [g,p]=setup(false); g.ball.pos={x:0,y:0}; g.ball.vel={x:-400,y:0}; let ctl=-1, st=-1, v0=0;
  for(let i=0;i<300;i++){ const inp=emptyInput(); inp.aim={x:5000,y:0}; inp.shoot=true; g.setInput(2,inp); g.step(CFG.DT);
    for(const e of g.events){ if(e.type==='control'&&ctl<0) ctl=i; if(e.type==='shot'){ st=i; v0=V.len(g.ball.vel); } } }
  out.holdCtlTick=ctl; out.holdShotTick=st; out.holdShotV0=v0; }
// 10) de primeira soltando antes do contato: sai no toque
{ const [g,p]=setup(false); g.ball.pos={x:0,y:0}; g.ball.vel={x:-400,y:0}; let ft=-1, v0=0;
  for(let i=0;i<300;i++){ const inp=emptyInput(); inp.aim={x:5000,y:0}; inp.shoot=(i<48); g.setInput(2,inp); g.step(CFG.DT);
    for(const e of g.events){ if(e.type==='first-touch'&&ft<0){ ft=i; v0=V.len(g.ball.vel); } } }
  out.firstTick=ft; out.firstV0=v0; }
console.log(JSON.stringify(out));
"""


def js_ref():
    root = json.dumps(os.path.abspath(ROOT))
    r = subprocess.run(['node', '-e', JS % {'root': root}], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)
    return json.loads(r.stdout.strip().splitlines()[-1])


def make(with_ball, x=-300.0):
    sim = TorchSim(B=1, team_size=3, device='cuda', seconds=600, seed=1)
    sim.state[:] = 1; sim.stateT[:] = 0
    sim.pos[:] = torch.tensor([0.0, -2000.0])
    p = 2
    sim.pos[0, p] = torch.tensor([x, 0.0]); sim.facing[0, p] = torch.tensor([1.0, 0.0]); sim.moveDir[0, p] = torch.tensor([1.0, 0.0])
    if with_ball:
        sim.owner[0] = p; sim.bpos[0] = torch.tensor([x + CFG['PLAYER_R'] + CFG['BALL_R'] + 3, 0.0])
    else:
        sim.owner[0] = -1; sim.bpos[0] = torch.tensor([0.0, -2000.0])
    sim.bvel[:] = 0; sim.lock[:] = -1
    sim._last_inp = sim._empty_input()
    return sim, p


def inp(sim, **kw):
    i = sim._empty_input()
    i['aim'][:] = torch.tensor([5000.0, 0.0])
    for k, v in kw.items():
        i[k][0, 2] = v
    sim._last_inp = i
    return i


def torch_ref():
    out = {}
    sim, p = make(False)
    for _ in range(360): sim.step(inp(sim, mx=1.0, sprint=True))
    out['run'] = float(sim.pos[0, p, 0] + 300); out['staminaAfterRun'] = float(sim.stamina[0, p])
    sim, p = make(False)
    for _ in range(360): sim.step(inp(sim, mx=1.0))
    out['walk'] = float(sim.pos[0, p, 0] + 300)
    sim, p = make(True)
    for _ in range(360): sim.step(inp(sim, mx=1.0, sprint=True))
    out['runBall'] = float(sim.pos[0, p, 0] + 300); out['ballOwned'] = bool(sim.owner[0] == p)
    sim, p = make(True, x=-900.0)
    v0, t = 0.0, 0
    for i in range(600):
        ev = sim.step(inp(sim, shoot=(i < 100)))
        if 'shot' in ev and ev['shot'].any(): v0 = float(sim.bvel[0].norm()); t = i
    out['shotV0'] = v0; out['shotTick'] = t; out['shotX'] = float(sim.bpos[0, 0])
    sim, p = make(True, x=-900.0)
    v0 = 0.0
    for i in range(200):
        ev = sim.step(inp(sim, shoot=(i < 1)))
        if 'shot' in ev and ev['shot'].any(): v0 = float(sim.bvel[0].norm())
    out['tapV0'] = v0
    sim, p = make(True)
    v0 = 0.0
    for i in range(200):
        ev = sim.step(inp(sim, pas=(i < 1)))
        if 'pass' in ev and ev['pass'].any(): v0 = float(sim.bvel[0].norm())
    out['passV0'] = v0
    sim, p = make(True)
    v0, regain = 0.0, -1
    for i in range(300):
        ev = sim.step(inp(sim, mx=1.0, sprint=True, special=(i == 30)))
        if 'push' in ev and ev['push'].any(): v0 = float(sim.bvel[0].norm())
        if v0 > 0 and regain < 0 and int(sim.owner[0]) == p: regain = i - 30
    out['pushV0'] = v0; out['pushRegain'] = regain
    for spd, key in ((500.0, 'control500'), (900.0, 'control900')):
        sim, p = make(False)
        sim.bpos[0] = torch.tensor([0.0, 0.0]); sim.bvel[0] = torch.tensor([-spd, 0.0])
        owned = False
        for i in range(120):
            sim.step(inp(sim)); owned = owned or int(sim.owner[0]) == p
        out[key] = owned
    sim, p = make(False)
    sim.bpos[0] = torch.tensor([0.0, 0.0]); sim.bvel[0] = torch.tensor([-400.0, 0.0])
    ctl, st, v0 = -1, -1, 0.0
    for i in range(300):
        ev = sim.step(inp(sim, shoot=True))
        if ctl < 0 and 'control' in ev and ev['control'].any(): ctl = i
        if 'shot' in ev and ev['shot'].any(): st = i; v0 = float(sim.bvel[0].norm())
    out['holdCtlTick'] = ctl; out['holdShotTick'] = st; out['holdShotV0'] = v0
    sim, p = make(False)
    sim.bpos[0] = torch.tensor([0.0, 0.0]); sim.bvel[0] = torch.tensor([-400.0, 0.0])
    ft, v0 = -1, 0.0
    for i in range(300):
        ev = sim.step(inp(sim, shoot=(i < 48)))
        if ft < 0 and 'first' in ev and ev['first'].any(): ft = i; v0 = float(sim.bvel[0].norm())
    out['firstTick'] = ft; out['firstV0'] = v0
    return out


if __name__ == '__main__':
    a, b = js_ref(), torch_ref()
    print(f"{'cenário':<18}{'JS':>12}{'torch':>12}")
    for k in a:
        va, vb = a[k], b.get(k)
        ok = (va == vb) if isinstance(va, bool) else (abs(float(va) - float(vb)) <= max(2.0, 0.03 * abs(float(va))))
        print(f"{k:<18}{str(round(va, 1) if not isinstance(va, bool) else va):>12}{str(round(vb, 1) if not isinstance(vb, bool) else vb):>12}  {'ok' if ok else 'DIFERENTE'}")
