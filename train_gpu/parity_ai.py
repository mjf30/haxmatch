"""Paridade entre js/ai.js e train_gpu/ai_torch.py num estado real de partida:
o JS joga alguns segundos com os bots script e exporta o estado; os dois lados então
calculam, para cada jogador e cada macro, a saída da execução (com estado de IA zerado e
sorteio fixo 0.99) e a decisão tática do script. Uso: python train_gpu/parity_ai.py"""
import json
import os
import subprocess
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from sim_torch import TorchSim, CFG, DT
import ai_torch as AT

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JS = r"""
const {loadSim}=require(%(bundle)s);const sim=loadSim();const {Game,AI,CFG,emptyInput}=sim;
const g=new Game({teamSize:%(team)d,seed:%(seed)d});
for(let i=0;i<60*%(secs)d;i++){for(const p of g.players)g.setInput(p.id,AI.think(p,g,CFG.DT));g.step(CFG.DT);}
g.rng=()=>0.99;
const st={players:g.players.map(p=>({pos:p.pos,vel:p.vel,facing:p.facing,moveDir:p.moveDir,stamina:p.stamina,exhausted:p.exhausted,effortBar:p.effortBar,effortT:p.effortT,
  isKeeper:p.isKeeper,held:p.held,stance:p.stance,cd:p.cd,action:p.action?p.action.type:null,recover:p.recover,fallen:p.fallen,getup:p.getup,dribbleLag:p.dribbleLag,
  charge:p.charge?{kind:p.charge.kind,t:p.charge.t}:null,queued:p.queued?{kind:p.queued.kind,t:p.queued.t,age:p.queued.age||0}:null,reach:!!p.reach})),
  ball:{pos:g.ball.pos,vel:g.ball.vel,owner:g.ball.owner?g.ball.owner.id:-1,lock:g.ball.lock?g.ball.lock.id:-1},score:g.score,time:g.time};
const fresh=()=>({t:0,mode:'none'});
const rows=[];
for(const p of g.players){
  const row={choose:null,exec:[]};
  p.ai=fresh(); const c0=AI.context(p,g); row.choose=AI.MACROS.indexOf(AI.chooseMacro(p,g,c0));
  for(const macro of AI.MACROS){
    p.ai=fresh(); p.ai.t+=CFG.DT; const c=AI.context(p,g); if(!c.hasBall)p.ai.mode='none';
    const inp=emptyInput();
    const out=p.isKeeper?AI.keeperExecute(p,g,inp,CFG.DT,c,macro):AI.execute(p,g,CFG.DT,c,macro,inp);
    row.exec.push({mx:out.mx,my:out.my,ax:out.aim.x,ay:out.aim.y,shoot:!!out.shoot,pass:!!out.pass,sprint:!!out.sprint,stance:!!out.stance,special:!!out.special,tackle:!!out.tackle,throw:!!out.throwBall,
      mode:p.ai.mode==='shoot'?1:(p.ai.mode==='pass'?2:0),chargeT:p.ai.chargeT||0,holdN:p.ai.holdN===undefined?0:p.ai.holdN});
  }
  rows.push(row);
}
st.rows=rows;
console.log(JSON.stringify(st));
"""


def load_state(st, team):
    sim = TorchSim(B=1, team_size=team, device='cuda', seconds=360, seed=1)
    ST = {'none': 0, 'drib': 1, 'def': 2}
    ACT = {None: 0, 'tackle': 1, 'slide': 2, 'dash': 3, 'dribble': 4, 'gkdive': 5}
    for i, p in enumerate(st['players']):
        sim.pos[0, i] = torch.tensor([p['pos']['x'], p['pos']['y']]); sim.vel[0, i] = torch.tensor([p['vel']['x'], p['vel']['y']])
        sim.facing[0, i] = torch.tensor([p['facing']['x'], p['facing']['y']]); sim.moveDir[0, i] = torch.tensor([p['moveDir']['x'], p['moveDir']['y']])
        sim.stamina[0, i] = p['stamina']; sim.exhausted[0, i] = p['exhausted']; sim.effortBar[0, i] = p['effortBar']; sim.effortT[0, i] = p['effortT']
        sim.isKeeper[0, i] = p['isKeeper']; sim.held[0, i] = p['held']; sim.stance[0, i] = ST[p['stance']]
        for k in ('tackle', 'slide', 'dribble', 'dash', 'dive', 'grab', 'gloves', 'through'):
            if k in p['cd']: sim.cd[k][0, i] = p['cd'][k]
        sim.act[0, i] = ACT[p['action']]; sim.recover[0, i] = p['recover']; sim.fallen[0, i] = p['fallen']; sim.getup[0, i] = p['getup']
        sim.dribbleLag[0, i] = p['dribbleLag']
        sim.chKind[0, i] = 0 if not p['charge'] else (1 if p['charge']['kind'] == 'shot' else 2); sim.chT[0, i] = p['charge']['t'] if p['charge'] else 0
        q = p['queued']
        sim.qKind[0, i] = 0 if not q else {'shot': 1, 'pass': 2, 'push': 3}[q['kind']]; sim.qT[0, i] = q['t'] if q else 0; sim.qAge[0, i] = q['age'] if q else 0
        sim.reach[0, i] = p['reach']
    b = st['ball']
    sim.bpos[0] = torch.tensor([b['pos']['x'], b['pos']['y']]); sim.bvel[0] = torch.tensor([b['vel']['x'], b['vel']['y']])
    sim.owner[0] = b['owner']; sim.lock[0] = b['lock']
    sim.score[0] = torch.tensor(st['score']); sim.time[0] = st['time']; sim.state[0] = 1
    return sim


def run(seed, secs, team):
    bundle = json.dumps(os.path.join(ROOT, 'train', 'bundle.js'))
    r = subprocess.run(['node', '-e', JS % {'bundle': bundle, 'seed': seed, 'secs': secs, 'team': team}], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)
    st = json.loads(r.stdout.strip().splitlines()[-1])
    sim = load_state(st, team)
    P = sim.P
    bad = 0; total = 0; badChoose = 0
    # decisão do script
    ai = AT.ScriptAI(sim); ai.rng = lambda: torch.full((1, P), 0.99, device='cuda')
    C = AT.Ctx(sim)
    ch = ai.choose_macro(C)[0].tolist()
    for i in range(P):
        if ch[i] != st['rows'][i]['choose']:
            badChoose += 1
            print(f'  decisão jogador {i}: js {AT.MACROS[st["rows"][i]["choose"]]} torch {AT.MACROS[ch[i]]}')
    # execução por macro
    for k, name in enumerate(AT.MACROS):
        ai = AT.ScriptAI(sim); ai.rng = lambda: torch.full((1, P), 0.99, device='cuda')
        C = AT.Ctx(sim)
        macro = torch.full((1, P), k, dtype=torch.long, device='cuda')
        out = ai.decide(C, macroFn=lambda C: macro)
        for i in range(P):
            ref = st['rows'][i]['exec'][k]
            got = dict(mx=float(out['mx'][0, i]), my=float(out['my'][0, i]), ax=float(out['aim'][0, i, 0]), ay=float(out['aim'][0, i, 1]),
                       shoot=bool(out['shoot'][0, i]), pass_=bool(out['pas'][0, i]), sprint=bool(out['sprint'][0, i]), stance=bool(out['stance'][0, i]),
                       special=bool(out['special'][0, i]), tackle=bool(out['tackle'][0, i]), throw=bool(out['throw'][0, i]),
                       mode=int(ai.mode[0, i]), chargeT=float(ai.chargeT[0, i]), holdN=int(ai.holdN[0, i]))
            ref['pass_'] = ref.pop('pass')
            diffs = []
            for key, gv in got.items():
                rv = ref[key]
                if isinstance(rv, bool):
                    ok = rv == gv
                else:
                    ok = abs(float(rv) - float(gv)) <= (0.5 if key in ('ax', 'ay') else 1e-3 * max(1.0, abs(float(rv))) + 1e-3)
                if not ok: diffs.append(f'{key}: js {rv} torch {gv}')
            total += 1
            if diffs:
                bad += 1
                if bad <= 25: print(f'  {name:<10} jogador {i} ({"gk" if st["players"][i]["isKeeper"] else "linha"}): ' + ' · '.join(diffs))
    print(f'seed {seed} t={secs}s {team}v{team}: execução {total - bad}/{total} iguais · decisão {P - badChoose}/{P} iguais')
    return bad + badChoose


if __name__ == '__main__':
    tot = 0
    for seed, secs, team in ((5, 7, 4), (9, 21, 4), (3, 40, 3), (11, 33, 5), (7, 55, 4)):
        tot += run(seed, secs, team)
    print('TOTAL divergências', tot)
