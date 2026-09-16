"""PPO com controle total do boneco, self-play em liga, na simulação vetorizada (GPU).
Uso: python train_gpu/ppo.py [--envs 1024] [--iters 2000] [--team 4] [--seconds 90] [--resume ckpt.pt]
Salva train_gpu/ckpt.pt e exporta js/nn_raw_weights.js a cada avaliação."""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from sim_torch import TorchSim, CFG, DT, M_PLAY, MOVE_DIRS, AIM_SECTORS
import features_torch as FT
import ai_torch as AT

HEADS = [('move', 9), ('aim', 16), ('curve', 3), ('shoot', 2), ('pas', 2), ('sprint', 2), ('stance', 2), ('special', 2), ('tackle', 2)]
NOUT = sum(n for _, n in HEADS)   # 40
HID = 256


MACRO_HEADS = [('macro', len(AT.MACROS))]
# índices na observação (features_torch): goleiro, bola no pé, bola na mão, alvo de primeira (reach)
OBS_KEEPER, OBS_HASBALL, OBS_HELD, OBS_REACH = 17, 18, 19, FT.SELF + 10
OFFBALL_SET = [AT.M[k] for k in ('chase', 'defend', 'cover', 'cutlane', 'openfwd', 'openwide', 'overlap', 'runbox', 'runspace', 'openback', 'openbest', 'guardgoal', 'home')]
KICK_NOSHOOT = [m for m in AT.KICK_SET if m != AT.M['shoot']]


def macro_mask(obs):
    """macros válidas por situação (as outras seriam convertidas pela execução, ex. 'runspace' com a bola = conduzir):
    com a bola: as 10 do portador; sem a bola: as 13 de linha (goleiro: as 5 gk_*), mais as de chute/passe de primeira quando há alvo."""
    n = len(AT.MACROS)
    keeper = obs[..., OBS_KEEPER] > 0.5
    ball = (obs[..., OBS_HASBALL] > 0.5) | (obs[..., OBS_HELD] > 0.5)
    reach = obs[..., OBS_REACH] > 0.5
    def setmask(idxs):
        m = torch.zeros(n, dtype=torch.bool, device=obs.device); m[idxs] = True; return m
    car, off, gk, kick, kickNS = setmask(AT.CARRIER_SET), setmask(OFFBALL_SET), setmask(AT.GK_SET), setmask(AT.KICK_SET), setmask(KICK_NOSHOOT)
    base = torch.where(keeper.unsqueeze(-1), gk, off)
    base = base | (reach.unsqueeze(-1) & torch.where(keeper.unsqueeze(-1), kickNS, kick))
    return torch.where(ball.unsqueeze(-1), car, base)


class Policy(nn.Module):
    """política exportável + cabeça de valor separada (207->256->256->1).
    raw: MLP tanh 207->256->256->40 (cabeças discretas do input cru)
    macro: MLP tanh 207->hid->28 (uma macro por jogador; a execução é a do script em ai_torch)"""
    def __init__(self, kind='raw', hid=None, depth=None):
        super().__init__()
        self.kind = kind
        self.depth = depth or (2 if kind == 'raw' else 1)   # camadas escondidas da política
        self.heads = HEADS if kind == 'raw' else MACRO_HEADS
        self.nout = sum(n for _, n in self.heads)
        self.hid = hid or (HID if kind == 'raw' else 64)
        self.l1 = nn.Linear(FT.SIZE, self.hid)
        self.l2 = nn.Linear(self.hid, self.hid) if self.depth >= 2 else None
        self.out = nn.Linear(self.hid, self.nout)
        self.v1 = nn.Linear(FT.SIZE, HID); self.v2 = nn.Linear(HID, HID); self.vout = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.out.weight, 0.01); nn.init.zeros_(self.out.bias)

    def logits(self, x):
        h = torch.tanh(self.l1(x))
        if self.l2 is not None: h = torch.tanh(self.l2(h))
        return self.out(h)

    def value(self, x):
        h = torch.tanh(self.v1(x)); h = torch.tanh(self.v2(h))
        return self.vout(h).squeeze(-1)

    def dists(self, x):
        lg = self.logits(x)
        if self.kind == 'macro':
            lg = torch.where(macro_mask(x), lg, torch.full_like(lg, -1e9))
        out, k = [], 0
        for _, n in self.heads:
            out.append(torch.distributions.Categorical(logits=lg[..., k:k + n])); k += n
        return out

    def act(self, x, deterministic=False):
        ds = self.dists(x)
        acts = [(d.probs.argmax(-1) if deterministic else d.sample()) for d in ds]
        logp = sum(d.log_prob(a) for d, a in zip(ds, acts))
        return torch.stack(acts, -1), logp

    def evaluate(self, x, acts):
        ds = self.dists(x)
        logp = sum(d.log_prob(acts[..., i]) for i, d in enumerate(ds))
        ent = sum(d.entropy() for d in ds)
        return logp, ent, self.value(x)

    def load_js_weights(self, w):
        """pesos planos no layout do js/nn.js (por camada: W[out][in] e bias) -> camadas da política"""
        w = torch.tensor(w, dtype=torch.float32)
        k = 0
        for lin in ((self.l1, self.l2, self.out) if self.l2 is not None else (self.l1, self.out)):
            o, i_ = lin.weight.shape
            lin.weight.data.copy_(w[k:k + o * i_].view(o, i_)); k += o * i_
            lin.bias.data.copy_(w[k:k + o]); k += o
        assert k == w.numel(), (k, w.numel())


def acts_to_dict(a):
    return {name: a[..., i] for i, (name, _) in enumerate(HEADS)}


def discretize_input(sim, inp):
    """input contínuo do script -> índices das cabeças do controle total [B,P,9] (alvo do guia)"""
    B, P = sim.B, sim.P
    flip = sim.dir.view(1, P)
    mv = torch.stack([inp['mx'] * flip, inp['my']], -1)                                   # referencial do time
    md = MOVE_DIRS.to(sim.dev)
    move = ((mv.unsqueeze(2) - md.view(1, 1, -1, 2)) ** 2).sum(-1).argmin(-1)
    move = torch.where(mv.norm(dim=-1) < 0.3, torch.zeros_like(move), move)
    rel = inp['aim'] - sim.pos
    ang = torch.atan2(rel[..., 1], rel[..., 0] * flip)
    sector = torch.round(ang / (2 * math.pi / AIM_SECTORS)).long() % AIM_SECTORS
    curve = torch.ones_like(move)                                                          # sem efeito
    cols = [move, sector, curve] + [inp[k].long() for k in ('shoot', 'pas', 'sprint', 'stance', 'special', 'tackle')]
    return torch.stack(cols, -1)


SKIP = 1


def export_js(pol, path, info):
    """exporta a política (sem a cabeça de valor) no formato lido por js/rawbot.js"""
    ws = []
    lins = (pol.l1, pol.l2, pol.out) if pol.l2 is not None else (pol.l1, pol.out)
    for lin in lins:
        ws.append(lin.weight.detach().cpu().reshape(-1)); ws.append(lin.bias.detach().cpu())
    w = torch.cat(ws)
    if pol.kind == 'raw':
        obj = {'kind': 'raw2', 'sizes': [FT.SIZE, HID, HID, NOUT], 'heads': [n for _, n in HEADS], 'obs': FT.SIZE, 'info': info, 'skip': SKIP,
               'w': [round(float(v), 5) for v in w.tolist()]}
        head = "'use strict';\n// Pesos PPO (controle total) exportados por train_gpu/ppo.py. %s\nconst NN_RAW_WEIGHTS = %s;\n"
    else:
        obj = {'kind': 'macro', 'sizes': [FT.SIZE] + [pol.hid] * pol.depth + [pol.nout], 'obs': FT.SIZE, 'info': info, 'skip': SKIP, 'mask': True, 'reflex': False,
               'w': [round(float(v), 5) for v in w.tolist()]}
        head = "'use strict';\n// Pesos PPO híbrido (decisão tática; execução do script) exportados por train_gpu/ppo.py. %s\nconst NN_WEIGHTS = %s;\n"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(head % (info, json.dumps(obj)))


def load_js_json(path):
    """lê pesos de um .json (train/*.json) ou de um .js (const NN_WEIGHTS = {...};)"""
    txt = open(path, encoding='utf-8').read()
    return json.loads(txt[txt.index('{'):txt.rindex('}') + 1])


def main():
    global SKIP
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', type=int, default=1024); ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--team', type=str, default='4'); ap.add_argument('--seconds', type=float, default=90)   # tamanhos de time, ex.: 3,4,5 (um sim por tamanho, mesma rede)
    ap.add_argument('--rollout', type=int, default=64); ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--minibatch', type=int, default=32768); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.995); ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--ent', type=float, default=0.005); ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--league_every', type=int, default=50); ap.add_argument('--league_frac', type=float, default=0.3)
    ap.add_argument('--resume', type=str, default=None); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--export_every', type=int, default=25)
    ap.add_argument('--skip', type=int, default=1)             # frame-skip: a política decide a cada N ticks e a ação é mantida
    ap.add_argument('--shape_decay', type=int, default=0)      # iterações para o shaping denso decair linearmente até --shape_floor (0 = sem decaimento)
    ap.add_argument('--shape_floor', type=float, default=0.3)
    ap.add_argument('--policy', type=str, default='raw')        # raw (controle total) | macro (híbrido)
    ap.add_argument('--hid', type=int, default=0)               # camada escondida da política macro (0 = 64)
    ap.add_argument('--depth', type=int, default=0)             # camadas escondidas da política macro (0 = 1)
    ap.add_argument('--init_js', type=str, default=None)        # pesos iniciais no layout JS (clonagem / ES), só para macro
    ap.add_argument('--out', type=str, default=None)            # arquivo .js de exportação
    ap.add_argument('--script_frac', type=float, default=0.0)   # fração das partidas com o script (torch) no time 1
    ap.add_argument('--attack', type=float, default=0.0)        # fração de partidas no cenário de finalização (episódios de 8 s)
    ap.add_argument('--build', type=float, default=0.0)         # fração no cenário de construção (episódios de 15 s)
    ap.add_argument('--curriculum', type=int, default=0)        # iterações em que as frações decaem linearmente a zero (0 = fixas)
    ap.add_argument('--guide', type=float, default=0.0)         # híbrido guiado: peso da entropia cruzada com a macro do script (decai até --guide_floor em --guide_decay iterações)
    ap.add_argument('--guide_decay', type=int, default=0)
    ap.add_argument('--guide_floor', type=float, default=0.0)
    ap.add_argument('--fixed_share', type=float, default=0.5)   # fração das partidas de liga que usam os adversários fixos (o resto: snapshots)
    ap.add_argument('--league_init', type=str, default=None)   # checkpoints congelados que ficam na liga o treino inteiro (benchmark), separados por vírgula
    ap.add_argument('--approach', type=float, default=0.02)   # shaping denso: aproximar-se da bola solta (currículo inicial)
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--sanity2', action='store_true')
    ap.add_argument('--ckpt', type=str, default=None)   # caminho do checkpoint (testes de sanidade usam outro)   # teste: recompensa densa trivial (todos se aproximam da bola) para validar o PPO
    args = ap.parse_args()
    dev = torch.device('cuda')
    torch.manual_seed(args.seed)
    teams = [int(x) for x in str(args.team).split(',')]
    Bw = max(1, args.envs // len(teams))
    class World: pass
    worlds = []
    for wi, T in enumerate(teams):   # um sim por tamanho de time; a mesma rede joga em todos
        W = World(); W.T = T; W.B = Bw
        W.sim = TorchSim(Bw, T, device='cuda', seconds=args.seconds, seed=args.seed + wi)
        W.P = W.sim.P
        W.ai = AT.ScriptAI(W.sim)   # execução das macros (híbrido) e script adversário
        worlds.append(W)
    B = args.envs; T = args.team
    pol = Policy(args.policy, args.hid or None, args.depth or None).to(dev)
    if args.init_js and args.policy == 'macro':
        pol.load_js_weights(load_js_json(args.init_js)['w']); print('política macro inicializada de', args.init_js)
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr, eps=1e-5)
    it0 = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        pol.load_state_dict(ck['pol'])
        if 'opt' in ck: opt.load_state_dict(ck['opt'])
        it0 = ck.get('it', 0)
        print('retomando de', args.resume, 'iteração', it0)
    league = []                                   # políticas congeladas (state_dicts)
    oppPols = []
    nFixed = 0
    if args.league_init:
        for path in args.league_init.split(','):
            ckl = torch.load(path, map_location=dev)
            snap = Policy(ckl.get('kind', args.policy), ckl.get('hid'), ckl.get('depth')).to(dev); snap.load_state_dict(ckl['pol']); snap.eval()
            for p_ in snap.parameters(): p_.requires_grad_(False)
            league.append(snap.state_dict()); oppPols.append(snap); nFixed += 1
            print('liga: adversário fixo', path, 'iteração', ckl.get('it'))
    SKIP = args.skip
    ckpt = args.ckpt or os.path.join(os.path.dirname(__file__), 'ckpt_sanity.pt' if (args.sanity or args.sanity2) else 'ckpt.pt')
    outjs = args.out or os.path.join(os.path.dirname(__file__), '..', 'js', 'nn_raw_weights.js') if args.policy == 'raw' else (args.out or os.path.join(os.path.dirname(__file__), 'macro_ppo_weights.js'))
    # estatísticas
    stat = dict(goals=0.0, matches=0.0, shots=0.0, control=0.0, passes=0.0, passOk=0.0, steps=0)
    W2 = CFG['FIELD_W'] / 2
    cur = {'scale': 1.0}
    def assign_scen(W, idx):
        """sorteia o cenário (0 partida, 1 finalização, 2 construção) das partidas idx do mundo W e monta a cena"""
        n = idx.numel()
        if n == 0: return
        fA, fB = args.attack * cur['scale'], args.build * cur['scale']
        rr = torch.rand(n, device=dev)
        k = torch.where(rr < fA, 1, torch.where(rr < fA + fB, 2, 0))
        W.scen[idx] = k; W.scenT[idx] = torch.where(k == 1, 8.0, torch.where(k == 2, 15.0, 0.0))
        s_ = k > 0
        if s_.any():
            att = (torch.rand(int(s_.sum()), device=dev) < 0.5).long()
            W.sim.setup_scenario(idx[s_], k[s_], att); W.sim.time[idx[s_]] = W.sim.seconds
    for W in worlds:   # estado por mundo
        Bw_, Pw_ = W.B, W.P
        W.team1 = (W.sim.team == 1).view(1, Pw_).expand(Bw_, Pw_)
        W.leagueEnv = torch.zeros(Bw_, dtype=torch.bool, device=dev); W.leagueIdx = torch.zeros(Bw_, dtype=torch.long, device=dev)
        W.scriptEnv = torch.zeros(Bw_, dtype=torch.bool, device=dev)
        W.prevScore = W.sim.score.clone()
        W.passTick = torch.full((Bw_,), -1e9, device=dev); W.passer = torch.zeros(Bw_, dtype=torch.long, device=dev); W.tick = 0
        W.prevDist = (W.sim.pos - W.sim.bpos.view(Bw_, 1, 2)).norm(dim=-1)
        W.possStreak = torch.zeros(Bw_, device=dev)
        W.lastKickTeam = torch.full((Bw_,), -1, dtype=torch.long, device=dev); W.lastKickTick = torch.full((Bw_,), -1e9, device=dev)
        W.scen = torch.zeros(Bw_, dtype=torch.long, device=dev); W.scenT = torch.zeros(Bw_, device=dev)
        assign_scen(W, torch.arange(Bw_, device=dev))
        W.obs = FT.build(W.sim)
    t0 = time.time()
    R = args.rollout
    shape = 1.0

    def rollout(W):
        """um rollout de R decisões no mundo W; devolve as amostras treináveis achatadas"""
        sim, ai, B, P, team1 = W.sim, W.ai, W.B, W.P, W.team1
        prevScore, passTick, passer, tick = W.prevScore, W.passTick, W.passer, W.tick
        prevDist, possStreak, lastKickTeam, lastKickTick = W.prevDist, W.possStreak, W.lastKickTeam, W.lastKickTick
        scen, scenT, scriptEnv, leagueEnv, leagueIdx, obs = W.scen, W.scenT, W.scriptEnv, W.leagueEnv, W.leagueIdx, W.obs
        assign = lambda idx: assign_scen(W, idx)
        obs_buf = torch.zeros(R, B, P, FT.SIZE, device=dev)
        act_buf = torch.zeros(R, B, P, len(pol.heads), dtype=torch.long, device=dev)
        guide_buf = torch.zeros(R, B, P, len(pol.heads), dtype=torch.long, device=dev)   # ação que o script escolheria (guia): macro, ou input discretizado
        logp_buf = torch.zeros(R, B, P, device=dev); val_buf = torch.zeros(R, B, P, device=dev)
        rew_buf = torch.zeros(R, B, P, device=dev); done_buf = torch.zeros(R, B, device=dev)
        for r in range(R):
            with torch.no_grad():
                a, logp = pol.act(obs)
                v = pol.value(obs)
                # oponentes da liga (time 1 das partidas marcadas)
                if oppPols and leagueEnv.any():
                    for li, op in enumerate(oppPols):
                        m = leagueEnv & (leagueIdx == li)
                        if m.any():
                            ao, _ = op.act(obs[m])
                            sel = team1[m]
                            a[m] = torch.where(sel.unsqueeze(-1), ao, a[m])
            if pol.kind == 'raw': inpPol = sim.act_from_discrete(acts_to_dict(a))
            rewT = torch.zeros(B, P, device=dev); doneT = torch.zeros(B, dtype=torch.bool, device=dev)
            for _k in range(args.skip):
                with torch.no_grad():
                    if pol.kind == 'raw':
                        inp = inpPol
                        if scriptEnv.any() or (args.guide > 0 and _k == 0):   # script (torch): adversário e/ou guia
                            sInp = ai.think(sim)
                            if _k == 0 and args.guide > 0: guide_buf[r] = discretize_input(sim, sInp)
                            if scriptEnv.any():
                                sm = scriptEnv.view(B, 1) & team1
                                inp = {k: torch.where(sm.unsqueeze(-1) if v.dim() == 3 else sm, sInp[k], v) for k, v in inp.items()}
                    else:   # macro: decisão mantida durante o skip; execução do script a cada tick
                        C = AT.Ctx(sim)
                        mac = a[..., 0]
                        if scriptEnv.any() or (args.guide > 0 and _k == 0):
                            sm = ai.choose_macro(C)
                            if _k == 0: guide_buf[r, ..., 0] = sm
                            mac = torch.where(scriptEnv.view(B, 1) & team1, sm, mac)
                        inp = ai.decide(C, macroFn=lambda C_, m_=mac: m_)
                # progresso da bola antes do passo (referencial de cada time)
                bx0 = sim.bpos[:, 0].clone()
                playing = (sim.state == M_PLAY)
                ev = sim.step(inp)
                # ---- recompensas ----
                rew = torch.zeros(B, P, device=dev)
                dsc = (sim.score - prevScore).float()          # [B,2] gols neste passo
                stat['goals'] += float(dsc.sum())
                prevScore = sim.score.clone()
                gteam = dsc[:, 0].view(B, 1) * (sim.team == 0).view(1, P).float() + dsc[:, 1].view(B, 1) * (sim.team == 1).view(1, P).float()
                gopp = dsc[:, 1].view(B, 1) * (sim.team == 0).view(1, P).float() + dsc[:, 0].view(B, 1) * (sim.team == 1).view(1, P).float()
                rew += 1.0 * gteam - 1.0 * gopp
                goalTerm = 1.0 * gteam - 1.0 * gopp
                # bola avançando no sentido do ataque (só durante o jogo, sem o reset do kickoff)
                dbx = (sim.bpos[:, 0] - bx0).view(B, 1) * sim.dir.view(1, P) / CFG['FIELD_W']
                dbx = torch.where(playing.view(B, 1), dbx.clamp(-0.05, 0.05), torch.zeros_like(dbx))
                rew += 0.3 * dbx
                if 'shot' in ev:
                    rew += 0.0 * ev['shot'].float()
                    # chute na direção do gol adversário (a trajetória reta cruza a boca do gol)
                    bv = sim.bvel; bp = sim.bpos
                    vx = bv[:, 0].view(B, 1) * sim.dir.view(1, P)
                    tcross = (W2 - bp[:, 0].view(B, 1) * sim.dir.view(1, P)) / vx.clamp(min=1e-3)
                    ycross = bp[:, 1].view(B, 1) + bv[:, 1].view(B, 1) * tcross
                    onT = ev['shot'] & (vx > 100) & (tcross > 0) & (ycross.abs() < CFG['GOAL_W'] / 2)
                    goalP = torch.stack([sim.dir.view(1, P).expand(B, P) * W2, torch.zeros(B, P, device=dev)], -1)
                    dShot = (sim.pos - goalP).norm(dim=-1)
                    nearF = (1 - (dShot - 500) / 500).clamp(0, 1)   # chute no alvo vale cheio até 500 px, nada além de 1000
                    rew += 0.04 * onT.float() * nearF
                    # defesa: chute adversário na direção do gol foi permitido -> penalidade coletiva do time que defende
                    onTb = onT.any(dim=1)
                    shooterTeam = torch.where(onTb, sim.team[onT.float().argmax(dim=1)], torch.full_like(sim.lastTeam, -1))
                    rew -= 0.04 * ((shooterTeam.view(B, 1) >= 0) & (sim.team.view(1, P) != shooterTeam.view(B, 1))).float()
                    # registra o time do último chute/passe (para interceptações)
                    kb = ev['shot'].any(dim=1)
                    lastKickTeam = torch.where(kb, sim.team[ev['shot'].float().argmax(dim=1)], lastKickTeam)
                    lastKickTick = torch.where(kb, torch.full_like(lastKickTick, float(tick)), lastKickTick)
                # aproximação da bola: só o companheiro mais próximo de cada time é recompensado por se aproximar
                dist = (sim.pos - sim.bpos.view(B, 1, 2)).norm(dim=-1)
                closest = torch.zeros(B, P, dtype=torch.bool, device=dev)
                for tteam in (0, 1):
                    tm = (sim.team == tteam).view(1, P).expand(B, P)
                    dd = torch.where(tm, dist, torch.full_like(dist, 1e9))
                    closest[torch.arange(B, device=dev), dd.argmin(dim=1)] = True
                approach = ((prevDist - dist) / 100).clamp(-0.05, 0.05)
                rew += torch.where(closest & playing.view(B, 1) & (sim.owner < 0).view(B, 1), args.approach * approach, torch.zeros_like(approach))
                prevDist = dist
                if args.sanity: rew = ((prevDist - dist) / 100).clamp(-0.05, 0.05) * 0 + (-dist / 1000) * 0.01
                if args.sanity2: rew = (a[..., 0] == 1).float() * 0.1   # recompensa por escolher mover na direção 1
                if 'control' in ev:
                    rew += 0.01 * ev['control'].float()
                    # interceptação: dominar até 2 s após chute/passe do adversário
                    recentK = (tick - lastKickTick) < 2.0 / DT
                    inter = ev['control'] & recentK.view(B, 1) & (lastKickTeam.view(B, 1) >= 0) & (sim.team.view(1, P) != lastKickTeam.view(B, 1))
                    rew += 0.08 * inter.float()
                if 'first' in ev: rew += 0.01 * ev['first'].float()
                if 'pass' in ev:
                    rew += 0.01 * ev['pass'].float()
                    kb = ev['pass'].any(dim=1)
                    lastKickTeam = torch.where(kb, sim.team[ev['pass'].float().argmax(dim=1)], lastKickTeam)
                    lastKickTick = torch.where(kb, torch.full_like(lastKickTick, float(tick)), lastKickTick)
                # passe completado: companheiro (outro jogador) domina até 2,5 s depois -> passador e receptor
                tick += 1
                if 'pass' in ev and ev['pass'].any():
                    pb = ev['pass'].any(dim=1)
                    passTick = torch.where(pb, torch.full_like(passTick, float(tick)), passTick)
                    passer = torch.where(pb, ev['pass'].float().argmax(dim=1), passer)
                if 'control' in ev or 'first' in ev:
                    ctrl = (ev.get('control', torch.zeros(B, P, dtype=torch.bool, device=dev)) | ev.get('first', torch.zeros(B, P, dtype=torch.bool, device=dev)))
                    recent = (tick - passTick) < 2.5 / DT
                    sameTeam = sim.team.view(1, P) == sim.team[passer].view(B, 1)
                    notSelf = torch.arange(P, device=dev).view(1, P) != passer.view(B, 1)
                    done_pass = ctrl & recent.view(B, 1) & sameTeam & notSelf
                    if done_pass.any():
                        # passe completado: 0,15 para receptor e passador; +0,05 se avançou a bola
                        fwd = ((sim.pos[..., 0] - sim.pos[torch.arange(B, device=dev), passer][:, 0].view(B, 1)) * sim.dir.view(1, P) > 150).float()
                        rew += (0.15 + 0.05 * fwd) * done_pass.float()
                        rew[torch.arange(B, device=dev)[done_pass.any(dim=1)], passer[done_pass.any(dim=1)]] += 0.15
                        stat['passOk'] += float(done_pass.sum())
                        passTick = torch.where(done_pass.any(dim=1), torch.full_like(passTick, -1e9), passTick)
                if 'steal' in ev: rew += 0.10 * ev['steal'].float()
                if 'save' in ev: rew += 0.10 * ev['save'].float()
                # posse: pequeno bônus por tick para o time com a bola (compartilhado)
                has = sim.owner >= 0
                oteam = torch.where(has, sim.team[sim.owner.clamp(min=0)], torch.full_like(sim.owner, -1))
                poss = (oteam.view(B, 1) == sim.team.view(1, P)).float()
                rew += 0.0 * poss   # (posse por tick removida: recompensava segurar a bola e nunca passar)
                # enrolar: mesmo time com a bola por mais de 4 s seguidos -> penalidade crescente
                possStreak = torch.where(has, possStreak + 1, torch.zeros_like(possStreak))
                stall = (possStreak > 4 / DT).float().view(B, 1) * poss
                rew -= 0.003 * stall
                # com a bola: espaçamento entre companheiros de linha (média das distâncias, cap 500) e aglomeração
                fieldm = (~sim.isKeeper).float()
                dpp = (sim.pos.view(B, P, 1, 2) - sim.pos.view(B, 1, P, 2)).norm(dim=-1).clamp(max=500)
                sameT = (sim.team.view(1, P, 1) == sim.team.view(1, 1, P)) & ~torch.eye(P, dtype=torch.bool, device=dev).view(1, P, P)
                wgt = sameT.float() * fieldm.view(B, 1, P)
                spread = (dpp * wgt).sum(-1) / wgt.sum(-1).clamp(min=1) / 500
                # espaçamento SEMPRE (com e sem a bola): média das distâncias entre companheiros de linha
                rew += 0.004 * spread * fieldm * poss   # espaçamento só com a posse (sem bola, compactar é o certo)
                # companheiros de linha colados (< 100 px): penalidade por par
                close = ((dpp < 100) & sameT).float() * fieldm.view(B, 1, P) * fieldm.view(B, P, 1)
                rew -= 0.003 * close.sum(-1)
                # três ou mais do mesmo time em cima da bola, em qualquer fase
                near = ((sim.pos - sim.bpos.view(B, 1, 2)).norm(dim=-1) < 150).float() * fieldm
                crowd = torch.zeros(B, P, device=dev)
                for tteam in (0, 1):
                    tm = (sim.team == tteam).view(1, P).float()
                    crowd += ((near * tm).sum(dim=1, keepdim=True) >= 3).float() * tm
                rew -= 0.006 * crowd
                # goleiro: fica na área (penalidade fora dela com a bola no próprio campo; bônus dentro);
                # jogadores de linha não entram na própria área quando há goleiro
                gk = sim.isKeeper.float()
                inBoxP = sim.in_own_box().float()
                ballOwnHalf = torch.where(sim.team.view(1, P) == 0, sim.bpos[:, 0].view(B, 1) < 0, sim.bpos[:, 0].view(B, 1) > 0).float()
                oppHasBall = ((oteam.view(B, 1) >= 0) & (oteam.view(B, 1) != sim.team.view(1, P))).float()
                rew -= 0.004 * gk * (1 - inBoxP) * ballOwnHalf * oppHasBall   # só quando o adversário ataca
                rew += 0.001 * gk * inBoxP
                teamHasGk = torch.zeros(B, P, device=dev)
                for tteam in (0, 1):
                    tm = (sim.team == tteam).view(1, P).float()
                    teamHasGk += ((gk * tm).sum(dim=1, keepdim=True) > 0).float() * tm
                rew -= 0.003 * (1 - gk) * inBoxP * teamHasGk
                # controle de campo: cada jogador de linha ganha pela própria fatia (Voronoi) acima da média
                cnt, _, _ = FT.pitch_control(sim)
                share = cnt / (FT.GX * FT.GY)
                rew += 0.01 * (share - 1.0 / P) * fieldm * poss   # fatia de campo só com a posse; metade do peso (a macro 'runspace' maximiza isso direto)
                rew = goalTerm + shape * (rew - goalTerm)   # shaping denso com decaimento; gols nunca decaem
                # partida terminou (tempo) -> episódio acaba; reinicia essas partidas
                done = (sim.state == 3)
                if done.any():
                    idx = done.nonzero().squeeze(1)
                    stat['matches'] += float(idx.numel())
                    sim.time[idx] = sim.seconds; sim.score[idx] = 0; prevScore[idx] = 0
                    sim.stamina[idx] = CFG['STAMINA_MAX']; sim.effortBar[idx] = 1; sim.exhausted[idx] = False
                    sim.kickoffTeam[idx] = (torch.rand(idx.numel(), device=dev) < 0.5).long()
                    sim.kickoff(idx)
                    assign(idx)
                    # sorteia oponentes da liga para essas partidas
                    rr = torch.rand(idx.numel(), device=dev)
                    scriptEnv[idx] = rr < args.script_frac
                    if league:
                        leagueEnv[idx] = (rr >= args.script_frac) & (rr < args.script_frac + args.league_frac)
                        nl = len(league); nf = nFixed
                        useFixed = (torch.rand(idx.numel(), device=dev) < args.fixed_share) & (nf > 0) & (nl > nf)
                        pickF = torch.randint(0, max(1, nf), (idx.numel(),), device=dev)
                        pickS = nf + torch.randint(0, max(1, nl - nf), (idx.numel(),), device=dev) if nl > nf else pickF
                        leagueIdx[idx] = torch.where(useFixed, pickF, torch.where(torch.tensor(nl > nf, device=dev), pickS, pickF))
                # cenários: episódio acaba por tempo ou gol -> recomeça (novo sorteio de cenário); conta como fim de episódio
                scenT = torch.where(scen > 0, scenT - DT, scenT)
                epEnd = (scen > 0) & ((scenT <= 0) | (dsc.sum(-1) > 0))
                if epEnd.any():
                    idx = epEnd.nonzero().squeeze(1)
                    sim.time[idx] = sim.seconds; sim.score[idx] = 0; prevScore[idx] = 0
                    sim.stamina[idx] = CFG['STAMINA_MAX']; sim.effortBar[idx] = 1; sim.exhausted[idx] = False
                    sim.kickoffTeam[idx] = (torch.rand(idx.numel(), device=dev) < 0.5).long()
                    sim.kickoff(idx)
                    assign(idx)
                    done = done | epEnd
                for k in ('shot', 'control', 'pass'):
                    if k in ev: stat['shots' if k == 'shot' else ('control' if k == 'control' else 'passes')] += float(ev[k].sum())
                stat['steps'] += B / args.envs
                rewT += rew; doneT |= done
            rew = rewT; done = doneT
            obs_buf[r] = obs; act_buf[r] = a; logp_buf[r] = logp; val_buf[r] = v; rew_buf[r] = rew; done_buf[r] = done.float()
            obs = FT.build(sim)
        # ---- GAE ----
        with torch.no_grad():
            nextv = pol.value(obs)
            adv = torch.zeros_like(rew_buf); last = torch.zeros(B, P, device=dev)
            for r in reversed(range(R)):
                nd = (1 - done_buf[r]).view(B, 1)
                nv = nextv if r == R - 1 else val_buf[r + 1]
                delta = rew_buf[r] + args.gamma * nv * nd - val_buf[r]
                last = delta + args.gamma * args.lam * nd * last
                adv[r] = last
            ret = adv + val_buf
        W.prevScore, W.passTick, W.passer, W.tick = prevScore, passTick, passer, tick
        W.prevDist, W.possStreak, W.lastKickTeam, W.lastKickTick = prevDist, possStreak, lastKickTeam, lastKickTick
        W.scen, W.scenT, W.scriptEnv, W.leagueEnv, W.leagueIdx, W.obs = scen, scenT, scriptEnv, leagueEnv, leagueIdx, obs
        # amostras treináveis: exclui o time 1 das partidas com oponente da liga ou script
        trainMask = ~((leagueEnv | scriptEnv).view(1, B, 1) & team1.view(1, B, P)).expand(R, B, P)
        idxs = trainMask.reshape(-1).nonzero().squeeze(1)
        return (obs_buf.reshape(-1, FT.SIZE)[idxs], act_buf.reshape(-1, len(pol.heads))[idxs], logp_buf.reshape(-1)[idxs],
                adv.reshape(-1)[idxs], ret.reshape(-1)[idxs], guide_buf.reshape(-1, len(pol.heads))[idxs])

    for it in range(it0 + 1, args.iters + 1):
        shape = max(args.shape_floor, 1.0 - it / args.shape_decay) if args.shape_decay > 0 else 1.0
        cur['scale'] = max(0.0, 1.0 - it / args.curriculum) if args.curriculum > 0 else 1.0
        parts = [rollout(W) for W in worlds]
        O = torch.cat([q[0] for q in parts]); A = torch.cat([q[1] for q in parts]); LP = torch.cat([q[2] for q in parts])
        AD = torch.cat([q[3] for q in parts]); RT = torch.cat([q[4] for q in parts]); GD = torch.cat([q[5] for q in parts])
        beta = (max(args.guide_floor, args.guide * (1 - it / args.guide_decay)) if args.guide_decay > 0 else args.guide) if args.guide > 0 else 0.0
        AD = (AD - AD.mean()) / (AD.std() + 1e-8)
        N = O.shape[0]
        stats_pi, stats_v, stats_ent, nb, stats_gn, stats_gl = 0.0, 0.0, 0.0, 0, 0.0, 0.0
        for ep in range(args.epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, args.minibatch):
                mb = perm[s:s + args.minibatch]
                logp, ent, v = pol.evaluate(O[mb], A[mb])
                ratio = torch.exp(logp - LP[mb])
                pl = -torch.min(ratio * AD[mb], ratio.clamp(1 - args.clip, 1 + args.clip) * AD[mb]).mean()
                vl = F.mse_loss(v, RT[mb])
                loss = pl + 0.5 * vl - args.ent * ent.mean()
                if beta > 0:
                    nll = -sum(d.log_prob(GD[mb][..., i]) for i, d in enumerate(pol.dists(O[mb])))   # guia: segue a decisão do script (macro ou input discretizado)
                    wB = 1 + 3 * ((O[mb][:, OBS_HASBALL] > 0.5) | (O[mb][:, OBS_HELD] > 0.5)).float()   # decisões com a bola são raras e decisivas: peso 4x
                    if pol.kind == 'raw': wB = wB + 3 * (GD[mb][:, 8] > 0).float()   # desarmes são raros no professor: peso 4x quando o script desarma
                    gl = (nll * wB).sum() / wB.sum()
                    loss = loss + beta * gl; stats_gl = stats_gl + float(gl.detach()) if 'stats_gl' in dir() else float(gl.detach())
                opt.zero_grad(); loss.backward(); gn = nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
                stats_gn = stats_gn + float(gn) if 'stats_gn' in dir() else float(gn)
                stats_pi += float(pl.detach()); stats_v += float(vl.detach()); stats_ent += float(ent.mean().detach()); nb += 1
        # ---- log ----
        if it % 5 == 0 or it == it0 + 1:
            m = max(1.0, stat['steps'] * args.envs / (worlds[0].sim.seconds * 60.0))   # partidas-equivalentes jogadas nesta janela
            print(f"it {it} · {(time.time() - t0) / 60:.1f} min · partidas {int(stat['matches'])} · gols/partida {stat['goals'] / m:.2f} · "
                  f"chutes/partida {stat['shots'] / m:.1f} · domínios/partida {stat['control'] / m:.1f} · passes/partida {stat['passes'] / m:.1f} (completos {stat['passOk'] / m:.1f}) · "
                  f"pi {stats_pi / nb:.3f} v {stats_v / nb:.3f} ent {stats_ent / nb:.2f} gn {stats_gn / nb:.3f} p(a0=1) {float(pol.dists(worlds[0].obs[:64].reshape(-1, FT.SIZE))[0].probs[:, 1].mean()):.3f} · liga {len(league)}" + (f' · guia {stats_gl / nb:.2f} (beta {beta:.2f})' if beta > 0 else ''), flush=True)
            stat = dict(goals=0.0, matches=0.0, shots=0.0, control=0.0, passes=0.0, passOk=0.0, steps=0)
        if it % args.league_every == 0:
            snap = Policy(pol.kind, pol.hid, pol.depth).to(dev); snap.load_state_dict(pol.state_dict()); snap.eval()
            for p_ in snap.parameters(): p_.requires_grad_(False)
            league.append(snap.state_dict()); oppPols.append(snap)
            if len(oppPols) > 6 + nFixed:   # os fixos nunca saem; o mais antigo dos demais sai
                oppPols.pop(nFixed); league.pop(nFixed)
                for W in worlds: W.leagueIdx = torch.where(W.leagueIdx > nFixed, W.leagueIdx - 1, W.leagueIdx)
        stopFlag = ckpt + '.stop'   # encerramento suave: crie este arquivo e o treino salva e sai (sem matar o processo)
        if it % args.export_every == 0 or os.path.exists(stopFlag):
            torch.save({'pol': pol.state_dict(), 'opt': opt.state_dict(), 'it': it, 'kind': pol.kind, 'hid': pol.hid, 'depth': pol.depth}, ckpt)
            export_js(pol, outjs, f'iteração {it}, {T}v{T}, {args.seconds}s/partida')
        if os.path.exists(stopFlag):
            os.remove(stopFlag); print('parada suave solicitada: checkpoint salvo na iteração', it, flush=True); return
    torch.save({'pol': pol.state_dict(), 'opt': opt.state_dict(), 'it': args.iters, 'kind': pol.kind, 'hid': pol.hid, 'depth': pol.depth}, ckpt)
    export_js(pol, outjs, f'iteração {args.iters}')


if __name__ == '__main__':
    main()
