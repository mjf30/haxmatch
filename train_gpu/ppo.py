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
from sim_torch import TorchSim, CFG, DT, M_PLAY
import features_torch as FT

HEADS = [('move', 9), ('aim', 16), ('curve', 3), ('shoot', 2), ('pas', 2), ('sprint', 2), ('stance', 2), ('special', 2), ('tackle', 2)]
NOUT = sum(n for _, n in HEADS)   # 40
HID = 256


class Policy(nn.Module):
    """MLP tanh 135->256->256->40 (política, exportável) + cabeça de valor separada"""
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(FT.SIZE, HID); self.l2 = nn.Linear(HID, HID); self.out = nn.Linear(HID, NOUT)
        self.v1 = nn.Linear(FT.SIZE, HID); self.v2 = nn.Linear(HID, HID); self.vout = nn.Linear(HID, 1)
        nn.init.orthogonal_(self.out.weight, 0.01); nn.init.zeros_(self.out.bias)

    def logits(self, x):
        h = torch.tanh(self.l1(x)); h = torch.tanh(self.l2(h))
        return self.out(h)

    def value(self, x):
        h = torch.tanh(self.v1(x)); h = torch.tanh(self.v2(h))
        return self.vout(h).squeeze(-1)

    def dists(self, x):
        lg = self.logits(x)
        out, k = [], 0
        for _, n in HEADS:
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


def acts_to_dict(a):
    return {name: a[..., i] for i, (name, _) in enumerate(HEADS)}


def export_js(pol, path, info):
    """exporta a política (sem a cabeça de valor) no formato lido por js/rawbot.js"""
    ws = []
    for lin in (pol.l1, pol.l2, pol.out):
        ws.append(lin.weight.detach().cpu().reshape(-1)); ws.append(lin.bias.detach().cpu())
    w = torch.cat(ws)
    obj = {'kind': 'raw2', 'sizes': [FT.SIZE, HID, HID, NOUT], 'heads': [n for _, n in HEADS], 'obs': FT.SIZE, 'info': info,
           'w': [round(float(v), 5) for v in w.tolist()]}
    with open(path, 'w', encoding='utf-8') as f:
        f.write("'use strict';\n// Pesos PPO (controle total) exportados por train_gpu/ppo.py. %s\nconst NN_RAW_WEIGHTS = %s;\n" % (info, json.dumps(obj)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', type=int, default=1024); ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--team', type=int, default=4); ap.add_argument('--seconds', type=float, default=90)
    ap.add_argument('--rollout', type=int, default=64); ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--minibatch', type=int, default=32768); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.995); ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--ent', type=float, default=0.005); ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--league_every', type=int, default=50); ap.add_argument('--league_frac', type=float, default=0.3)
    ap.add_argument('--resume', type=str, default=None); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--export_every', type=int, default=25)
    ap.add_argument('--league_init', type=str, default=None)   # checkpoints congelados que ficam na liga o treino inteiro (benchmark), separados por vírgula
    ap.add_argument('--approach', type=float, default=0.02)   # shaping denso: aproximar-se da bola solta (currículo inicial)
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--sanity2', action='store_true')
    ap.add_argument('--ckpt', type=str, default=None)   # caminho do checkpoint (testes de sanidade usam outro)   # teste: recompensa densa trivial (todos se aproximam da bola) para validar o PPO
    args = ap.parse_args()
    dev = torch.device('cuda')
    torch.manual_seed(args.seed)
    B, T = args.envs, args.team
    sim = TorchSim(B, T, device='cuda', seconds=args.seconds, seed=args.seed)
    P = sim.P
    pol = Policy().to(dev)
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr, eps=1e-5)
    it0 = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        pol.load_state_dict(ck['pol'])
        if 'opt' in ck: opt.load_state_dict(ck['opt'])
        it0 = ck.get('it', 0)
        print('retomando de', args.resume, 'iteração', it0)
    league = []                                   # políticas congeladas (state_dicts)
    # quais partidas usam oponente da liga no time 1 (essas amostras do time 1 não treinam)
    leagueEnv = torch.zeros(B, dtype=torch.bool, device=dev)
    leagueIdx = torch.zeros(B, dtype=torch.long, device=dev)
    oppPols = []
    nFixed = 0
    if args.league_init:
        for path in args.league_init.split(','):
            ckl = torch.load(path, map_location=dev)
            snap = Policy().to(dev); snap.load_state_dict(ckl['pol']); snap.eval()
            for p_ in snap.parameters(): p_.requires_grad_(False)
            league.append(snap.state_dict()); oppPols.append(snap); nFixed += 1
            print('liga: adversário fixo', path, 'iteração', ckl.get('it'))
    team1 = (sim.team == 1).view(1, P).expand(B, P)
    ckpt = args.ckpt or os.path.join(os.path.dirname(__file__), 'ckpt_sanity.pt' if (args.sanity or args.sanity2) else 'ckpt.pt')
    outjs = os.path.join(os.path.dirname(__file__), '..', 'js', 'nn_raw_weights.js')
    # estatísticas
    stat = dict(goals=0.0, matches=0.0, shots=0.0, control=0.0, passes=0.0, passOk=0.0, steps=0)
    prevScore = sim.score.clone()
    passTick = torch.full((B,), -1e9, device=dev); passer = torch.zeros(B, dtype=torch.long, device=dev); tick = 0
    prevDist = (sim.pos - sim.bpos.view(B, 1, 2)).norm(dim=-1)   # para o shaping de aproximação
    possStreak = torch.zeros(B, device=dev)                    # ticks seguidos com a bola no mesmo time
    lastKickTeam = torch.full((B,), -1, dtype=torch.long, device=dev); lastKickTick = torch.full((B,), -1e9, device=dev)
    W2 = CFG['FIELD_W'] / 2
    obs = FT.build(sim)
    t0 = time.time()
    for it in range(it0 + 1, args.iters + 1):
        # ---- rollout ----
        R = args.rollout
        obs_buf = torch.zeros(R, B, P, FT.SIZE, device=dev)
        act_buf = torch.zeros(R, B, P, len(HEADS), dtype=torch.long, device=dev)
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
            inp = sim.act_from_discrete(acts_to_dict(a))
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
                rew += 0.04 * onT.float()
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
            rew += 0.002 * poss
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
            rew += 0.004 * spread * fieldm
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
            rew += 0.02 * (share - 1.0 / P) * fieldm
            # partida terminou (tempo) -> episódio acaba; reinicia essas partidas
            done = (sim.state == 3)
            if done.any():
                idx = done.nonzero().squeeze(1)
                stat['matches'] += float(idx.numel())
                sim.time[idx] = sim.seconds; sim.score[idx] = 0; prevScore[idx] = 0
                sim.stamina[idx] = CFG['STAMINA_MAX']; sim.effortBar[idx] = 1; sim.exhausted[idx] = False
                sim.kickoffTeam[idx] = (torch.rand(idx.numel(), device=dev) < 0.5).long()
                sim.kickoff(idx)
                # sorteia oponentes da liga para essas partidas
                if league:
                    use = torch.rand(idx.numel(), device=dev) < args.league_frac
                    leagueEnv[idx] = use
                    leagueIdx[idx] = torch.randint(0, len(league), (idx.numel(),), device=dev)
            for k in ('shot', 'control', 'pass'):
                if k in ev: stat['shots' if k == 'shot' else ('control' if k == 'control' else 'passes')] += float(ev[k].sum())
            stat['steps'] += 1
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
        # amostras treináveis: exclui o time 1 das partidas com oponente da liga
        trainMask = ~(leagueEnv.view(1, B, 1) & team1.view(1, B, P)).expand(R, B, P)
        idxs = trainMask.reshape(-1).nonzero().squeeze(1)
        O = obs_buf.reshape(-1, FT.SIZE)[idxs]; A = act_buf.reshape(-1, len(HEADS))[idxs]
        LP = logp_buf.reshape(-1)[idxs]; AD = adv.reshape(-1)[idxs]; RT = ret.reshape(-1)[idxs]
        AD = (AD - AD.mean()) / (AD.std() + 1e-8)
        N = O.shape[0]
        stats_pi, stats_v, stats_ent, nb, stats_gn = 0.0, 0.0, 0.0, 0, 0.0
        for ep in range(args.epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, args.minibatch):
                mb = perm[s:s + args.minibatch]
                logp, ent, v = pol.evaluate(O[mb], A[mb])
                ratio = torch.exp(logp - LP[mb])
                pl = -torch.min(ratio * AD[mb], ratio.clamp(1 - args.clip, 1 + args.clip) * AD[mb]).mean()
                vl = F.mse_loss(v, RT[mb])
                loss = pl + 0.5 * vl - args.ent * ent.mean()
                opt.zero_grad(); loss.backward(); gn = nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
                stats_gn = stats_gn + float(gn) if 'stats_gn' in dir() else float(gn)
                stats_pi += float(pl.detach()); stats_v += float(vl.detach()); stats_ent += float(ent.mean().detach()); nb += 1
        # ---- log ----
        if it % 5 == 0 or it == it0 + 1:
            m = max(1.0, stat['steps'] * B / (sim.seconds * 60.0))   # partidas-equivalentes jogadas nesta janela
            gsteps = max(1, stat['steps']) * B
            print(f"it {it} · {(time.time() - t0) / 60:.1f} min · partidas {int(stat['matches'])} · gols/partida {stat['goals'] / m:.2f} · "
                  f"chutes/partida {stat['shots'] / m:.1f} · domínios/partida {stat['control'] / m:.1f} · passes/partida {stat['passes'] / m:.1f} (completos {stat['passOk'] / m:.1f}) · "
                  f"pi {stats_pi / nb:.3f} v {stats_v / nb:.3f} ent {stats_ent / nb:.2f} gn {stats_gn / nb:.3f} p(move=1) {float(pol.dists(obs[:64].reshape(-1, FT.SIZE))[0].probs[:, 1].mean()):.3f} · liga {len(league)}", flush=True)
            stat = dict(goals=0.0, matches=0.0, shots=0.0, control=0.0, passes=0.0, passOk=0.0, steps=0)
        if it % args.league_every == 0:
            snap = Policy().to(dev); snap.load_state_dict(pol.state_dict()); snap.eval()
            for p_ in snap.parameters(): p_.requires_grad_(False)
            league.append(snap.state_dict()); oppPols.append(snap)
            if len(oppPols) > 6 + nFixed:   # os fixos nunca saem; o mais antigo dos demais sai
                oppPols.pop(nFixed); league.pop(nFixed)
                leagueIdx = torch.where(leagueIdx > nFixed, leagueIdx - 1, leagueIdx)
        if it % args.export_every == 0:
            torch.save({'pol': pol.state_dict(), 'opt': opt.state_dict(), 'it': it}, ckpt)
            export_js(pol, outjs, f'iteração {it}, {T}v{T}, {args.seconds}s/partida')
    torch.save({'pol': pol.state_dict(), 'opt': opt.state_dict(), 'it': args.iters}, ckpt)
    export_js(pol, outjs, f'iteração {args.iters}')


if __name__ == '__main__':
    main()
