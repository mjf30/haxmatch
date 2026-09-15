"""Porte 1:1 de js/ai.js para tensores (batched): decisão tática (chooseMacro) e execução
das macros (execute / keeperExecute) para todos os jogadores de todas as partidas de uma vez.

Uso:
    ai = ScriptAI(sim)                 # estado por jogador (modo, mira, carga, tempo de reação)
    inp = ai.think(sim, macroFn=None)  # dict de input para sim.step; macroFn(C) -> macro [B,P] (rede híbrida)

Toda a lógica segue a ordem e os limiares do JS; os sorteios (g.rng()) viram ai.rng(), que pode ser
trocada por uma constante nos testes de paridade (train_gpu/parity_ai.py).
"""
import math
import torch

from sim_torch import CFG, DT, ST_DEF, ST_DRIB, Q_SHOT, ACT_NONE
from features_torch import GX as FT_GX, GY as FT_GY

MACROS = ['shoot', 'shootq', 'pass', 'passback', 'longpass', 'through', 'switch', 'dribble', 'carryspace', 'hold',
          'chase', 'defend', 'cover', 'cutlane', 'openfwd', 'openwide', 'overlap', 'runbox', 'runspace', 'openback', 'openbest', 'guardgoal', 'home',
          'gk_angle', 'gk_press', 'gk_rush', 'gk_line', 'gk_up']
M = {name: i for i, name in enumerate(MACROS)}
CARRIER_SET = [M[k] for k in ('shoot', 'shootq', 'pass', 'passback', 'longpass', 'through', 'switch', 'dribble', 'carryspace', 'hold')]
KICK_SET = [M[k] for k in ('shoot', 'shootq', 'pass', 'passback', 'longpass', 'through', 'switch')]
GK_SET = [M[k] for k in ('gk_angle', 'gk_press', 'gk_rush', 'gk_line', 'gk_up')]
OPEN_SET = [M[k] for k in ('openfwd', 'openwide', 'openback', 'overlap', 'runbox', 'runspace')]
MODE_NONE, MODE_SHOOT, MODE_PASS = 0, 1, 2
BIG = 1e9
W2 = CFG['FIELD_W'] / 2
H2 = CFG['FIELD_H'] / 2


def isin(x, values):
    m = torch.zeros_like(x, dtype=torch.bool)
    for v in values:
        m |= (x == v)
    return m


def norm(v):
    l = v.norm(dim=-1, keepdim=True)
    return torch.where(l > 1e-9, v / l.clamp(min=1e-12), torch.zeros_like(v))


def rot(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return torch.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1)


def perp(v):
    return torch.stack([-v[..., 1], v[..., 0]], -1)


def dot(a, b):
    return (a * b).sum(-1)


def clamp_field(pt, m=0.0):
    return torch.stack([pt[..., 0].clamp(-W2 + m, W2 - m), pt[..., 1].clamp(-H2 + m, H2 - m)], -1)


def seg_dist(pt, a, b):
    """distância de pt ao segmento a->b (broadcast nas dimensões iniciais)"""
    ab = b - a
    t = (dot(pt - a, ab) / dot(ab, ab).clamp(min=1e-9)).clamp(0, 1)
    return (pt - (a + ab * t.unsqueeze(-1))).norm(dim=-1)


def masked_min(vals, mask):
    v = torch.where(mask, vals, torch.full_like(vals, BIG))
    mn, idx = v.min(dim=-1)
    return mn, idx, mn < BIG / 2


def masked_max(vals, mask):
    v = torch.where(mask, vals, torch.full_like(vals, -BIG))
    mx, idx = v.max(dim=-1)
    return mx, idx, mx > -BIG / 2


def gather2(x, idx):
    """x [B,Q,2], idx [B,P] -> [B,P,2]"""
    return torch.gather(x, 1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, 2))


def gather1(x, idx):
    return torch.gather(x, 1, idx.clamp(min=0))


class Ctx:
    """contexto compartilhado (equivalente a AI.context + tensores pareados)"""

    def __init__(self, sim):
        B, P, d = sim.B, sim.P, sim.dev
        self.B, self.P, self.d = B, P, d
        self.pos, self.vel = sim.pos, sim.vel
        self.team = sim.team.view(1, P).expand(B, P)
        self.dir = sim.dir.view(1, P).expand(B, P)
        self.dir3 = self.dir.unsqueeze(-1)
        self.ownGoal = torch.stack([-self.dir * W2, torch.zeros_like(self.dir)], -1)
        self.oppGoal = torch.stack([self.dir * W2, torch.zeros_like(self.dir)], -1)
        self.bpos = sim.bpos.view(B, 1, 2).expand(B, P, 2)
        self.bvel = sim.bvel.view(B, 1, 2).expand(B, P, 2)
        self.bposB, self.bvelB = sim.bpos, sim.bvel
        self.owner = sim.owner.view(B, 1).expand(B, P)
        self.hasOwner = self.owner >= 0
        self.ownerTeam = torch.where(self.hasOwner, sim.team[sim.owner.clamp(min=0)].view(B, 1).expand(B, P), torch.full_like(self.owner, -1))
        self.ownerSame = self.hasOwner & (self.ownerTeam == self.team)
        self.ownerOpp = self.hasOwner & (self.ownerTeam != self.team)
        self.ownerPos = gather2(sim.pos, sim.owner.view(B, 1).expand(B, P))
        self.ownerVel = gather2(sim.vel, sim.owner.view(B, 1).expand(B, P))
        self.anchor = torch.where(self.hasOwner.unsqueeze(-1), self.ownerPos, self.bpos)
        pidx = torch.arange(P, device=d).view(1, P)
        self.pidx = pidx.expand(B, P)
        self.isOwner = self.owner == pidx
        self.hasBall = self.isOwner                         # JS: c.hasBall = ball.owner === p (inclui bola na mão)
        self.held = sim.held
        self.firstTouch = (~self.hasOwner) & sim.reach
        self.keeper = sim.isKeeper
        active = sim.active_mask()
        eye = torch.eye(P, dtype=torch.bool, device=d).view(1, P, P)
        sameT = (sim.team.view(1, P, 1) == sim.team.view(1, 1, P)).expand(B, P, P)
        self.same = sameT & ~eye & active.view(B, 1, P)
        self.opp = ~sameT & active.view(B, 1, P)
        self.sameInc = self.same | (eye & active.view(B, 1, P))
        keeperQ = sim.isKeeper.view(B, 1, P)
        self.matesNK = self.same & ~keeperQ
        self.oppNK = self.opp & ~keeperQ
        self.teamHasKeeper = (self.same & keeperQ).any(-1)
        self.oppHasKeeper = (self.opp & keeperQ).any(-1)
        self.dist = (sim.pos.view(B, 1, P, 2) - sim.pos.view(B, P, 1, 2)).norm(dim=-1)   # [B,p,q]
        self.dOppNearest, self.oppNearestIdx, self.hasOppNearest = masked_min(self.dist, self.opp)
        self.posQ = sim.pos.view(B, 1, P, 2)
        self.velQ = sim.vel.view(B, 1, P, 2)
        self.dBallQ = (sim.pos - sim.bpos.view(B, 1, 2)).norm(dim=-1)     # [B,q] distância de cada jogador à bola
        self.dGoalP = (self.pos - self.oppGoal).norm(dim=-1)
        self.dOwnGoalP = (self.pos - self.ownGoal).norm(dim=-1)
        self.inOwnBoxQ = sim.in_own_box()
        self.sim = sim

    # ---- auxiliares geométricos ----
    def in_box_team(self, pt):
        """pt [B,P,...,2] no referencial do time de p"""
        team = self.team
        while team.dim() < pt.dim() - 1:
            team = team.unsqueeze(-1)
        return self.sim.in_box(pt, team.expand(pt.shape[:-1]))

    def lane_clear(self, a, b, blockers, w):
        """a, b [B,P,K,2]; blockers [B,P,1,Q] ou [B,P,K,Q] -> [B,P,K]"""
        B, P = self.B, self.P
        pq = self.posQ.unsqueeze(2)                                   # [B,1,1,Q,2]
        da = (pq - a.unsqueeze(3)).norm(dim=-1)                       # [B,P,K,Q]
        sd = seg_dist(pq, a.unsqueeze(3), b.unsqueeze(3))
        blocked = blockers & (da > 30) & (sd < CFG['PLAYER_R'] + w)
        return ~blocked.any(-1)

    def min_opp_dist_to(self, pts):
        """pts [B,P,K,2] -> menor distância de um adversário de p a cada ponto [B,P,K]"""
        dd = (self.posQ.unsqueeze(2) - pts.unsqueeze(3)).norm(dim=-1)   # [B,P,K,Q]
        mn, _, _ = masked_min(dd, self.opp.unsqueeze(2))
        return mn

    def min_mate_dist_to(self, pts):
        dd = (self.posQ.unsqueeze(2) - pts.unsqueeze(3)).norm(dim=-1)
        mn, _, _ = masked_min(dd, self.same.unsqueeze(2))
        return mn

    def predict_ball(self, t):
        k = CFG['BALL_DRAG']
        f = (1 - torch.exp(-k * t)) / k
        return clamp_field(self.bpos + self.bvel * f.unsqueeze(-1), 20)

    def home_pos(self):
        B, P = self.B, self.P
        h = self.sim.home.view(1, P, 2).expand(B, P, 2).clone()
        hx = h[..., 0] + (self.bpos[..., 0] * 0.45).clamp(-W2 * 0.4, W2 * 0.4)
        hy = h[..., 1] + self.bpos[..., 1] * 0.35
        hx = torch.where(self.ownerOpp, hx - self.dir * 150, hx)
        edge = -self.dir * (W2 - CFG['BOX_W'] - 35)
        fix = self.teamHasKeeper & (hy.abs() < CFG['BOX_H'] / 2 + 30) & ((hx - edge) * self.dir < 0)
        hx = torch.where(fix, edge, hx)
        return clamp_field(torch.stack([hx, hy], -1), 40)

    def openness(self, pts):
        """pts [B,P,K,2] -> [B,P,K]"""
        dOpp = self.min_opp_dist_to(pts)
        lane = self.lane_clear(self.bpos.unsqueeze(2).expand_as(pts), pts, self.opp.unsqueeze(2), 24).float()
        dMate = self.min_mate_dist_to(pts)
        return dOpp.clamp(max=260) + 120 * lane + dMate.clamp(max=260) * 0.8 - torch.where(dMate < 120, 150.0, 0.0)

    def best_open_point(self, nominal, radius):
        """nominal [B,P,2], radius [B,P] -> [B,P,2]"""
        offs = torch.tensor([[i, j] for i in range(-2, 3) for j in range(-2, 3)], device=self.d, dtype=torch.float)   # [25,2]
        pts = clamp_field(nominal.unsqueeze(2) + offs.view(1, 1, 25, 2) * (radius / 2).view(self.B, self.P, 1, 1), 45)
        valid = ~(self.teamHasKeeper.unsqueeze(-1) & self.in_box_team(pts))
        s = self.openness(pts) - 0.15 * (pts - nominal.unsqueeze(2)).norm(dim=-1)
        _, idx, has = masked_max(s, valid)
        best = torch.gather(pts, 2, idx.view(self.B, self.P, 1, 1).expand(-1, -1, 1, 2)).squeeze(2)
        return torch.where(has.unsqueeze(-1), best, clamp_field(nominal, 45))

    def support_nominal(self, kind):
        """kind [B,P] (código da macro) -> alvo nominal [B,P,2]"""
        a = self.anchor
        dir_ = self.dir
        py = self.pos[..., 1]
        side = torch.where(py >= a[..., 1], 1.0, -1.0)
        openfwd = torch.stack([a[..., 0] + dir_ * 280, a[..., 1] * 0.5 + side * 90], -1)
        yQ = self.posQ[..., 1].expand(self.B, self.P, self.P)
        edge, _, hasE = masked_max(yQ * side.unsqueeze(-1), self.oppNK)                # borda do bloco do lado 'side'
        edge = torch.where(hasE, edge, torch.full_like(edge, -1e9))
        yW = torch.maximum(a[..., 1] * side + 380, edge + 120) * side
        yO = torch.maximum(a[..., 1] * side + 340, edge + 120) * side
        openwide = torch.stack([a[..., 0] + dir_ * 60, yW.clamp(-H2 + 80, H2 - 80)], -1)
        overlap = torch.stack([a[..., 0] + dir_ * 320, yO.clamp(-H2 + 70, H2 - 70)], -1)
        lastX, _, hasDef = masked_max(self.posQ[..., 0].expand(self.B, self.P, self.P) * self.dir3, self.oppNK)
        lastX = torch.where(hasDef, lastX, a[..., 0] * dir_ + 200)
        rx = dir_ * torch.minimum(lastX + 70, torch.full_like(lastX, W2 - CFG['BOX_W'] * 0.7))
        runspace = torch.stack([rx, (py * 0.7).clamp(-H2 * 0.6, H2 * 0.6)], -1)
        sy = torch.where(py >= 0, 1.0, -1.0)
        runbox = torch.stack([dir_ * (W2 - CFG['BOX_W'] * 0.55), (py * 0.6 + sy * 60).clamp(-CFG['BOX_H'] / 2 * 0.7, CFG['BOX_H'] / 2 * 0.7)], -1)
        ownH = a[..., 0] * dir_ < 0
        openback = torch.stack([a[..., 0] - dir_ * torch.where(ownH, 120.0, 240.0), a[..., 1] * 0.4 + side * torch.where(ownH, 200.0, 120.0)], -1)
        out = openback
        for code, val in ((M['openfwd'], openfwd), (M['openwide'], openwide), (M['overlap'], overlap), (M['runspace'], runspace), (M['runbox'], runbox)):
            out = torch.where((kind == code).unsqueeze(-1), val, out)
        return out

    # ---- alvos de passe ----
    def through_target(self):
        """-> has [B,P], lead [B,P,2], d [B,P]"""
        B, P = self.B, self.P
        fwd = self.velQ[..., 0].expand(B, P, P) * self.dir3                         # [B,p,m]
        leadM = clamp_field(self.sim.pos + self.sim.vel * 0.9, 40).view(B, 1, P, 2).expand(B, P, P, 2)
        d = (leadM - self.pos.unsqueeze(2)).norm(dim=-1)
        prog = (leadM[..., 0] - self.pos[..., 0].unsqueeze(-1)) * self.dir3
        lane = self.lane_clear(self.bpos.unsqueeze(2).expand(B, P, P, 2), leadM, self.opp.unsqueeze(2), 26)
        on = self.min_opp_dist_to(leadM)
        ok = self.matesNK & (fwd >= 60) & (d >= 200) & (d <= 900) & (prog >= 100) & lane & (on >= 90)
        s = fwd + on * 0.5 + prog * 0.3
        _, idx, has = masked_max(s, ok)
        lead = torch.gather(leadM, 2, idx.view(B, P, 1, 1).expand(-1, -1, 1, 2)).squeeze(2)
        dd = torch.gather(d, 2, idx.unsqueeze(-1)).squeeze(-1)
        return has, lead, dd

    def switch_target(self):
        B, P = self.B, self.P
        pm = self.posQ.expand(B, P, P, 2)
        dy = (pm[..., 1] - self.pos[..., 1].unsqueeze(-1)).abs()
        d = self.dist
        prog = (pm[..., 0] - self.pos[..., 0].unsqueeze(-1)) * self.dir3
        lane = self.lane_clear(self.bpos.unsqueeze(2).expand(B, P, P, 2), pm, self.opp.unsqueeze(2), 26)
        on = self.dOppNearest.view(B, 1, P).expand(B, P, P)
        ok = self.matesNK & (dy >= 300) & (d <= 1100) & (prog >= -150) & lane & (on >= 90)
        s = on + dy * 0.3
        _, idx, has = masked_max(s, ok)
        return has, idx

    def long_pass_target(self):
        B, P = self.B, self.P
        pm = self.posQ.expand(B, P, P, 2)
        d = self.dist
        prog = (pm[..., 0] - self.pos[..., 0].unsqueeze(-1)) * self.dir3
        leadM = (self.sim.pos + self.sim.vel * 0.6).view(B, 1, P, 2).expand(B, P, P, 2)
        lane = self.lane_clear(self.bpos.unsqueeze(2).expand(B, P, P, 2), leadM, self.opp.unsqueeze(2), 24)
        on = self.min_opp_dist_to(leadM)
        ok = self.matesNK & (d >= 450) & (d <= 1500) & (prog >= 150) & lane & (on >= 85)
        s = prog + on * 0.5
        _, idx, has = masked_max(s, ok)
        return has, idx

    def safe_pass_target(self):
        B, P = self.B, self.P
        pm = self.posQ.expand(B, P, P, 2)
        d = self.dist
        prog = (pm[..., 0] - self.pos[..., 0].unsqueeze(-1)) * self.dir3
        lane = self.lane_clear(self.bpos.unsqueeze(2).expand(B, P, P, 2), pm, self.opp.unsqueeze(2), 24)
        on = self.dOppNearest.view(B, 1, P).expand(B, P, P)
        ok = self.same & (d >= 80) & (d <= 700) & (prog <= 60) & lane & (on >= 90)
        s = on - d * 0.2
        _, idx, has = masked_max(s, ok)
        return has, idx

    def best_pass(self, minDist):
        B, P = self.B, self.P
        pm = self.posQ.expand(B, P, P, 2)
        d = self.dist
        on = self.dOppNearest.view(B, 1, P).expand(B, P, P)
        keeperM = self.keeper.view(B, 1, P).expand(B, P, P)
        lane = self.lane_clear(self.bpos.unsqueeze(2).expand(B, P, P, 2), pm, self.opp.unsqueeze(2), 26)
        prog = (pm[..., 0] - self.pos[..., 0].unsqueeze(-1)) * self.dir3
        ok = self.same & (d >= minDist) & (d <= 900) & (on >= 75) & ~(keeperM & (on < 200)) & lane
        s = prog + on * 0.8 - d * 0.25
        _, idx, has = masked_max(s, ok)
        return has, idx

    # ---- espaço ----
    def ray_free(self, dirs, maxD):
        """dirs [B,P,K,2] -> [B,P,K]"""
        rel = self.posQ.unsqueeze(2) - self.pos.view(self.B, self.P, 1, 1, 2)      # [B,P,1,Q,2]
        dv = dirs.unsqueeze(3)                                                      # [B,P,K,1,2]
        along = dot(rel, dv)                                                        # [B,P,K,Q]
        lateral = (rel[..., 0] * dv[..., 1] - rel[..., 1] * dv[..., 0]).abs()
        valid = self.opp.unsqueeze(2) & (along > 0) & (along <= maxD) & (lateral < 70)
        return torch.where(valid, along, torch.full_like(along, float(maxD))).min(dim=-1).values

    def space_dir(self):
        toGoal = norm(self.oppGoal - self.pos)
        dirs = torch.stack([rot(toGoal, k * 0.35) for k in range(-4, 5)], 2)      # [B,P,9,2]
        fr = self.ray_free(dirs, 500)
        pen = torch.tensor([abs(k) * 30.0 for k in range(-4, 5)], device=self.d).view(1, 1, 9)
        s = fr - pen
        best, idx = s.max(dim=-1)
        d = torch.gather(dirs, 2, idx.view(self.B, self.P, 1, 1).expand(-1, -1, 1, 2)).squeeze(2)
        return d, best

    def best_grid_point(self, radius):
        B, P = self.B, self.P
        xs = torch.tensor([-W2 + (i + 0.5) * CFG['FIELD_W'] / 12 for i in range(12)], device=self.d)
        ys = torch.tensor([-H2 + (j + 0.5) * CFG['FIELD_H'] / 7 for j in range(7)], device=self.d)
        pts = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), -1).reshape(1, 1, 84, 2).expand(B, P, 84, 2)
        dp = (pts - self.pos.unsqueeze(2)).norm(dim=-1)
        da = (pts - self.anchor.unsqueeze(2)).norm(dim=-1)
        valid = (dp <= radius) & ~(self.teamHasKeeper.unsqueeze(-1) & self.in_box_team(pts)) & (da >= 120) & (da <= 800)
        s = self.openness(pts) + 0.5 * (pts[..., 0] - self.anchor[..., 0].unsqueeze(-1)) * self.dir3 - 0.15 * dp
        _, idx, has = masked_max(s, valid)
        best = torch.gather(pts, 2, idx.view(B, P, 1, 1).expand(-1, -1, 1, 2)).squeeze(2)
        return torch.where(has.unsqueeze(-1), best, self.home_pos())

    def freest_opp(self):
        B, P = self.B, self.P
        # on[b,p,o] = menor distância de (companheiros de p + p) até o
        dqo = self.dist.view(B, 1, P, P)                                              # [B,1,q?]: dist[b,o,q] simétrica
        on, _, _ = masked_min(dqo.expand(B, P, P, P), self.sameInc.unsqueeze(2).expand(B, P, P, P))   # min sobre q -> [B,p,o]
        ok = self.oppNK & ~(self.pidx.unsqueeze(1).expand(B, P, P) == self.owner.unsqueeze(-1))
        s = on + (self.posQ[..., 0].expand(B, P, P) - self.bpos[..., 0].unsqueeze(-1)) * (-self.dir3) * 0.3
        _, idx, has = masked_max(s, ok)
        return has, idx

    def shot_target(self):
        B, P = self.B, self.P
        keeperQ = self.keeper.view(B, 1, P).expand(B, P, P)
        gkm = self.opp & keeperQ
        gkY, _, hasGk = masked_max(self.posQ[..., 1].expand(B, P, P), gkm)
        cornerY = (CFG['GOAL_W'] / 2 - 30) * torch.where(hasGk & (gkY > 0), -1.0, 1.0)
        return torch.stack([self.oppGoal[..., 0], cornerY], -1)


class ScriptAI:
    def __init__(self, sim):
        B, P, d = sim.B, sim.P, sim.dev
        self.sim = sim
        f = lambda: torch.zeros(B, P, device=d)
        self.mode = torch.zeros(B, P, dtype=torch.long, device=d)
        self.modeT, self.chargeT, self.t = f(), f(), f()
        self.aim = torch.zeros(B, P, 2, device=d)
        self.holdN = torch.zeros(B, P, dtype=torch.long, device=d)
        self.macro = torch.full((B, P), M['home'], dtype=torch.long, device=d)
        self.lastOwner = torch.full((B, P), -2, dtype=torch.long, device=d)
        self.reactUntil = f()
        self.lastPush = f() - 9.0   # último empurrão na condução para o espaço (JS ai.lastPush)
        self.gkPress = torch.zeros(B, P, dtype=torch.bool, device=d)   # histerese do gk_press (JS ai.gkPress)
        self.lastInp = None
        self.rng = lambda: sim.rand(B, P)

    def empty(self):
        return self.sim._empty_input()

    # ---------- estado ----------
    def snapshot(self):
        return dict(mode=self.mode.clone(), modeT=self.modeT.clone(), chargeT=self.chargeT.clone(), aim=self.aim.clone(),
                    holdN=self.holdN.clone(), macro=self.macro.clone(), t=self.t.clone(), lastPush=self.lastPush.clone(), gkPress=self.gkPress.clone(),
                    lastInp=None if self.lastInp is None else {k: v.clone() for k, v in self.lastInp.items()})

    def restore(self, mask, snap):
        m3 = mask.unsqueeze(-1)
        self.mode = torch.where(mask, snap['mode'], self.mode); self.modeT = torch.where(mask, snap['modeT'], self.modeT)
        self.chargeT = torch.where(mask, snap['chargeT'], self.chargeT); self.aim = torch.where(m3, snap['aim'], self.aim)
        self.holdN = torch.where(mask, snap['holdN'], self.holdN); self.macro = torch.where(mask, snap['macro'], self.macro)
        self.t = torch.where(mask, snap['t'], self.t); self.lastPush = torch.where(mask, snap['lastPush'], self.lastPush)
        self.gkPress = torch.where(mask, snap['gkPress'], self.gkPress)
        if snap['lastInp'] is not None and self.lastInp is not None:
            self.lastInp = {k: torch.where(m3 if v.dim() == 3 else mask, snap['lastInp'][k], v) for k, v in self.lastInp.items()}

    # ---------- decisão ----------
    def choose_carrier(self, C):
        """portador (js chooseCarrierMacro v2): nota contínua por opção, vence a maior"""
        B, P = C.B, C.P
        sim = C.sim
        prevKick = isin(self.macro, KICK_SET)
        dGoal = C.dGoalP
        dOpp = C.dOppNearest
        notFirst = (~C.hasBall) & ~((dGoal < 520) | (dOpp < 130))
        pressure = (1 - dOpp / 220).clamp(0, 1)
        NEG = torch.full((B, P), -1e9, device=C.d)
        scores, macros = [], []
        def opt(mask, macro, score):
            scores.append(torch.where(mask, score, NEG)); macros.append(macro if torch.is_tensor(macro) else torch.full((B, P), macro, dtype=torch.long, device=C.d))
        # ---- chute ----
        target = C.shot_target()
        lane = C.lane_clear(C.bpos.unsqueeze(2), target.unsqueeze(2), C.oppNK.unsqueeze(2), 30).squeeze(2)
        angle = 1 - (C.pos[..., 1].abs() / (CFG['FIELD_H'] * 0.42)).clamp(max=1)
        distF = torch.where(dGoal < 300, 1.0, torch.where(dGoal < 550, 0.75, torch.where(dGoal < 800, 0.4, 0.0)))
        keeperQ = C.keeper.view(B, 1, P).expand(B, P, P)
        gkX, _, hasGk = masked_max(C.posQ[..., 0].expand(B, P, P), C.opp & keeperQ)
        gkOut = torch.where(hasGk, (((gkX - C.oppGoal[..., 0]).abs() - 120) / 200).clamp(0, 1), torch.ones_like(gkX)) * torch.where(dGoal < 900, 0.3, 0.0)
        shot = (distF * (0.5 + 0.5 * angle) * torch.where(lane, 1.0, 0.35) + gkOut) * torch.where(C.hasBall | (dGoal < 300), 1.0, 0.3)
        gkD, _, hasGkD = masked_min(C.dist, C.opp & keeperQ)
        gkD = torch.where(hasGkD, gkD, torch.full_like(gkD, 9999.0))
        quick = (dOpp < 90) | (gkD < 200) | ~C.hasBall
        opt(torch.ones_like(lane), torch.where(quick, M['shootq'], M['shoot']), shot)
        # ---- passes ----
        cnt, _, _ = self._pc(sim)
        nAct = sim.active_mask().sum(-1, keepdim=True).float()
        cells = float(FT_GX * FT_GY)
        shareQ = ((cnt.float() / (cells / nAct)).clamp(0, 2) / 2)                # [B,q] fatia média = 0.5
        owner_grid = self._pc_owner(sim)                                          # [B,GX*GY] dono de cada célula
        def pass_value(tgt, recv):
            prog = (((tgt[..., 0] - C.pos[..., 0]) * C.dir) / 500).clamp(-0.5, 1)
            dOppT = C.min_opp_dist_to(tgt.unsqueeze(2)).squeeze(2)
            space = dOppT.clamp(max=260) / 260
            # margem da linha: menor distância de um adversário (a mais de 30 px da bola) ao segmento bola->alvo
            pq = C.posQ.unsqueeze(2)
            sd = seg_dist(pq, C.bpos.unsqueeze(2).unsqueeze(3), tgt.unsqueeze(2).unsqueeze(3)).squeeze(2)   # [B,P,Q]
            far = (pq - C.bpos.unsqueeze(2).unsqueeze(3)).norm(dim=-1).squeeze(2) > 30
            mn, _, _ = masked_min(sd, C.opp & far)
            margin = ((mn - 30) / 80).clamp(0, 1)
            share = gather1(shareQ, recv)
            v = 0.5 * prog + 0.3 * space + 0.2 * margin + 0.15 * share
            v = v + 0.1 * self._voronoi_ours(C, tgt, owner_grid)
            v = torch.where((tgt[..., 0] * C.dir < -W2 * 0.5) & (dOppT < 200), v - 0.3, v)
            return v
        thrHas, thrLead, thrD = C.through_target()
        _, _, _ = thrHas, thrLead, thrD
        # índice do companheiro da enfiada (para a fatia): recomputa o argmax como em through_target
        opt(thrHas, M['through'], pass_value(thrLead, self._through_idx(C)) + 0.15 - torch.where(thrD > 800, 0.1, 0.0))
        lpHas, lpIdx = C.long_pass_target()
        opt(lpHas, M['longpass'], pass_value(gather2(sim.pos, lpIdx) + gather2(sim.vel, lpIdx) * 0.6, lpIdx) - 0.03)
        swHas, swIdx = C.switch_target()
        crowded = torch.where((C.opp & (C.dist < 260)).sum(-1) >= 2, 0.15, 0.0)
        opt(swHas, M['switch'], pass_value(gather2(sim.pos, swIdx), swIdx) + crowded)
        bpHas, bpIdx = C.best_pass(100)
        opt(bpHas, M['pass'], pass_value(gather2(sim.pos, bpIdx) + gather2(sim.vel, bpIdx) * 0.3, bpIdx))
        spHas, spIdx = C.safe_pass_target()
        opt(spHas, M['passback'], pass_value(gather2(sim.pos, spIdx), spIdx) - 0.2 + 0.4 * (pressure - 0.3).clamp(min=0))
        # ---- conduzir / proteger / dominar ----
        sd_, free = C.space_dir()
        ownHalf = torch.where(C.pos[..., 0] * C.dir < 0, 0.12, 0.0)
        carry = (free / 500).clamp(0, 1) * 0.55 * (1 - pressure) * torch.where(C.keeper, 0.3, 1.0) + torch.where(dGoal < 900, 0.1, 0.0) + ownHalf
        opt(C.hasBall, torch.where((free > 380) & (dOpp > 150), M['carryspace'], M['dribble']), carry)
        opt(C.hasBall, M['hold'], 0.1 + 0.35 * pressure * (dOpp < 70).float())
        opt(~C.hasBall, M['chase'], torch.full((B, P), 0.3, device=C.d))
        S = torch.stack(scores, -1); Mm = torch.stack(macros, -1)
        best = S.argmax(dim=-1)
        res = torch.gather(Mm, 2, best.unsqueeze(-1)).squeeze(-1)
        res = torch.where(notFirst, torch.full_like(res, M['chase']), res)
        res = torch.where(self.mode == MODE_PASS, torch.where(prevKick, self.macro, torch.full_like(res, M['pass'])), res)
        res = torch.where(self.mode == MODE_SHOOT, torch.where(prevKick, self.macro, torch.full_like(res, M['shoot'])), res)
        return res

    def _pc(self, sim):
        import features_torch as FT
        return FT.pitch_control(sim)

    def _pc_owner(self, sim):
        """dono (índice do jogador) de cada célula da grade de controle de campo [B, GX*GY], como Features.pitchControl"""
        B, P = sim.B, sim.P
        xs = torch.tensor([-W2 + (i + 0.5) * CFG['FIELD_W'] / FT_GX for i in range(FT_GX)], device=sim.dev)
        ys = torch.tensor([-H2 + (j + 0.5) * CFG['FIELD_H'] / FT_GY for j in range(FT_GY)], device=sim.dev)
        pts = torch.stack(torch.meshgrid(xs, ys, indexing='ij'), -1).reshape(1, FT_GX * FT_GY, 1, 2)   # ordem i-major: idx = i*GY + j
        d2 = ((pts - sim.pos.view(B, 1, P, 2)) ** 2).sum(-1)                                        # [B,cells,P]
        d2 = torch.where(sim.active_mask().view(B, 1, P), d2, torch.full_like(d2, BIG))
        return d2.argmin(dim=-1)                                                                   # [B,cells]

    def _voronoi_ours(self, C, pt, owner_grid):
        """1 se a célula do ponto é de um companheiro (ou minha), 0 se do adversário"""
        B, P = C.B, C.P
        i = ((pt[..., 0] + W2) / (CFG['FIELD_W'] / FT_GX)).floor().long().clamp(0, FT_GX - 1)
        j = ((pt[..., 1] + H2) / (CFG['FIELD_H'] / FT_GY)).floor().long().clamp(0, FT_GY - 1)
        cell = i * FT_GY + j
        own = torch.gather(owner_grid, 1, cell)                                                    # [B,P] índice do dono
        return (C.sim.team[own] == C.team).float()

    def _through_idx(self, C):
        """índice do companheiro escolhido pela enfiada (mesmo critério de through_target)"""
        B, P = C.B, C.P
        fwd = C.velQ[..., 0].expand(B, P, P) * C.dir3
        leadM = clamp_field(C.sim.pos + C.sim.vel * 0.9, 40).view(B, 1, P, 2).expand(B, P, P, 2)
        d = (leadM - C.pos.unsqueeze(2)).norm(dim=-1)
        prog = (leadM[..., 0] - C.pos[..., 0].unsqueeze(-1)) * C.dir3
        lane = C.lane_clear(C.bpos.unsqueeze(2).expand(B, P, P, 2), leadM, C.opp.unsqueeze(2), 26)
        on = C.min_opp_dist_to(leadM)
        ok = C.matesNK & (fwd >= 60) & (d >= 200) & (d <= 900) & (prog >= 100) & lane & (on >= 90)
        _, idx, _ = masked_max(fwd + on * 0.5 + prog * 0.3, ok)
        return idx

    def choose_offball(self, C):
        B, P = C.B, C.P
        rng = self.rng
        keeperQ = C.keeper.view(B, 1, P)
        keeperMate = (C.same & keeperQ).any(-1)
        keeperInBox = (C.same & keeperQ & C.inOwnBoxQ.view(B, 1, P)).any(-1)
        # distância de cada q ao gol do time de p: |q - ownGoal_p|
        dQown = (C.posQ - C.ownGoal.unsqueeze(2)).norm(dim=-1)                  # [B,p,q]
        closer = (C.matesNK & (dQown < C.dOwnGoalP.unsqueeze(-1))).any(-1)
        guard = (~keeperMate | ~keeperInBox) & (C.bpos[..., 0] * C.dir < W2 * 0.2) & ~closer & ~C.isOwner
        # companheiro com a bola: papéis pela estrutura (js v2). field = linha do meu time sem o portador (inclui eu)
        fieldQ = C.sameInc & ~keeperQ & ~(C.pidx.unsqueeze(1).expand(B, P, P) == C.owner.unsqueeze(-1))
        dQc = (C.posQ - C.ownerPos.unsqueeze(2)).norm(dim=-1)                   # [B,p,q] distância de q ao portador
        xQ = C.posQ[..., 0].expand(B, P, P) * C.dir3; yQ = C.posQ[..., 1].expand(B, P, P)
        xC = C.ownerPos[..., 0].unsqueeze(-1) * C.dir3; yC = C.ownerPos[..., 1].unsqueeze(-1)
        attacking = C.ownerPos[..., 0] * C.dir > W2 * 0.3
        me = C.pidx.unsqueeze(1).expand(B, P, P) == C.pidx.unsqueeze(-1)       # [B,p,q] q == p
        # apoio: mais perto do portador entre os que não estão à frente
        behind = fieldQ & ((xQ - xC) <= 40)
        _, supIdx, supHas = masked_min(dQc, behind)
        assigned = supHas.unsqueeze(-1) & (C.pidx.unsqueeze(1).expand(B, P, P) == supIdx.unsqueeze(-1))
        supRole = torch.where(attacking, M['overlap'], M['openback'])
        # largura por lado
        wideIdx, wideHas = [], []
        for side in (1.0, -1.0):
            edgeS, _, hasES = masked_max(yQ * side, C.oppNK)
            edgeS = torch.where(hasES, edgeS, torch.full_like(edgeS, -1e9))
            openS = (fieldQ & (((yQ - yC) * side > 250) | (yQ * side > (edgeS + 40).unsqueeze(-1))) & ~(assigned & (supRole.unsqueeze(-1) == M['openback']))).any(-1)
            cand = fieldQ & ~assigned & ((yQ - yC) * side >= 0)
            _, wIdx, wHas = masked_max(yQ * side, cand)
            wHas = wHas & ~openS
            wideIdx.append(wIdx); wideHas.append(wHas)
            assigned = assigned | (wHas.unsqueeze(-1) & (C.pidx.unsqueeze(1).expand(B, P, P) == wIdx.unsqueeze(-1)))
        # profundidade: o mais avançado dos livres
        _, advIdx, advHas = masked_max(xQ, fieldQ & ~assigned)
        lastX, _, hasDef = masked_max(xQ, C.oppNK)
        lastX = torch.where(hasDef, lastX, torch.full_like(lastX, W2))
        advRole = torch.where(attacking, M['runbox'], torch.where(lastX - C.ownerPos[..., 0] * C.dir > 100, M['runspace'], M['openfwd']))
        freeM = fieldQ & ~assigned
        nFree = freeM.sum(-1)
        xQ2 = torch.where(freeM & ~(C.pidx.unsqueeze(1).expand(B, P, P) == advIdx.unsqueeze(-1)), xQ, torch.full_like(xQ, -BIG))
        _, adv2Idx, adv2Has = masked_max(xQ2, xQ2 > -BIG / 2)
        adv2Has = adv2Has & (nFree >= 3)
        adv2Role = torch.where(attacking, M['runbox'], M['runspace'])
        sameOwner = torch.full((B, P), M['openbest'], dtype=torch.long, device=C.d)
        sameOwner = torch.where(adv2Has & (adv2Idx == C.pidx), adv2Role, sameOwner)
        sameOwner = torch.where(advHas & (advIdx == C.pidx), advRole, sameOwner)
        for wIdx, wHas in zip(wideIdx, wideHas):
            sameOwner = torch.where(wHas & (wIdx == C.pidx), torch.full_like(sameOwner, M['openwide']), sameOwner)
        sameOwner = torch.where(supHas & (supIdx == C.pidx), supRole, sameOwner)
        # aglomeração: companheiro (não portador) a menos de 110 px e não sou o apoio -> abrir no ponto mais livre
        mateClose = (C.same & ~(C.pidx.unsqueeze(1).expand(B, P, P) == C.owner.unsqueeze(-1)) & (C.dist < 110)).any(-1)
        sameOwner = torch.where(mateClose & (sameOwner != M['openback']), torch.full_like(sameOwner, M['openbest']), sameOwner)
        # adversário com a bola: pressiona / corta linha perigosa / último homem cobre / compacta
        dPc = (C.pos - C.ownerPos).norm(dim=-1)
        orderD = (C.matesNK & (dQc <= dPc.unsqueeze(-1))).sum(-1)   # <=: no JS o sort estável põe os companheiros antes de mim em empates
        dQown = (C.posQ - C.ownGoal.unsqueeze(2)).norm(dim=-1)
        lastMan = ~(C.matesNK & (dQown < C.dOwnGoalP.unsqueeze(-1))).any(-1)
        foHas, foIdx = C.freest_opp()
        foPos = gather2(C.sim.pos, foIdx)
        dFo = (foPos - C.ownGoal).norm(dim=-1); dBallGoal = (C.bpos - C.ownGoal).norm(dim=-1)
        dangerous = foHas & ((dFo < dBallGoal + 100) | (dFo < 600))
        oppOwner = torch.where(orderD == 0, torch.full_like(orderD, M['defend']),
                               torch.where(orderD == 1, torch.where(dangerous & ~lastMan, M['cutlane'], M['cover']),
                                           torch.where(lastMan, M['cover'], M['home'])))
        # bola solta
        dBall = C.dBallQ
        pred = C.predict_ball((dBall / 450).clamp(0, 1.2))
        dQpred = (C.posQ - pred.unsqueeze(2)).norm(dim=-1)
        dPpred = (C.pos - pred).norm(dim=-1)
        rank = (C.matesNK & (dQpred < dPpred.unsqueeze(-1))).sum(-1)
        loose = torch.where(rank == 0, torch.full_like(orderD, M["chase"]),
                            torch.where(rank == 1, torch.where((pred[..., 0] - C.pos[..., 0]) * C.dir > 0, M['cover'], M['openfwd']), torch.full_like(orderD, M["home"])))
        res = torch.where(C.ownerSame, sameOwner, torch.where(C.ownerOpp, oppOwner, loose))
        res = torch.where(guard, torch.full_like(res, M['guardgoal']), res)
        return res

    def choose_keeper(self, C):
        B, P = C.B, C.P
        rng = self.rng
        loose = ~C.hasOwner
        inBox = C.sim.in_box(C.bpos, C.team)
        myD = C.dBallQ
        oppD, _, _ = masked_min(C.dBallQ.view(B, 1, P).expand(B, P, P), C.opp)
        dBallGoal = (C.bpos - C.ownGoal).norm(dim=-1)
        rush1 = loose & inBox & ((myD < oppD - 10) | (myD < 90))
        rush2 = C.ownerOpp & inBox & (myD < 160)
        press = C.ownerOpp & (dBallGoal < torch.where(self.gkPress, 560.0, 480.0))
        up = (C.bpos[..., 0] * C.dir > W2 * 0.35) & (loose | C.ownerSame)
        towardMe = loose & (C.bvel[..., 0] * C.dir < 0) & (C.bvel.norm(dim=-1) > 400)
        line = towardMe & (dBallGoal < 500)
        res = torch.full((B, P), M['gk_angle'], dtype=torch.long, device=C.d)
        res = torch.where(line, M['gk_line'], res)
        res = torch.where(up, M['gk_up'], res)
        res = torch.where(press, M['gk_press'], res)
        res = torch.where(rush1 | rush2, M['gk_rush'], res)
        self.gkPress = torch.where(C.keeper & ~C.hasBall & ~(rush1 | rush2), press, self.gkPress)
        carrier = self.choose_carrier(C)
        return torch.where(C.hasBall, carrier, res)

    def choose_macro(self, C):
        carrier = self.choose_carrier(C)
        off = self.choose_offball(C)
        keeper = self.choose_keeper(C)
        res = torch.where(C.hasBall | C.firstTouch, carrier, off)
        return torch.where(C.keeper, keeper, res)

    # ---------- execução ----------
    def _move_to(self, out, mask, C, pt, sprint):
        d = pt - C.pos
        l = d.norm(dim=-1)
        n = d / l.clamp(min=1e-9).unsqueeze(-1)
        mv = mask & (l > 8)
        out['mx'] = torch.where(mv, n[..., 0], out['mx']); out['my'] = torch.where(mv, n[..., 1], out['my'])
        out['sprint'] = torch.where(mask, sprint & (l > 40), out['sprint'])

    def _charge_of(self, sim):
        """carga real ou da trava (chargeOf): (existe, t)"""
        has = (sim.chKind != 0) | (sim.qKind == Q_SHOT)
        t = torch.where(sim.chKind != 0, sim.chT, sim.qT)
        return has, t

    def _shoot_hold(self, out, mask, C):
        """segura o chute até a carga desejada (modo shoot já configurado)"""
        sim = C.sim
        has, ct = self._charge_of(sim)
        out['aim'] = torch.where(mask.unsqueeze(-1), self.aim, out['aim'])
        done = has & (ct >= self.chargeT)
        timeout = ~has & (self.t - self.modeT > 0.2)
        out['shoot'] = torch.where(mask, ~done, out['shoot'])
        self.mode = torch.where(mask & (done | timeout), MODE_NONE, self.mode)

    def _pass_hold(self, out, mask, C):
        out['aim'] = torch.where(mask.unsqueeze(-1), self.aim, out['aim'])
        out['pas'] = torch.where(mask, self.holdN > 0, out['pas'])
        self.holdN = torch.where(mask, self.holdN - 1, self.holdN)
        self.mode = torch.where(mask & (self.holdN < 0), MODE_NONE, self.mode)

    def execute(self, C, macro):
        """execução de linha (AI.execute) para todos; devolve dict de input"""
        sim = C.sim
        B, P, d = C.B, C.P, C.d
        rng = self.rng
        out = self.empty()
        out['aim'] = C.bpos.clone()
        first = ~C.hasBall & C.firstTouch & isin(macro, KICK_SET)
        carrier = C.hasBall | first
        macro = macro.clone()

        # ================= com a bola =================
        mc = torch.where(carrier & ~isin(macro, CARRIER_SET), torch.full_like(macro, M['dribble']), macro)
        self.mode = torch.where(carrier & (self.mode == MODE_SHOOT) & ~isin(mc, [M['shoot'], M['shootq'], M['longpass'], M['through']]), MODE_NONE, self.mode)
        self.mode = torch.where(carrier & (self.mode == MODE_PASS) & ~isin(mc, [M['pass'], M['passback'], M['switch']]), MODE_NONE, self.mode)
        # longpass
        m = carrier & (mc == M['longpass']) & (self.mode != MODE_SHOOT)
        if m.any():
            has, idx = C.long_pass_target()
            lead = gather2(sim.pos, idx) + gather2(sim.vel, idx) * 0.6
            dd = (lead - C.pos).norm(dim=-1)
            mc = torch.where(m & ~has, torch.full_like(mc, M['dribble']), mc)
            ok = m & has
            self.mode = torch.where(ok, MODE_SHOOT, self.mode); self.modeT = torch.where(ok, self.t, self.modeT)
            self.aim = torch.where(ok.unsqueeze(-1), lead, self.aim)
            self.chargeT = torch.where(ok, ((dd - 300) / 1200).clamp(0.3, 0.85) * CFG['CHARGE_MAX'], self.chargeT)
        # through
        m = carrier & (mc == M['through']) & (self.mode != MODE_SHOOT)
        if m.any():
            has, lead, dd = C.through_target()
            mc = torch.where(m & ~has, torch.full_like(mc, M['dribble']), mc)
            ok = m & has
            self.mode = torch.where(ok, MODE_SHOOT, self.mode); self.modeT = torch.where(ok, self.t, self.modeT)
            self.aim = torch.where(ok.unsqueeze(-1), lead, self.aim)
            self.chargeT = torch.where(ok, ((dd - 150) / 1400).clamp(0.05, 0.6) * CFG['CHARGE_MAX'], self.chargeT)
        # switch
        m = carrier & (mc == M['switch']) & (self.mode != MODE_PASS)
        if m.any():
            has, idx = C.switch_target()
            tp = gather2(sim.pos, idx); tv = gather2(sim.vel, idx)
            mc = torch.where(m & ~has, torch.full_like(mc, M['dribble']), mc)
            ok = m & has
            self.mode = torch.where(ok, MODE_PASS, self.mode)
            self.aim = torch.where(ok.unsqueeze(-1), tp + tv * 0.35, self.aim)
            self.holdN = torch.where(ok, 3 + torch.floor((tp - C.pos).norm(dim=-1) / 220).long(), self.holdN)
        # passback
        m = carrier & (mc == M['passback']) & (self.mode != MODE_PASS)
        if m.any():
            has, idx = C.safe_pass_target()
            tp = gather2(sim.pos, idx); tv = gather2(sim.vel, idx)
            mc = torch.where(m & ~has, torch.full_like(mc, M['hold']), mc)
            ok = m & has
            self.mode = torch.where(ok, MODE_PASS, self.mode)
            self.aim = torch.where(ok.unsqueeze(-1), tp + tv * 0.3, self.aim)
            self.holdN = torch.where(ok, 2 + torch.floor((tp - C.pos).norm(dim=-1) / 250).long(), self.holdN)
        # shoot / shootq: configura o modo (também de primeira: trava o chute no alvo)
        m = carrier & isin(mc, [M['shoot'], M['shootq']]) & (self.mode != MODE_SHOOT)
        if m.any():
            self.mode = torch.where(m, MODE_SHOOT, self.mode); self.modeT = torch.where(m, self.t, self.modeT)
            self.aim = torch.where(m.unsqueeze(-1), C.shot_target(), self.aim)
            self.chargeT = torch.where(m, torch.where(mc == M['shootq'], 0.4, (0.5 + C.dGoalP / 600).clamp(0.6, 1.0)) * CFG['CHARGE_MAX'], self.chargeT)
        # de primeira sem alvo válido: vai na bola (chase); quem já 'retornou' (segurando chute/passe) não cai aqui
        early = (isin(mc, [M['longpass'], M['through'], M['shoot'], M['shootq']]) & (self.mode == MODE_SHOOT)) | (isin(mc, [M['switch'], M['passback']]) & (self.mode == MODE_PASS))
        needChase = carrier & ~C.hasBall & ~early
        self.mode = torch.where(needChase, MODE_NONE, self.mode)
        # pass: configura o modo (sem pedido de bola no sim vetorizado)
        m = carrier & C.hasBall & (mc == M['pass']) & (self.mode != MODE_PASS)
        if m.any():
            has, idx = C.best_pass(100)
            tp = gather2(sim.pos, idx); tv = gather2(sim.vel, idx)
            mc = torch.where(m & ~has, torch.full_like(mc, M['dribble']), mc)
            ok = m & has
            self.mode = torch.where(ok, MODE_PASS, self.mode)
            self.aim = torch.where(ok.unsqueeze(-1), tp + tv * 0.3, self.aim)
            self.holdN = torch.where(ok, 2 + torch.floor((tp - C.pos).norm(dim=-1) / 250).long(), self.holdN)

        withBall = carrier & C.hasBall
        # saídas: longpass/through (segurar chute), switch/passback/pass (segurar passe)
        self._shoot_hold(out, carrier & isin(mc, [M['longpass'], M['through']]) & (self.mode == MODE_SHOOT), C)
        self._pass_hold(out, ((carrier & isin(mc, [M['switch'], M['passback']])) | (withBall & (mc == M['pass']))) & (self.mode == MODE_PASS), C)
        # shoot: segurar + andar para o gol
        ms = carrier & isin(mc, [M['shoot'], M['shootq']]) & (self.mode == MODE_SHOOT)
        if ms.any():
            self._move_to(out, ms & C.hasBall, C, C.oppGoal, torch.zeros_like(ms))
            self._shoot_hold(out, ms, C)
        # carryspace
        mcs = withBall & (mc == M['carryspace'])
        if mcs.any():
            sd, free = C.space_dir()
            out['mx'] = torch.where(mcs, sd[..., 0], out['mx']); out['my'] = torch.where(mcs, sd[..., 1], out['my'])
            out['aim'] = torch.where(mcs.unsqueeze(-1), C.pos + sd * 150, out['aim'])
            out['sprint'] = torch.where(mcs, (sim.stamina > 8) & (free > 200), out['sprint'])
            pushOk = mcs & (sim.cd['grab'] <= 0) & (free > 350) & (sim.vel.norm(dim=-1) > 100) & (self.t - self.lastPush > 0.6)
            out['special'] = torch.where(mcs, pushOk, out['special'])
            self.lastPush = torch.where(pushOk, self.t, self.lastPush)
        # hold
        mh = withBall & (mc == M['hold'])
        if mh.any():
            noPos = gather2(sim.pos, C.oppNearestIdx)
            away = torch.where(C.hasOppNearest.unsqueeze(-1), norm(C.pos - noPos), torch.stack([-C.dir, torch.zeros_like(C.dir)], -1))
            toGoal = norm(C.oppGoal - C.pos)
            st = norm(away * 0.7 + toGoal * 0.3)
            out['mx'] = torch.where(mh, st[..., 0] * 0.6, out['mx']); out['my'] = torch.where(mh, st[..., 1] * 0.6, out['my'])
            out['stance'] = torch.where(mh, torch.ones_like(mh), out['stance'])
            out['aim'] = torch.where(mh.unsqueeze(-1), C.pos + toGoal * 150, out['aim'])
            out['special'] = torch.where(mh, (C.dOppNearest < 60) & (sim.cd['dribble'] <= 0) & (rng() < 0.08), out['special'])
        # dribble (padrão com a bola)
        md = withBall & (mc == M['dribble'])
        if md.any():
            noPos = gather2(sim.pos, C.oppNearestIdx)
            dOpp = C.dOppNearest
            tgt = torch.stack([C.oppGoal[..., 0] - C.dir * 100, C.pos[..., 1] * 0.4], -1)
            steer = norm(tgt - C.pos)
            toOpp = noPos - C.pos
            side = perp(steer)
            sgn = torch.where(dot(side, toOpp) > 0, -1.0, 1.0)
            amt = torch.where(dOpp < 90, 1.5, 0.8)
            steer2 = norm(steer + side * (sgn * amt).unsqueeze(-1))
            useSteer2 = C.hasOppNearest & (dOpp < 170) & (dot(norm(toOpp), steer) > 0.2)
            steer = torch.where(useSteer2.unsqueeze(-1), steer2, steer)
            out['mx'] = torch.where(md, steer[..., 0], out['mx']); out['my'] = torch.where(md, steer[..., 1], out['my'])
            out['aim'] = torch.where(md.unsqueeze(-1), C.pos + steer * 120, out['aim'])
            relq = C.posQ - C.pos.unsqueeze(2)
            lq = relq.norm(dim=-1)
            blocking = C.opp & (lq < 300) & (dot(norm(relq), steer.unsqueeze(2)) > 0.5)
            clearAhead = ~blocking.any(-1)
            out['sprint'] = torch.where(md, clearAhead & (sim.stamina > 10) & (C.dGoalP > 600), out['sprint'])
            close = dOpp < 80
            out['stance'] = torch.where(md & close, torch.ones_like(md), out['stance'])
            out['special'] = torch.where(md & close, (sim.cd['dribble'] <= 0) & (rng() < 0.06), out['special'])

        # ================= sem a bola =================
        off = ~carrier | needChase
        self.mode = torch.where(~carrier, MODE_NONE, self.mode)
        mo = torch.where(needChase, torch.full_like(macro, M['chase']), macro)
        mo = torch.where(isin(mo, CARRIER_SET), torch.full_like(mo, M['chase']), mo)
        mo = torch.where(isin(mo, GK_SET), torch.full_like(mo, M['cover']), mo)
        mo = torch.where((mo == M['runbox']) & (C.bpos[..., 0] * C.dir < 0), torch.full_like(mo, M['openfwd']), mo)
        mo = torch.where((mo == M['defend']) & ~C.ownerOpp, torch.where(C.hasOwner, M['openback'], M['chase']), mo)
        mo = torch.where((mo == M['chase']) & C.hasOwner, torch.where(C.ownerSame, M['openback'], M['defend']), mo)
        foHas, foIdx = C.freest_opp()
        mo = torch.where((mo == M['cutlane']) & (~foHas | ~C.ownerOpp), torch.full_like(mo, M['cover']), mo)
        ballAim = C.bpos
        # openbest
        m = off & (mo == M['openbest'])
        if m.any():
            tg = C.best_grid_point(520)
            self._move_to(out, m, C, tg, (tg - C.pos).norm(dim=-1) > 220)
        # cutlane
        m = off & (mo == M['cutlane'])
        if m.any():
            foPos = gather2(sim.pos, foIdx)
            tg = C.ownerPos + (foPos - C.ownerPos) * 0.55
            dd = (tg - C.pos).norm(dim=-1)
            self._move_to(out, m, C, tg, dd > 160)
            out['stance'] = torch.where(m & (dd < 60), torch.ones_like(m), out['stance'])
        # guardgoal
        m = off & (mo == M['guardgoal'])
        if m.any():
            tg = torch.stack([C.ownGoal[..., 0] + C.dir * 70, (C.bpos[..., 1] * 0.3).clamp(-CFG['GOAL_W'] / 2 * 0.6, CFG['GOAL_W'] / 2 * 0.6)], -1)
            dd = (tg - C.pos).norm(dim=-1)
            self._move_to(out, m, C, tg, torch.ones_like(m))
            out['stance'] = torch.where(m & (dd < 60), torch.ones_like(m), out['stance'])
        # abrir espaço
        m = off & isin(mo, OPEN_SET)
        if m.any():
            radius = torch.where(mo == M['runbox'], 110.0, 140.0)
            tg = C.best_open_point(C.support_nominal(mo), radius)
            dd = (tg - C.pos).norm(dim=-1)
            fast = isin(mo, [M['overlap'], M['runspace']])
            self._move_to(out, m, C, tg, torch.where(fast, dd > 120, dd > 260))
        # cover
        m = off & (mo == M['cover'])
        if m.any():
            bspeed = C.bvel.norm(dim=-1)
            towardGoal = ~C.hasOwner & (C.bvel[..., 0] * C.dir < -150) & (bspeed > 200)
            dirB = norm(C.bvel)
            tt = (dot(C.pos - C.bpos, dirB) / bspeed.clamp(min=1)).clamp(0.05, 1.2)
            ptI = C.predict_ball(tt)
            # perigo: adversário de linha (não portador) mais perto do meu gol
            dQown = (C.posQ - C.ownGoal.unsqueeze(2)).norm(dim=-1)
            dangerM = C.oppNK & ~(C.pidx.unsqueeze(1).expand(B, P, P) == C.owner.unsqueeze(-1))
            dDan, dIdx, hasDan = masked_min(dQown, dangerM)
            danPos = gather2(sim.pos, dIdx)
            dBallGoal = (C.bpos - C.ownGoal).norm(dim=-1)
            useDan = hasDan & (dDan < dBallGoal + 150)
            tgD = danPos + norm(C.ownGoal - danPos) * 45
            tgL = C.bpos + (C.ownGoal - C.bpos) * 0.35
            tg = clamp_field(torch.where(useDan.unsqueeze(-1), tgD, tgL), 40)
            fixX = C.teamHasKeeper & C.in_box_team(tg)
            tg = torch.stack([torch.where(fixX, -C.dir * (W2 - CFG['BOX_W'] - 40), tg[..., 0]), tg[..., 1]], -1)
            dd = (tg - C.pos).norm(dim=-1)
            mT = m & towardGoal; mN = m & ~towardGoal
            self._move_to(out, mT, C, ptI, torch.ones_like(m))
            out['stance'] = torch.where(mT, torch.ones_like(m), out['stance'])
            self._move_to(out, mN, C, tg, dd > 200)
            out['stance'] = torch.where(mN & (C.dBallQ < 160), torch.ones_like(m), out['stance'])
        # defend
        m = off & (mo == M['defend'])
        if m.any():
            goalSide = norm(C.ownGoal - C.ownerPos)
            inter = C.ownerPos + goalSide * 26
            dd = C.dBallQ
            self._move_to(out, m, C, inter, dd > 150)
            out['stance'] = torch.where(m & (dd < 150), torch.ones_like(m), out['stance'])
            toBall = norm(C.bpos - C.pos)
            tk = m & (dd < 54) & (sim.cd['tackle'] <= 0) & (dot(toBall, sim.moveDir) > 0.2)
            sl = m & ~tk & (dd > 70) & (dd < 135) & (sim.cd['slide'] <= 0) & (sim.stamina > 18) & \
                (dot(C.ownerVel, C.ownerPos - C.pos) > 40) & (C.ownerVel.norm(dim=-1) > 130) & (rng() < 0.15)
            out['tackle'] = torch.where(tk | sl, torch.ones_like(m), out['tackle'])
            out['sprint'] = torch.where(tk, torch.zeros_like(m), out['sprint'])
            out['sprint'] = torch.where(sl, torch.ones_like(m), out['sprint'])
            out['stance'] = torch.where(sl, torch.zeros_like(m), out['stance'])
        # chase (bola solta)
        m = off & (mo == M['chase'])
        if m.any():
            dBall = C.dBallQ
            pred = C.predict_ball((dBall / 450).clamp(0, 1.2))
            self._move_to(out, m, C, pred, dBall > 90)
            bs = C.bvel.norm(dim=-1)
            st = (bs > CFG['CONTROL_MAX'] * 0.85) & (dBall < 150) & (dot(C.bvel, C.pos - C.bpos) > 0)
            out['stance'] = torch.where(m & st, torch.ones_like(m), out['stance'])
        # home
        m = off & (mo == M['home'])
        if m.any():
            self._move_to(out, m, C, C.home_pos(), torch.zeros_like(m))
        # mira padrão sem a bola: a bola (home mantém a bola também)
        out['aim'] = torch.where(off.unsqueeze(-1), ballAim, out['aim'])   # todos os ramos sem bola miram a bola
        return out, mc

    def keeper_execute(self, C, macro, out_line):
        """sobrescreve a saída dos goleiros (keeperExecute). out_line: saída de execute() já calculada."""
        sim = C.sim
        B, P, d = C.B, C.P, C.d
        k = C.keeper
        out = {kk: v.clone() for kk, v in out_line.items()}
        withBall = k & (C.hasBall | (C.firstTouch & isin(macro, KICK_SET) & (macro != M['shoot'])))
        # bola na mão: conduzir solta a bola; segurar espera na zona de repulsão
        heldK = withBall & C.held
        mDrop = heldK & isin(macro, [M['dribble'], M['carryspace']])
        mHold = heldK & isin(macro, [M['hold']] + GK_SET)
        for kk in ('mx', 'my'):
            out[kk] = torch.where(mDrop | mHold, torch.zeros_like(out[kk]), out[kk])
        for kk in ('shoot', 'pas', 'sprint', 'stance', 'special', 'tackle', 'throw'):
            out[kk] = torch.where(mDrop | mHold, torch.zeros_like(out[kk]), out[kk])
        out['special'] = torch.where(mDrop, torch.ones_like(mDrop), out['special'])
        out['aim'] = torch.where((mDrop | mHold).unsqueeze(-1), C.oppGoal, out['aim'])
        # passe com a bola na mão e alvo longe: arremesso
        thr = withBall & C.held & (self.mode == MODE_PASS) & ((C.pos - self.aim).norm(dim=-1) > 320) & out['pas']
        out['throw'] = torch.where(thr, torch.ones_like(thr), out['throw'])
        out['pas'] = torch.where(thr, torch.zeros_like(thr), out['pas'])

        # ---- sem a bola ----
        m = k & ~withBall
        if not m.any():
            return out
        gm = torch.where(isin(macro, GK_SET), macro, torch.full_like(macro, M['gk_angle']))
        toBall = C.bpos - C.ownGoal
        dBall = toBall.norm(dim=-1)
        speed = C.bvel.norm(dim=-1)
        loose = ~C.hasOwner
        towardMe = C.bvel[..., 0] * C.dir < 0
        xlo = torch.where(C.team == 0, -W2 + 24, W2 - 170)
        xhi = torch.where(C.team == 0, -W2 + 170, W2 - 24)
        bvx = C.bvel[..., 0]
        safeDiv = torch.where(bvx.abs() > 1e-9, bvx, torch.full_like(bvx, 1e-9))
        # limpa as saídas de linha dos goleiros
        for kk in ('mx', 'my'):
            out[kk] = torch.where(m, torch.zeros_like(out[kk]), out[kk])
        for kk in ('shoot', 'pas', 'sprint', 'stance', 'special', 'tackle', 'throw'):
            out[kk] = torch.where(m, torch.zeros_like(out[kk]), out[kk])
        out['aim'] = torch.where(m.unsqueeze(-1), C.bpos, out['aim'])
        # gk_rush
        mr = m & (gm == M['gk_rush'])
        myD = C.dBallQ
        tgR = torch.where(loose.unsqueeze(-1), C.predict_ball((myD / 500).clamp(0, 0.5)), C.bpos)
        tkR = mr & ~loose & (myD < 58) & (sim.cd['tackle'] <= 0)
        # gk_press
        mp = m & (gm == M['gk_press'])
        depthP = torch.minimum(torch.maximum(torch.full_like(dBall, 60.0), dBall - 70), torch.full_like(dBall, CFG['BOX_W'] - 25))
        tgP = C.ownGoal + norm(toBall) * depthP.unsqueeze(-1)
        tgP = torch.stack([tgP[..., 0], tgP[..., 1].clamp(-CFG['BOX_H'] / 2 * 0.8, CFG['BOX_H'] / 2 * 0.8)], -1)
        # gk_line
        txL = (xlo - C.bpos[..., 0]) / safeDiv
        yL = torch.where(loose & towardMe & (speed > 180) & (txL > 0) & (txL < 1.5), C.bpos[..., 1] + C.bvel[..., 1] * txL, C.bpos[..., 1])
        tgL = torch.stack([-C.dir * (W2 - 30), yL.clamp(-CFG['GOAL_W'] / 2 * 0.85, CFG['GOAL_W'] / 2 * 0.85)], -1)
        # gk_up
        depthU = CFG['BOX_W'] - 30
        tgU = C.ownGoal + norm(toBall) * depthU
        lim = -C.dir * (W2 - depthU)
        tgU = torch.stack([torch.minimum(torch.maximum(tgU[..., 0], torch.minimum(xlo, lim)), torch.maximum(xhi, lim)),
                           tgU[..., 1].clamp(-CFG['BOX_H'] / 2 * 0.8, CFG['BOX_H'] / 2 * 0.8)], -1)
        # gk_angle
        depthA = (50 + dBall * 0.06).clamp(50, 115)
        tgA = C.ownGoal + norm(toBall) * depthA.unsqueeze(-1)
        txA = (C.pos[..., 0] - C.bpos[..., 0]) / safeDiv
        useA = loose & towardMe & (speed > 180) & (txA > 0) & (txA < 1.2)
        tgA = torch.where(useA.unsqueeze(-1), torch.stack([C.pos[..., 0], C.bpos[..., 1] + C.bvel[..., 1] * txA], -1), tgA)
        tgA = torch.stack([torch.minimum(torch.maximum(tgA[..., 0], xlo), xhi), tgA[..., 1].clamp(-CFG['GOAL_W'] / 2 * 0.85, CFG['GOAL_W'] / 2 * 0.85)], -1)

        tg = torch.where((gm == M['gk_rush']).unsqueeze(-1), tgR,
             torch.where((gm == M['gk_line']).unsqueeze(-1), tgL,
             torch.where((gm == M['gk_up']).unsqueeze(-1), tgU, tgA)))
        sprint = (gm == M['gk_rush'])
        mn = m & ~mp
        self._move_to(out, mn, C, tg, sprint)
        out['tackle'] = torch.where(tkR, torch.ones_like(tkR), out['tackle'])
        stanceN = (dBall < 300) & (speed < 260) & ~sprint & ~out['tackle']
        out['stance'] = torch.where(mn, stanceN, out['stance'])
        # gk_press: postura defensiva, sem tackle/mergulho
        dP = (tgP - C.pos).norm(dim=-1)
        self._move_to(out, mp, C, tgP, dP > 120)
        out['stance'] = torch.where(mp, torch.ones_like(mp), out['stance'])
        # mergulho: último recurso (não no gk_press)
        txD = txA
        yAt = C.bpos[..., 1] + C.bvel[..., 1] * txD
        dy = yAt - C.pos[..., 1]
        onTarget = yAt.abs() < CFG['GOAL_W'] / 2 + 12
        reachRunning = dy.abs() < CFG['SPRINT'] * txD + 20
        dive = mn & loose & (sim.cd['dive'] <= 0) & towardMe & (speed > 260) & (txD > 0.02) & (txD < 0.7) & onTarget & ~reachRunning & (dy.abs() < 190)
        out['special'] = torch.where(dive, torch.ones_like(dive), out['special'])
        out['stance'] = torch.where(dive, torch.zeros_like(dive), out['stance'])
        out['tackle'] = torch.where(dive, torch.zeros_like(dive), out['tackle'])
        out['mx'] = torch.where(dive, torch.zeros_like(out['mx']), out['mx'])
        out['my'] = torch.where(dive, torch.sign(dy), out['my'])
        out['aim'] = torch.where(dive.unsqueeze(-1), torch.stack([C.pos[..., 0], yAt], -1), out['aim'])
        return out

    def keeper_prestep(self, C, macro):
        """parte de keeperExecute que roda antes de execute(): coerção gk->hold e chutão sem alvo"""
        k = C.keeper
        withBall = k & (C.hasBall | (C.firstTouch & isin(macro, KICK_SET) & (macro != M['shoot'])))
        mc = torch.where(withBall & isin(macro, GK_SET), torch.full_like(macro, M['hold']), macro)
        m = withBall & (mc == M['longpass']) & (self.mode != MODE_SHOOT)
        if m.any():
            has, _ = C.long_pass_target()
            ok = m & ~has
            self.mode = torch.where(ok, MODE_SHOOT, self.mode); self.modeT = torch.where(ok, self.t, self.modeT)
            aimY = (self.rng() - 0.5) * 500
            self.aim = torch.where(ok.unsqueeze(-1), torch.stack([C.oppGoal[..., 0] * 0.55, aimY], -1), self.aim)
            self.chargeT = torch.where(ok, torch.full_like(self.chargeT, 0.75 * CFG['CHARGE_MAX']), self.chargeT)
        return mc

    # ---------- decide / think ----------
    def decide(self, C, macroFn=None):
        self.t = self.t + DT
        self.mode = torch.where(~C.hasBall, MODE_NONE, self.mode)
        macro = macroFn(C) if macroFn is not None else self.choose_macro(C)
        self.macro = macro
        mc = self.keeper_prestep(C, macro)
        out, _ = self.execute(C, mc)
        out = self.keeper_execute(C, mc, out)
        return out

    def think(self, sim, macroFn=None):
        """input para todos os jogadores, com tempo de reação (só no script, como no JS)"""
        C = Ctx(sim)
        B, P = C.B, C.P
        ownerId = C.owner
        uninit = self.lastOwner == -2
        self.lastOwner = torch.where(uninit, ownerId, self.lastOwner)
        changed = ownerId != self.lastOwner
        self.lastOwner = torch.where(changed, ownerId, self.lastOwner)
        base = torch.where(C.keeper, CFG['BOT_REACTION_GK'], CFG['BOT_REACTION'])
        self.reactUntil = torch.where(changed, self.t + base * (0.7 + 0.6 * self.rng()), self.reactUntil)
        hold = torch.zeros(B, P, dtype=torch.bool, device=C.d)
        if macroFn is None and self.lastInp is not None:
            hold = (self.t < self.reactUntil)
        snap = self.snapshot()
        out = self.decide(C, macroFn)
        if hold.any():
            self.restore(hold, snap)
            self.t = torch.where(hold, snap['t'] + DT, self.t)
            held = {k: v.clone() for k, v in self.lastInp.items()}
            held['special'] = torch.zeros_like(held['special']); held['tackle'] = torch.zeros_like(held['tackle'])
            out = {k: torch.where(hold.unsqueeze(-1) if v.dim() == 3 else hold, held[k], v) for k, v in out.items()}
        newLast = {k: v.clone() for k, v in out.items()}
        if self.lastInp is None:
            self.lastInp = newLast
        else:
            self.lastInp = {k: torch.where(hold.unsqueeze(-1) if v.dim() == 3 else hold, self.lastInp[k], newLast[k]) for k, v in newLast.items()}
        return out
