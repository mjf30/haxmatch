"""Simulação vetorizada do HaxMatch em PyTorch (B partidas em paralelo, na GPU).

Reproduz as regras de js/game.js com o mesmo CFG (lido de js/config.js), para
treinar uma política de controle total (movimento, mira, botões) por PPO.
Diferenças conscientes em relação ao JS (documentadas em cada ponto):
  - disputas simultâneas (vários jogadores na bola no mesmo tick) são resolvidas
    por "mais perto vence" em vez da ordem de iteração;
  - o efeito do chute vem de uma ação discreta (-1/0/+1), não do arrasto do mouse;
  - o mergulho do goleiro e o dash usam a direção de movimento (ou a mira);
  - sem pedir bola, sem arremesso separado (o goleiro com a bola nas mãos passa/chuta).
"""
import json
import math
import re
import os

import torch

# ---------------------------------------------------------------- CFG do JS
def load_cfg():
    path = os.path.join(os.path.dirname(__file__), '..', 'js', 'config.js')
    src = open(path, encoding='utf-8').read()
    body = src[src.index('const CFG = {') + len('const CFG = '):]
    body = body[:body.index('\n};') + 2]
    body = re.sub(r'//[^\n]*', '', body)                 # comentários
    body = re.sub(r'(\d)\s*\*\s*(\d)', lambda m: str(int(m.group(1)) * int(m.group(2))), body)  # 6 * 60
    body = re.sub(r'1 / 60', str(1 / 60), body)
    body = re.sub(r"([A-Z_][A-Z0-9_]*)\s*:", r'"\1":', body)  # chaves
    body = re.sub(r"'", '"', body)
    body = re.sub(r',\s*}', '}', body)
    body = re.sub(r',\s*]', ']', body)
    return json.loads(body)

CFG = load_cfg()
DT = 1.0 / 60.0

# índices de ação (multi-discreta)
MOVE_DIRS = torch.tensor([[0.0, 0.0]] + [[math.cos(a), math.sin(a)] for a in [i * math.pi / 4 for i in range(8)]])  # 9
AIM_SECTORS = 16
N_MOVE, N_AIM, N_CURVE = 9, AIM_SECTORS, 3

ACT_NONE, ACT_TACKLE, ACT_SLIDE, ACT_DASH, ACT_DRIBBLE, ACT_GKDIVE = 0, 1, 2, 3, 4, 5
ST_NONE, ST_DRIB, ST_DEF = 0, 1, 2
CH_NONE, CH_SHOT, CH_PASS = 0, 1, 2
Q_NONE, Q_SHOT, Q_PASS, Q_PUSH = 0, 1, 2, 3
M_KICKOFF, M_PLAY, M_GOAL, M_END = 0, 1, 2, 3


class TorchSim:
    def __init__(self, B, team_size, device='cuda', seconds=None, seed=0):
        self.B, self.T = B, team_size
        self.P = 2 * team_size
        self.dev = torch.device(device)
        self.seconds = seconds or CFG['MATCH_TIME']
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        d, B, P = self.dev, self.B, self.P
        f = lambda *s: torch.zeros(*s, device=d)
        b = lambda *s: torch.zeros(*s, dtype=torch.bool, device=d)
        i = lambda *s: torch.zeros(*s, dtype=torch.long, device=d)
        self.team = torch.tensor([0] * self.T + [1] * self.T, device=d)          # [P]
        self.dir = torch.where(self.team == 0, 1.0, -1.0).to(d)               # [P] direção de ataque
        self.idx = torch.tensor(list(range(self.T)) * 2, device=d)             # [P]
        self.r = torch.full((P,), float(CFG['PLAYER_R']), device=d)
        self.home = self._formation()                                         # [P,2]
        # jogadores
        self.pos, self.vel, self.facing, self.moveDir = f(B, P, 2), f(B, P, 2), f(B, P, 2), f(B, P, 2)
        self.moving = b(B, P)
        self.stamina, self.exhausted, self.regenDelay = f(B, P), b(B, P), f(B, P)
        self.effortBar, self.effortT, self.lastSprintTap = f(B, P), f(B, P), f(B, P) - 10.0   # JS: lastSprintTap = -10
        self.stance, self.sprinting = i(B, P), b(B, P)
        self.act, self.actT, self.actDur, self.actDir = i(B, P), f(B, P), f(B, P), f(B, P, 2)
        self.actHit, self.actRolled, self.actBody = b(B, P), b(B, P), b(B, P)
        self.recover, self.fallen, self.getup = f(B, P), f(B, P), f(B, P)
        self.cd = {k: f(B, P) for k in ['tackle', 'slide', 'dash', 'dribble', 'dive', 'grab', 'gloves', 'through']}
        self.chKind, self.chT, self.chRel, self.chDir, self.chSpin = i(B, P), f(B, P), f(B, P), f(B, P, 2), f(B, P)
        self.qKind, self.qT, self.qAge, self.qCharging, self.qDir, self.qSpin = i(B, P), f(B, P), f(B, P), b(B, P), f(B, P, 2), f(B, P)
        self.armedShoot, self.armedPass = b(B, P), b(B, P)
        self.held, self.holdT = b(B, P), f(B, P)
        self.isKeeper = b(B, P)
        self.dribbleN, self.dribbleChainT, self.dribbleLag = i(B, P), f(B, P), f(B, P)
        self.pushFlash = f(B, P)
        self.reach = b(B, P)                 # alvo disponível (bola solta ao alcance realizável)
        self.now = f(B)
        # bola
        self.bpos, self.bvel, self.bprev = f(B, 2), f(B, 2), f(B, 2)
        self.bspin = f(B)
        self.owner, self.lastTouch, self.lastTeam, self.lock = i(B) - 1, i(B) - 1, i(B) - 1, i(B) - 1
        # partida
        self.time, self.state, self.stateT = f(B), i(B), f(B)
        self.score = i(B, 2)
        self.kickoffTeam = i(B)
        # inputs anteriores (edge)
        self.prev = self._empty_input()
        self.events = {}
        self.reset_all()

    # ------------------------------------------------------------ util
    def _formation(self):
        forms = {3: [(-0.9, 0), (-0.4, -0.35), (-0.4, 0.35)],
                 4: [(-0.9, 0), (-0.5, 0), (-0.27, -0.5), (-0.27, 0.5)],
                 5: [(-0.9, 0), (-0.55, -0.4), (-0.55, 0.4), (-0.25, -0.55), (-0.25, 0.55)]}
        out = []
        for t in range(2):
            d = 1 if t == 0 else -1
            for x, y in forms[self.T]:
                out.append([x * d * CFG['FIELD_W'] / 2, y * d * CFG['FIELD_H'] / 2])
        return torch.tensor(out, device=self.dev)

    def _empty_input(self):
        B, P, d = self.B, self.P, self.dev
        return dict(mx=torch.zeros(B, P, device=d), my=torch.zeros(B, P, device=d),
                    aim=torch.zeros(B, P, 2, device=d), curve=torch.zeros(B, P, device=d),
                    shoot=torch.zeros(B, P, dtype=torch.bool, device=d), pas=torch.zeros(B, P, dtype=torch.bool, device=d),
                    sprint=torch.zeros(B, P, dtype=torch.bool, device=d), stance=torch.zeros(B, P, dtype=torch.bool, device=d),
                    special=torch.zeros(B, P, dtype=torch.bool, device=d), tackle=torch.zeros(B, P, dtype=torch.bool, device=d))

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.dev)

    def in_box(self, pt, team):
        """pt [...,2], team [...] (0/1) -> bool"""
        W2 = CFG['FIELD_W'] / 2
        xin = torch.where(team == 0, pt[..., 0] <= -W2 + CFG['BOX_W'], pt[..., 0] >= W2 - CFG['BOX_W'])
        return xin & (pt[..., 1].abs() <= CFG['BOX_H'] / 2)

    def in_own_box(self):
        return self.in_box(self.pos, self.team.view(1, -1).expand(self.B, -1))

    def in_own_half(self):
        return torch.where(self.team.view(1, -1) == 0, self.pos[..., 0] <= 0, self.pos[..., 0] >= 0)

    @staticmethod
    def clamp_arena(pt, r):
        """limites com a boca do gol aberta (jogadores e bola dominada)"""
        W2, H2 = CFG['FIELD_W'] / 2, CFG['FIELD_H'] / 2
        x, y = pt[..., 0], pt[..., 1]
        mouth = y.abs() < CFG['GOAL_W'] / 2 - r
        inside = x.abs() > W2
        x2 = torch.where(inside & ~mouth, x.clamp(-W2 + r, W2 - r), torch.where(inside, x.clamp(-W2 - CFG['GOAL_D'] + r, W2 + CFG['GOAL_D'] - r), x))
        y2 = torch.where(x2.abs() > W2, y.clamp(-CFG['GOAL_W'] / 2 + r, CFG['GOAL_W'] / 2 - r), y.clamp(-H2 + r, H2 - r))
        return torch.stack([x2, y2], -1)

    # ------------------------------------------------------------ reset / kickoff
    def reset_all(self):
        idx = torch.arange(self.B, device=self.dev)
        self.time[:] = self.seconds
        self.score[:] = 0
        self.stamina[:] = CFG['STAMINA_MAX']; self.effortBar[:] = 1.0; self.exhausted[:] = False; self.regenDelay[:] = 0
        self.lastSprintTap[:] = -10.0
        self.kickoffTeam = (self.rand(self.B) < 0.5).long()
        self.now[:] = 0
        self.prev = self._empty_input()
        self.kickoff(idx)

    def kickoff(self, idx):
        """reinício das partidas idx (tensor de índices)"""
        n = idx.numel()
        if n == 0:
            return
        self.pos[idx] = self.home.view(1, self.P, 2).expand(n, -1, -1)
        self.vel[idx] = 0
        self.act[idx] = ACT_NONE; self.chKind[idx] = CH_NONE; self.qKind[idx] = Q_NONE
        self.fallen[idx] = 0; self.getup[idx] = 0; self.recover[idx] = 0
        self.isKeeper[idx] = False; self.held[idx] = False; self.holdT[idx] = 0
        self.stance[idx] = ST_NONE; self.sprinting[idx] = False; self.effortT[idx] = 0
        self.dribbleN[idx] = 0; self.dribbleChainT[idx] = 0; self.dribbleLag[idx] = 0
        for k in self.cd:
            self.cd[k][idx] = 0
        self.bpos[idx] = 0; self.bvel[idx] = 0; self.bspin[idx] = 0; self.owner[idx] = -1; self.lock[idx] = -1
        # saída: jogador mais avançado do time do kickoff no meio, de frente para o próprio campo
        team = self.kickoffTeam[idx]
        d = torch.where(team == 0, 1.0, -1.0)
        kicker = torch.where(team == 0, torch.full_like(team, self.T - 1), torch.full_like(team, 2 * self.T - 1))
        rk = self.r[kicker]
        self.pos[idx, kicker, 0] = -d * (rk + 4); self.pos[idx, kicker, 1] = 0
        self.facing[idx, kicker, 0] = -d; self.facing[idx, kicker, 1] = 0
        self.moveDir[idx, kicker] = self.facing[idx, kicker]
        self.owner[idx] = kicker; self.lastTouch[idx] = kicker; self.lastTeam[idx] = team
        self.bpos[idx, 0] = -d * (rk * 2 + CFG['BALL_R'] + CFG['CARRY_DIST'] + 4); self.bpos[idx, 1] = 0
        self.state[idx] = M_KICKOFF; self.stateT[idx] = CFG['KICKOFF_FREEZE']

    # ------------------------------------------------------------ passo
    def step(self, inp):
        """inp: dict como _empty_input (mx,my em [-1,1]; aim ponto no mundo; curve [-1,1]; botões bool)"""
        B, P, d = self.B, self.P, self.dev
        self.events = {}
        self.now += DT
        # estado da partida
        goal = self.state == M_GOAL
        self.stateT = torch.where(goal | (self.state == M_KICKOFF), self.stateT - DT, self.stateT)
        restart = goal & (self.stateT <= 0)
        self.kickoff(restart.nonzero().squeeze(1))
        start = (self.state == M_KICKOFF) & (self.stateT <= 0)
        self.state = torch.where(start, torch.full_like(self.state, M_PLAY), self.state)
        play = self.state == M_PLAY
        self.time = torch.where(play, self.time - DT, self.time)
        ended = play & (self.time <= 0)
        self.state = torch.where(ended, torch.full_like(self.state, M_END), self.state)
        play = self.state == M_PLAY
        frozen = ~play                                                        # [B]
        fz = frozen.view(B, 1).expand(B, P)
        # inputs congelados = vazios
        z = self._empty_input()
        cur = {k: torch.where(fz if v.dim() == 2 else fz.unsqueeze(-1), z[k], v) for k, v in inp.items()}
        prev = self.prev

        self._update_players(cur, prev, fz)
        self._collide_players()
        self._keeper_repel()
        self.pos = self.clamp_arena(self.pos, self.r.view(1, P))
        self._clamp_keeper_box()
        self._update_ball()
        self._resolve_locks()
        self._contacts(play)
        loose = self.owner < 0
        self.bpos = torch.where(loose.unsqueeze(-1), self.bpos, self.clamp_arena(self.bpos, CFG['BALL_R']))
        self._walls_free(loose)
        self._gloves(play)
        self._check_goal(play)
        self.prev = {k: v.clone() for k, v in cur.items()}
        return self.events

    # ------------------------------------------------------------ jogadores
    def _update_players(self, inp, prev, fz):
        B, P, d = self.B, self.P, self.dev
        pidx = torch.arange(P, device=d).view(1, P).expand(B, P)
        hasBall = (self.owner.view(B, 1) == pidx) & ~self.held
        held = (self.owner.view(B, 1) == pidx) & self.held
        for k in self.cd:
            self.cd[k] = (self.cd[k] - DT).clamp(min=0)
        self.recover = (self.recover - DT).clamp(min=0); self.fallen = (self.fallen - DT).clamp(min=0)
        self.getup = (self.getup - DT).clamp(min=0); self.pushFlash = (self.pushFlash - DT).clamp(min=0)
        self.effortT = (self.effortT - DT).clamp(min=0)
        full = (self.stamina >= CFG['STAMINA_MAX'] - 0.01) & ~(self.isKeeper & self.in_own_half())   # líbero: recarga normal
        self.effortBar = torch.where(self.effortT > 0, (self.effortBar - (0.5 / CFG['EFFORT_DUR']) * DT).clamp(min=0),
                                     (self.effortBar + DT / torch.where(full, CFG['EFFORT_RECHARGE_FULL'], CFG['EFFORT_RECHARGE'])).clamp(max=1))
        self.dribbleLag = (self.dribbleLag - DT).clamp(min=0)
        chain = self.dribbleChainT > 0
        self.dribbleChainT = torch.where(chain, (self.dribbleChainT - DT).clamp(min=0), self.dribbleChainT)
        expired = chain & (self.dribbleChainT == 0) & (self.dribbleN == 1) & (self.act == ACT_NONE)
        self.dribbleN = torch.where(expired, 0, self.dribbleN)
        self.cd['dribble'] = torch.where(expired, torch.full_like(self.cd['dribble'], CFG['DRIBBLE_CD_SINGLE']), self.cd['dribble'])
        noball = ~hasBall & ~held
        self.dribbleLag = torch.where(noball, 0.0, self.dribbleLag)
        self.dribbleN = torch.where(noball & (self.dribbleN == 1) & (self.act == ACT_NONE), 0, self.dribbleN)

        # mira e movimento
        aimV = inp['aim'] - self.pos
        aimLocked = ((self.chKind == CH_SHOT) | (self.qKind == Q_SHOT))
        aimOk = (aimV.norm(dim=-1) > 2) & ~aimLocked
        self.facing = torch.where(aimOk.unsqueeze(-1), aimV / aimV.norm(dim=-1, keepdim=True).clamp(min=1e-6), self.facing)
        mv = torch.stack([inp['mx'], inp['my']], -1)
        ml = mv.norm(dim=-1)
        self.moving = ml > 0.1
        self.moveDir = torch.where(self.moving.unsqueeze(-1), mv / ml.clamp(min=1e-6).unsqueeze(-1), self.moveDir)
        self.stance = torch.where(inp['stance'], torch.where(hasBall, ST_DRIB, ST_DEF), torch.full_like(self.stance, ST_NONE))

        # sprint / arrancada
        pressedSprint = inp['sprint'] & ~prev['sprint']
        releasedSprint = ~inp['sprint'] & prev['sprint']
        nowP = self.now.view(B, 1)
        effort = pressedSprint & (nowP - self.lastSprintTap < CFG['DOUBLE_TAP']) & (self.effortBar >= 1) & ~held
        self.effortT = torch.where(effort, torch.full_like(self.effortT, CFG['EFFORT_DUR']), self.effortT)
        self.effortBar = torch.where(effort, torch.full_like(self.effortBar, 0.5), self.effortBar)
        self.lastSprintTap = torch.where(pressedSprint | releasedSprint, nowP.expand(B, P), self.lastSprintTap)
        self.sprinting = inp['sprint'] & (self.stamina > 0) & ~held & ~self.exhausted

        incap = (self.fallen > 0) | (self.getup > 0)
        inAct = self.act != ACT_NONE
        # ações em andamento
        self._run_actions(inAct)
        # incapacitado: freia
        self.vel = torch.where((incap & ~inAct).unsqueeze(-1), self.vel * max(0.0, 1 - 8 * DT), self.vel)
        self.chKind = torch.where(incap & ~inAct, CH_NONE, self.chKind)
        self.qKind = torch.where(incap & ~inAct, Q_NONE, self.qKind)
        free = ~inAct & ~incap
        # alvo (reach): bola solta ao alcance
        self._compute_reach()
        self._handle_actions(inp, prev, free & ~fz, hasBall, held)
        # recomputa posse (ações podem ter mudado)
        hasBall = (self.owner.view(B, 1) == pidx) & ~self.held
        held = (self.owner.view(B, 1) == pidx) & self.held
        self._move(inp, free & (self.act == ACT_NONE), hasBall, held)

        # stamina
        keeperFree = self.isKeeper & self.in_own_half()
        drain = self.sprinting & self.moving & ~keeperFree & (self.act == ACT_NONE)
        self.stamina = torch.where(drain, (self.stamina - CFG['STAMINA_SPRINT'] * DT).clamp(min=0),
                                   torch.where(self.regenDelay > 0, self.stamina, (self.stamina + CFG['STAMINA_REGEN'] * DT).clamp(max=CFG['STAMINA_MAX'])))
        self.regenDelay = torch.where(~drain & (self.regenDelay > 0), (self.regenDelay - DT).clamp(min=0), self.regenDelay)
        newEx = (self.stamina <= 0) & ~self.exhausted
        self.exhausted = self.exhausted | newEx
        self.regenDelay = torch.where(newEx, torch.full_like(self.regenDelay, CFG['STAMINA_REGEN_DELAY']), self.regenDelay)
        self.exhausted = self.exhausted & ~(self.stamina >= CFG['STAMINA_MAX'] * CFG['EXHAUST_RECOVER'])
        self.stamina = torch.where(keeperFree, torch.full_like(self.stamina, CFG['STAMINA_MAX']), self.stamina)
        self.exhausted = self.exhausted & ~keeperFree
        # goleiro com a bola nas mãos: tempo e área
        self.holdT = torch.where(held, self.holdT + DT, self.holdT)
        drop = held & ((self.holdT > CFG['GK_HOLD_MAX']) | ~self.in_own_box())
        self.held = self.held & ~drop
        self.pos = self.pos + self.vel * DT

    def _compute_reach(self):
        """alvo realizável: bola solta a ACTION_RADIUS e alcançável na janela LOCK_MAX à velocidade atual (previsão simples, sem paredes)"""
        B, P = self.B, self.P
        loose = (self.owner < 0).view(B, 1)
        rel = self.bpos.view(B, 1, 2) - self.pos
        dist = rel.norm(dim=-1)
        near = dist < self.r.view(1, P) + CFG['BALL_R'] + CFG['ACTION_RADIUS']
        cap = self._snap_cap()
        reachR = self._ball_hitbox() + CFG['BALL_R']
        ok = torch.zeros_like(near)
        bp, bv = self.bpos.clone(), self.bvel.clone()
        dt = 1 / 30
        t = 0.0
        while t <= CFG['LOCK_MAX'] + 1e-6:
            dd = (bp.view(B, 1, 2) - self.pos).norm(dim=-1)
            ok = ok | (dd - reachR <= cap * t)
            s = bv.norm(dim=-1)
            s2 = (s * math.exp(-CFG['BALL_DRAG'] * dt) - CFG['BALL_DECEL'] * dt).clamp(min=0)
            bv = bv * (s2 / s.clamp(min=1e-6)).unsqueeze(-1)
            bp = bp + bv * dt
            t += dt
        self.reach = loose & near & ok

    def _snap_cap(self):
        cap = torch.where(self.sprinting, CFG['SPRINT'], CFG['SPEED']) * torch.ones_like(self.stamina)
        cap = torch.where(self.effortT > 0, torch.full_like(cap, CFG['SPRINT'] * CFG['EXTRA_EFFORT']), cap)
        cap = torch.where(self.exhausted & (self.effortT <= 0), cap * CFG['MUL_EXHAUSTED'], cap)
        cap = torch.where(self.recover > 0, cap * CFG['MUL_RECOVER'], cap)
        return cap.clamp(max=CFG['LOCK_SNAP_SPEED'])

    def _ball_hitbox(self):
        r = self.r.view(1, -1)
        gk = (self.stance == ST_DEF) & self.isKeeper & self.in_own_box()
        return torch.where(gk, r + CFG['GK_DEF_MARGIN'], torch.where(self.stance == ST_DEF, r + CFG['GRAB_MARGIN_DEF'], r * CFG['HITBOX_MUL'] + CFG['GRAB_MARGIN']))

    def _lunge_dir(self):
        return torch.where(self.moving.unsqueeze(-1), self.moveDir, self.facing)

    # ------------------------------------------------------------ ações de botão
    def _handle_actions(self, inp, prev, ok, hasBall, held):
        B, P, d = self.B, self.P, self.dev
        pressed = lambda k: inp[k] & ~prev[k] & ok
        released = lambda k: ~inp[k] & prev[k] & ok
        canKick = (hasBall | held) & ok
        inZone = self.reach & ok
        # botões armados
        for k, arm in (('shoot', 'armedShoot'), ('pas', 'armedPass')):
            a = getattr(self, arm)
            a = torch.where(pressed(k), True, a)
            a = torch.where(~inp[k], False, a)
            setattr(self, arm, a)
        kickDir = self._kick_dir(inp)

        # ---- ação travada (queued) ----
        q = (self.qKind != Q_NONE) & ok
        self.qAge = torch.where(q, self.qAge + DT, self.qAge)
        cancel = q & (~inZone | canKick)
        expired = q & ~cancel & (self.qAge > CFG['LOCK_MAX'])
        self.recover = torch.where(expired, torch.maximum(self.recover, torch.full_like(self.recover, 0.15)), self.recover)
        self.qKind = torch.where(cancel | expired, Q_NONE, self.qKind)
        q = (self.qKind != Q_NONE) & ok
        qc = q & self.qCharging
        self.qT = torch.where(qc, self.qT + DT, self.qT)
        self.qSpin = torch.where(qc & (self.qKind == Q_SHOT), (self.qSpin + inp['curve'] * CFG['SPIN_MAX'] * 0.08).clamp(-CFG['SPIN_MAX'], CFG['SPIN_MAX']), self.qSpin)
        self.qCharging = self.qCharging & ~(qc & (((self.qKind == Q_SHOT) & released('shoot')) | ((self.qKind == Q_PASS) & released('pas'))))
        self.qKind = torch.where(q & pressed('tackle'), Q_NONE, self.qKind)   # cancela para tacklear
        okq = ok & (self.qKind == Q_NONE)                                    # com trava ativa, nada mais
        # novas travas (bola no alvo, sem a bola)
        newLock = okq & ~canKick & inZone
        ns = newLock & self.armedShoot
        npz = newLock & ~ns & self.armedPass
        npu = newLock & ~ns & ~npz & pressed('special')
        self.armedShoot = self.armedShoot & ~ns; self.armedPass = self.armedPass & ~npz
        anyNew = ns | npz | npu
        self.qKind = torch.where(ns, Q_SHOT, torch.where(npz, Q_PASS, torch.where(npu, Q_PUSH, self.qKind)))
        self.qT = torch.where(anyNew, 0.0, self.qT); self.qAge = torch.where(anyNew, 0.0, self.qAge)
        self.qCharging = torch.where(anyNew, ns | npz, self.qCharging)
        self.qDir = torch.where(ns.unsqueeze(-1), kickDir, self.qDir); self.qSpin = torch.where(ns, 0.0, self.qSpin)
        okq = okq & ~anyNew

        # ---- carga com a bola dominada ----
        c = (self.chKind != CH_NONE) & okq
        self.chT = torch.where(c, self.chT + DT, self.chT)
        lost = c & ~canKick
        self.chKind = torch.where(lost, CH_NONE, self.chKind)
        c = c & ~lost
        cs = c & (self.chKind == CH_SHOT)
        cp = c & (self.chKind == CH_PASS)
        self.chSpin = torch.where(cs, (self.chSpin + inp['curve'] * CFG['SPIN_MAX'] * 0.08).clamp(-CFG['SPIN_MAX'], CFG['SPIN_MAX']), self.chSpin)
        # soltar fixa a força
        self.chRel = torch.where(cs & released('shoot') & (self.chRel < 0), self.chT, self.chRel)
        self.chRel = torch.where(cp & released('pas') & (self.chRel < 0), self.chT, self.chRel)
        doneS = cs & torch.where(self.chRel >= 0, self.chT >= CFG['SHOT_WINDUP'], self.chT >= CFG['CHARGE_MAX'])
        doneP = cp & torch.where(self.chRel >= 0, self.chT >= CFG['PASS_WINDUP'], self.chT >= CFG['PASS_CHARGE'])
        fake = cs & (self.chRel < 0) & (pressed('special') | pressed('pas'))
        powS = torch.where(self.chRel >= 0, self.chRel, self.chT) / CFG['CHARGE_MAX']
        powP = torch.where(self.chRel >= 0, self.chRel, self.chT) / CFG['PASS_CHARGE']
        self._shoot(doneS, self.chDir, powS.clamp(0, 1), self.chSpin, first=False)
        self._pass(doneP, inp, powP.clamp(0, 1))
        self.chKind = torch.where(doneS | doneP | fake, CH_NONE, self.chKind)
        self.armedShoot = self.armedShoot & ~doneS; self.armedPass = self.armedPass & ~doneP
        okc = okq & ~c   # sem carga em andamento
        # nova carga
        canKick = ((self.owner.view(B, 1) == torch.arange(P, device=d).view(1, P)) & ok)   # recomputa (chutes soltaram a bola)
        ns2 = okc & canKick & pressed('shoot')
        np2 = okc & canKick & ~ns2 & pressed('pas')
        self.chKind = torch.where(ns2, CH_SHOT, torch.where(np2, CH_PASS, self.chKind))
        self.chT = torch.where(ns2 | np2, 0.0, self.chT); self.chRel = torch.where(ns2 | np2, -1.0, self.chRel)
        self.chDir = torch.where(ns2.unsqueeze(-1), kickDir, self.chDir); self.chSpin = torch.where(ns2, 0.0, self.chSpin)
        self.armedShoot = self.armedShoot & ~ns2; self.armedPass = self.armedPass & ~np2
        self.dribbleLag = torch.where(ns2 | np2, 0.0, self.dribbleLag)
        okc = okc & ~(ns2 | np2)

        # ---- Espaço com a bola ----
        heldNow = canKick & self.held
        sp = okc & canKick & pressed('special')
        dropHands = sp & heldNow
        self.held = self.held & ~dropHands; self.holdT = torch.where(dropHands, 0.0, self.holdT)
        sp = sp & ~dropHands
        lagCancel = sp & (self.dribbleLag > 0)
        pushLag = lagCancel & (self.stance != ST_DRIB)
        self.dribbleLag = torch.where(pushLag, 0.0, self.dribbleLag)
        self._push(pushLag, strong=self.sprinting)
        sp = sp & ~lagCancel
        drib = sp & (self.stance == ST_DRIB)
        chainOk = (self.dribbleN == 1) & (self.dribbleChainT > 0)
        startDrib = drib & (chainOk | (self.cd['dribble'] <= 0)) & (self.stamina >= CFG['COST_DRIBBLE'] * 0.5)
        self._start_dribble(startDrib)
        self._push(sp & ~drib, strong=self.sprinting)
        # arrancada com a bola: empurra sozinho
        auto = okc & canKick & ~self.held & (self.effortT > 0) & self.moving & ~sp
        self._push(auto, strong=torch.ones_like(auto))
        okn = okc & ~canKick

        # ---- sem a bola: tackle / carrinho / dash / mergulho ----
        tk = okn & pressed('tackle')
        slide = tk & self.sprinting & self.moving & (self.cd['slide'] <= 0) & (self.stamina >= 6)
        tackle = tk & ~(self.sprinting & self.moving) & (self.cd['tackle'] <= 0)
        self._start_action(slide, ACT_SLIDE, self.moveDir, CFG['SLIDE_DUR'])
        self.cd['slide'] = torch.where(slide, torch.full_like(self.cd['slide'], CFG['SLIDE_CD']), self.cd['slide'])
        self.stamina = torch.where(slide, (self.stamina - CFG['COST_SLIDE']).clamp(min=0), self.stamina)
        self._start_action(tackle, ACT_TACKLE, self._lunge_dir(), CFG['TACKLE_DUR'])
        self.cd['tackle'] = torch.where(tackle, torch.full_like(self.cd['tackle'], CFG['TACKLE_CD']), self.cd['tackle'])
        self.stamina = torch.where(tackle, (self.stamina - CFG['COST_TACKLE']).clamp(min=0), self.stamina)
        spn = okn & ~tk & pressed('special')
        dash = spn & (self.stance == ST_DEF) & (self.cd['dash'] <= 0)
        self._start_action(dash, ACT_DASH, self._lunge_dir(), CFG['DASH_DUR'])
        self.cd['dash'] = torch.where(dash, torch.full_like(self.cd['dash'], CFG['DASH_CD']), self.cd['dash'])
        self.stamina = torch.where(dash, (self.stamina - CFG['COST_DASH']).clamp(min=0), self.stamina)
        dive = spn & (self.stance != ST_DEF) & self.isKeeper & self.in_own_box() & (self.cd['dive'] <= 0)
        self._start_action(dive, ACT_GKDIVE, self._lunge_dir(), CFG['GK_DIVE_DUR'])

    def _kick_dir(self, inp):
        dvec = inp['aim'] - self.bpos.view(self.B, 1, 2)
        n = dvec.norm(dim=-1, keepdim=True)
        return torch.where(n > 4, dvec / n.clamp(min=1e-6), self.facing)

    def _start_action(self, mask, kind, direction, dur):
        self.act = torch.where(mask, torch.full_like(self.act, kind), self.act)
        self.actT = torch.where(mask, 0.0, self.actT)
        self.actDur = torch.where(mask, torch.full_like(self.actDur, dur), self.actDur)
        self.actDir = torch.where(mask.unsqueeze(-1), direction, self.actDir)
        self.actHit = self.actHit & ~mask; self.actRolled = self.actRolled & ~mask; self.actBody = self.actBody & ~mask
        self.chKind = torch.where(mask, CH_NONE, self.chKind)

    def _start_dribble(self, mask):
        second = mask & (self.dribbleN == 1) & (self.dribbleChainT > 0)
        self._start_action(mask, ACT_DRIBBLE, self._lunge_dir(), CFG['DRIBBLE_DUR'])
        self.actBody = torch.where(mask, second, self.actBody)   # actBody reutilizado como "é o 2º drible"
        self.dribbleN = torch.where(mask, torch.where(second, 2, 1), self.dribbleN)
        self.dribbleChainT = torch.where(mask, 0.0, self.dribbleChainT)
        self.stamina = torch.where(mask, (self.stamina - CFG['COST_DRIBBLE']).clamp(min=0), self.stamina)

    # ------------------------------------------------------------ chutes
    def _release_ball(self, mask, vel, spin, cooldown):
        """mask [B,P]: quem chuta (no máximo um por partida, o dono). Bola sai da posição atual."""
        B, P = self.B, self.P
        any_ = mask.any(dim=1)
        who = mask.float().argmax(dim=1)
        self.owner = torch.where(any_, -1, self.owner)
        self.held = self.held & ~mask
        self.bvel = torch.where(any_.unsqueeze(-1), vel[torch.arange(B, device=self.dev), who], self.bvel)
        self.bspin = torch.where(any_, spin[torch.arange(B, device=self.dev), who], self.bspin)
        self.lastTouch = torch.where(any_, who, self.lastTouch)
        self.lastTeam = torch.where(any_, self.team[who], self.lastTeam)
        self.cd['grab'] = torch.where(mask, torch.full_like(self.cd['grab'], cooldown), self.cd['grab'])
        self.cd['through'] = torch.where(mask, torch.full_like(self.cd['through'], CFG['PASS_THROUGH']), self.cd['through'])
        self.lock = torch.where(any_, -1, self.lock)

    def _shoot(self, mask, direction, power, spin, first):
        if not mask.any():
            return
        mn = CFG['SHOT_MIN_FIRST'] if first else CFG['SHOT_MIN']
        speed = mn + (CFG['SHOT_MAX'] - mn) * power.clamp(0, 1) ** CFG['SHOT_CURVE']
        sp = spin * (1 - CFG['SPIN_POWER_FADE'] * power)
        vel = direction * speed.unsqueeze(-1) + self.vel * 0.1
        self._release_ball(mask, vel, sp, CFG['KICK_COOLDOWN'])
        self.recover = torch.where(mask, 0.1 + 0.2 * power, self.recover)
        self.events['shot'] = self.events.get('shot', torch.zeros_like(mask)) | mask

    def _pass(self, mask, inp, power):
        """passe com assistência: companheiro a até PASS_ASSIST_DEG da mira recebe força ajustada"""
        if not mask.any():
            return
        B, P, d = self.B, self.P, self.dev
        direction = self._kick_dir(inp)
        speed = CFG['PASS_MIN'] + (CFG['PASS_MAX'] - CFG['PASS_MIN']) * power
        # assistência: melhor companheiro dentro do cone
        rel = self.pos.view(B, 1, P, 2) - self.bpos.view(B, 1, 1, 2)          # [B,p,m,2] vetor bola->m
        dist = rel.norm(dim=-1)
        cosang = (rel * direction.view(B, P, 1, 2)).sum(-1) / dist.clamp(min=1e-6)
        same = (self.team.view(1, P, 1) == self.team.view(1, 1, P)) & ~torch.eye(P, dtype=torch.bool, device=d).view(1, P, P)
        okm = same & (dist >= 60) & (cosang > math.cos(math.radians(CFG['PASS_ASSIST_DEG'])))
        score = torch.where(okm, cosang, torch.full_like(cosang, -2.0))
        best = score.argmax(dim=-1)                                            # [B,P]
        has = score.max(dim=-1).values > -1.5
        tgt = torch.gather(self.pos, 1, best.unsqueeze(-1).expand(B, P, 2)) + 0.3 * torch.gather(self.vel, 1, best.unsqueeze(-1).expand(B, P, 2))
        toT = tgt - self.bpos.view(B, 1, 2)
        dT = toT.norm(dim=-1)
        dirA = toT / dT.clamp(min=1e-6).unsqueeze(-1)
        spA = self._speed_for_distance(dT + 30).clamp(CFG['PASS_MIN'] * 0.8, CFG['PASS_MAX'])
        direction = torch.where(has.unsqueeze(-1), dirA, direction)
        speed = torch.where(has, spA, speed)
        vel = direction * speed.unsqueeze(-1) + self.vel * 0.1
        self._release_ball(mask, vel, torch.zeros_like(speed), CFG['KICK_COOLDOWN'])
        self.events['pass'] = self.events.get('pass', torch.zeros_like(mask)) | mask

    @staticmethod
    def _speed_for_distance(dist):
        """aprox. fechada da distância percorrida: v0 ~ solve via tabela"""
        # tabela pré-calculada de travel(v) (mesma integração do JS)
        if not hasattr(TorchSim, '_tbl'):
            vs = torch.arange(150, CFG['BALL_MAX'] + 1, 10.0)
            tr = []
            for v in vs.tolist():
                s, x, dt = v, 0.0, 1 / 30
                t = 0.0
                while t < 5 and s > 1:
                    s = max(0.0, s * math.exp(-CFG['BALL_DRAG'] * dt) - CFG['BALL_DECEL'] * dt)
                    x += s * dt
                    t += dt
                tr.append(x)
            TorchSim._tbl = (vs, torch.tensor(tr))
        vs, tr = TorchSim._tbl
        vs, tr = vs.to(dist.device), tr.to(dist.device)
        idx = torch.searchsorted(tr, dist.reshape(-1).contiguous()).clamp(max=len(vs) - 1)
        return vs[idx].reshape(dist.shape)

    def _push(self, mask, strong):
        if not mask.any():
            return
        direction = self._lunge_dir()
        base = torch.where(strong, CFG['PUSH_SPEED'], CFG['PUSH_WALK_SPEED']) * torch.ones_like(self.stamina)
        speed = torch.maximum(self.vel.norm(dim=-1) + 60, base)
        vel = direction * speed.unsqueeze(-1)
        self._release_ball(mask, vel, torch.zeros_like(speed), CFG['PUSH_COOLDOWN'])
        self.pushFlash = torch.where(mask, torch.full_like(self.pushFlash, 0.35), self.pushFlash)
        self.events['push'] = self.events.get('push', torch.zeros_like(mask)) | mask

    def _fire_queued(self, mask, inp):
        """toque de primeira no contato"""
        if not mask.any():
            return
        s = mask & (self.qKind == Q_SHOT)
        p = mask & (self.qKind == Q_PASS)
        u = mask & (self.qKind == Q_PUSH)
        self.qKind = torch.where(mask, Q_NONE, self.qKind)
        self.lock = torch.where(mask.any(dim=1), -1, self.lock)
        self._shoot(s, self.qDir, (self.qT / CFG['CHARGE_MAX']).clamp(0, 1), self.qSpin, first=True)
        self._pass(p, inp, (self.qT / CFG['PASS_CHARGE']).clamp(0, 1))
        self._push(u, strong=self.sprinting)
        self.events['first'] = self.events.get('first', torch.zeros_like(mask)) | mask

    # ------------------------------------------------------------ movimento
    def _move(self, inp, mask, hasBall, held):
        base = torch.full_like(self.stamina, CFG['SPEED'])
        base = torch.where(self.sprinting & self.moving, torch.full_like(base, CFG['SPRINT']), base)
        base = torch.where((self.effortT > 0) & self.moving, torch.full_like(base, CFG['SPRINT'] * CFG['EXTRA_EFFORT']), base)
        base = torch.where(self.stance == ST_DEF, torch.full_like(base, CFG['SPEED'] * CFG['MUL_DEF']), base)
        wb = torch.where(self.sprinting & self.moving & (self.stance != ST_DRIB), CFG['SPRINT_BALL'], CFG['SPEED_BALL']) * torch.ones_like(base)
        wb = torch.where(self.stance == ST_DRIB, wb * CFG['MUL_DRIBBLE'], wb)
        base = torch.where(hasBall, wb, base)
        base = torch.where(held, torch.full_like(base, CFG['KEEPER_HOLD_SPEED']), base)
        base = torch.where(self.chKind == CH_SHOT, base * CFG['MUL_CHARGE'], base)
        base = torch.where(self.chKind == CH_PASS, base * CFG['MUL_CHARGE_PASS'], base)
        base = torch.where(self.recover > 0, base * CFG['MUL_RECOVER'], base)
        base = torch.where(self.exhausted & (self.effortT <= 0), base * CFG['MUL_EXHAUSTED'], base)
        base = torch.where(self.dribbleLag > 0, base * CFG['DRIBBLE_LAG_MUL'], base)
        desired = torch.stack([inp['mx'], inp['my']], -1) * base.unsqueeze(-1)
        # snap da trava: vai até a bola na velocidade permitida
        q = self.qKind != Q_NONE
        toBall = self.bpos.view(self.B, 1, 2) - self.pos
        snap = toBall / toBall.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self._snap_cap().unsqueeze(-1)
        vel_snap = torch.where((toBall.norm(dim=-1) > 2).unsqueeze(-1), snap, torch.zeros_like(snap))
        diff = desired - self.vel
        dl = diff.norm(dim=-1, keepdim=True)
        acc = torch.where(desired.norm(dim=-1, keepdim=True) > self.vel.norm(dim=-1, keepdim=True), CFG['ACCEL'], CFG['DECEL']) * DT
        newv = torch.where(dl <= acc, desired, self.vel + diff * (acc / dl.clamp(min=1e-6)))
        newv = torch.where(q.unsqueeze(-1), vel_snap, newv)
        self.vel = torch.where(mask.unsqueeze(-1), newv, self.vel)

    # ------------------------------------------------------------ ações em andamento
    def _run_actions(self, inAct):
        B, P, d = self.B, self.P, self.dev
        self.actT = torch.where(inAct, self.actT + DT, self.actT)
        k = (1 - self.actT / self.actDur.clamp(min=1e-6)).clamp(min=0)
        sp = torch.zeros_like(self.stamina)
        sp = torch.where(self.act == ACT_TACKLE, CFG['TACKLE_SPEED'] * k, sp)
        sp = torch.where(self.act == ACT_SLIDE, CFG['SLIDE_SPEED'] * k, sp)
        sp = torch.where(self.act == ACT_DASH, torch.full_like(sp, CFG['SPEED'] * CFG['MUL_DEF'] * CFG['DASH_MUL']), sp)
        sp = torch.where(self.act == ACT_DRIBBLE, torch.full_like(sp, CFG['SPEED_BALL'] * CFG['DRIBBLE_MUL']), sp)
        sp = torch.where(self.act == ACT_GKDIVE, CFG['GK_DIVE_SPEED'] * k, sp)
        self.vel = torch.where(inAct.unsqueeze(-1), self.actDir * sp.unsqueeze(-1), self.vel)
        self._tackle_hit(inAct & (self.act == ACT_TACKLE))
        self._slide_hit(inAct & (self.act == ACT_SLIDE))
        self._dive_hit(inAct & (self.act == ACT_GKDIVE))
        done = inAct & (self.actT >= self.actDur)
        # fim
        tk = done & (self.act == ACT_TACKLE) & ~self.actHit
        self.recover = torch.where(tk, torch.full_like(self.recover, CFG['TACKLE_MISS_RECOVER']), self.recover)
        sl = done & (self.act == ACT_SLIDE)
        hit = self.actHit | self.actBody
        self.getup = torch.where(sl, torch.where(hit, CFG['SLIDE_RECOVER'], CFG['SLIDE_MISS_RECOVER']) * torch.ones_like(self.getup), self.getup)
        self.stamina = torch.where(sl & ~hit, (self.stamina - CFG['COST_SLIDE_MISS']).clamp(min=0), self.stamina)
        dv = done & (self.act == ACT_GKDIVE)
        self.getup = torch.where(dv, torch.full_like(self.getup, CFG['GK_DIVE_RECOVER']), self.getup)
        self.cd['dive'] = torch.where(dv, torch.full_like(self.cd['dive'], CFG['GK_DIVE_CD']), self.cd['dive'])
        dr = done & (self.act == ACT_DRIBBLE)
        second = dr & self.actBody
        self.dribbleN = torch.where(second, 0, self.dribbleN)
        self.dribbleLag = torch.where(second, torch.full_like(self.dribbleLag, CFG['DRIBBLE_LAG']), self.dribbleLag)
        self.cd['dribble'] = torch.where(second, torch.full_like(self.cd['dribble'], CFG['DRIBBLE_CD']), self.cd['dribble'])
        self.dribbleChainT = torch.where(dr & ~second, torch.full_like(self.dribbleChainT, CFG['DRIBBLE_CHAIN_WINDOW']), self.dribbleChainT)
        self.act = torch.where(done, ACT_NONE, self.act)

    def _owner_info(self):
        """dono da bola por partida: team [B], held [B], immune [B], stance drib [B]"""
        B = self.B
        bi = torch.arange(B, device=self.dev)
        has = self.owner >= 0
        o = self.owner.clamp(min=0)
        oteam = torch.where(has, self.team[o], torch.full_like(self.owner, -1))
        oheld = has & self.held[bi, o]
        oact = self.act[bi, o]
        immune = has & ((oact == ACT_DRIBBLE) | (oact == ACT_DASH))
        odrib = has & (self.stance[bi, o] == ST_DRIB)
        return has, o, oteam, oheld, immune, odrib

    def _take(self, mask, hands):
        """mask [B,P] com no máximo um True por partida"""
        B, P = self.B, self.P
        any_ = mask.any(dim=1)
        who = mask.float().argmax(dim=1)
        self.owner = torch.where(any_, who, self.owner)
        self.bspin = torch.where(any_, 0.0, self.bspin)
        self.lastTouch = torch.where(any_, who, self.lastTouch)
        self.lastTeam = torch.where(any_, self.team[who], self.lastTeam)
        h = hands if torch.is_tensor(hands) else torch.full_like(mask, bool(hands))
        self.held = torch.where(mask, h, self.held)
        self.holdT = torch.where(mask, 0.0, self.holdT)

    def _tackle_hit(self, m):
        if not m.any():
            return
        B, P = self.B, self.P
        center = self.pos + self.actDir * (self.r.view(1, P) * 0.5).unsqueeze(-1)
        near = (center - self.bpos.view(B, 1, 2)).norm(dim=-1) < self.r.view(1, P) + CFG['TACKLE_REACH'] + CFG['BALL_R']
        m = m & ~self.actRolled & near
        has, o, oteam, oheld, immune, odrib = self._owner_info()
        loose = ~has
        # bola solta: pega (um por partida: mais perto)
        takeL = m & loose.view(B, 1)
        takeL = self._one_per_match(takeL)
        self._take(takeL, False)
        self.actHit = self.actHit | takeL; self.actRolled = self.actRolled | takeL
        # bola do adversário (não nas mãos)
        opp = m & has.view(B, 1) & (self.team.view(1, P) != oteam.view(B, 1)) & ~oheld.view(B, 1)
        opp = self._one_per_match(opp)
        chance = torch.where(odrib.view(B, 1), CFG['TACKLE_CHANCE_DRIBBLE'], CFG['TACKLE_CHANCE'])
        win = opp & ~immune.view(B, 1) & (self.rand(B, P) < chance)
        self.actRolled = self.actRolled | opp
        # vítima
        bi = torch.arange(B, device=self.dev)
        vict = win.any(dim=1)
        self.cd['grab'][bi[vict], o[vict]] = 0.45
        self.recover[bi[vict], o[vict]] = torch.maximum(self.recover[bi[vict], o[vict]], torch.full_like(self.recover[bi[vict], o[vict]], 0.35))
        self._take(win, False)
        self.actHit = self.actHit | win
        self.events['steal'] = self.events.get('steal', torch.zeros_like(win)) | win

    def _one_per_match(self, m):
        """mantém só o jogador mais perto da bola entre os True de cada partida"""
        if not m.any():
            return m
        dist = (self.pos - self.bpos.view(self.B, 1, 2)).norm(dim=-1)
        dist = torch.where(m, dist, torch.full_like(dist, 1e9))
        best = dist.argmin(dim=1)
        out = torch.zeros_like(m)
        out[torch.arange(self.B, device=self.dev), best] = True
        return out & m

    def _slide_hit(self, m):
        if not m.any():
            return
        B, P = self.B, self.P
        center = self.pos + self.actDir * (self.r.view(1, P) * 0.5).unsqueeze(-1)
        near = (center - self.bpos.view(B, 1, 2)).norm(dim=-1) < self.r.view(1, P) + CFG['SLIDE_REACH'] + CFG['BALL_R']
        has, o, oteam, oheld, immune, odrib = self._owner_info()
        stealable = (~has.view(B, 1) | ((self.team.view(1, P) != oteam.view(B, 1)) & ~oheld.view(B, 1)))
        hit = self._one_per_match(m & ~self.actHit & near & stealable)
        if hit.any():
            bi = torch.arange(B, device=self.dev)
            hb = hit.any(dim=1)
            self.cd['grab'][bi[hb & has], o[hb & has]] = 0.35
            vel = self.actDir * CFG['SLIDE_KNOCK'] + self.vel * 0.2
            self._release_ball(hit, vel, torch.zeros_like(self.stamina), 0.3)
            self.actHit = self.actHit | hit
            self.events['slide'] = self.events.get('slide', torch.zeros_like(hit)) | hit
        # derruba o portador (só ele) em contato de corpo
        has, o, oteam, oheld, immune, odrib = self._owner_info()
        bi = torch.arange(B, device=self.dev)
        opos = self.pos[bi, o.clamp(min=0)]
        touch = m & has.view(B, 1) & (self.team.view(1, P) != oteam.view(B, 1)) & ~oheld.view(B, 1) & \
            ((self.pos - opos.view(B, 1, 2)).norm(dim=-1) < self.r.view(1, P) * 2 + 2)
        touch = self._one_per_match(touch)
        if touch.any():
            hb = touch.any(dim=1)
            vict_b = bi[hb]; vict_p = o[hb]
            self.fallen[vict_b, vict_p] = CFG['FALL_DUR']
            self.act[vict_b, vict_p] = ACT_NONE; self.chKind[vict_b, vict_p] = CH_NONE
            self.cd['grab'][vict_b, vict_p] = 0.5
            self.actBody = self.actBody | touch
            slider = touch.float().argmax(dim=1)
            vel = self.actDir[vict_b, slider[hb]] * 250 + self.vel[vict_b, vict_p] * 0.5
            self.owner[vict_b] = -1
            self.bvel[vict_b] = vel
            self.events['knockdown'] = self.events.get('knockdown', torch.zeros_like(touch)) | touch

    def _dive_hit(self, m):
        if not m.any():
            return
        B, P = self.B, self.P
        near = (self.pos - self.bpos.view(B, 1, 2)).norm(dim=-1) < self.r.view(1, P) + CFG['GK_DIVE_REACH'] + CFG['BALL_R']
        has, o, oteam, oheld, immune, odrib = self._owner_info()
        okm = m & ~self.actHit & near & (~has.view(B, 1) | ((self.team.view(1, P) != oteam.view(B, 1)) & ~oheld.view(B, 1)))
        okm = self._one_per_match(okm)
        if not okm.any():
            return
        self.actHit = self.actHit | okm
        speed = self.bvel.norm(dim=-1).view(B, 1)
        hands = okm & self.in_own_box() & (speed < CFG['GK_PARRY_SPEED']) & (self.lastTeam.view(B, 1) != self.team.view(1, P))
        feet = okm & self.in_own_box() & (speed < CFG['GK_PARRY_SPEED']) & ~hands
        bi = torch.arange(B, device=self.dev)
        hb = okm.any(dim=1) & has
        self.cd['grab'][bi[hb], o[hb]] = 0.4
        self._take(hands | feet, hands)
        parry = okm & ~hands & ~feet
        if parry.any():
            n = self.bpos.view(B, 1, 2) - self.pos
            n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            refl = self.bvel.view(B, 1, 2) - 2 * (self.bvel.view(B, 1, 2) * n).sum(-1, keepdim=True) * n
            vel = refl * 0.45 + self.actDir * 260
            self._release_ball(parry, vel, torch.zeros_like(self.stamina), 0.3)
        self.events['save'] = self.events.get('save', torch.zeros_like(okm)) | hands

    # ------------------------------------------------------------ bola
    def _update_ball(self):
        B, P = self.B, self.P
        self.bprev = self.bpos.clone()
        has = self.owner >= 0
        bi = torch.arange(B, device=self.dev)
        o = self.owner.clamp(min=0)
        opos, ovel = self.pos[bi, o], self.vel[bi, o]
        oheld = self.held[bi, o]
        # nas mãos
        handsPos = opos + self.facing[bi, o] * (self.r[o] * 0.45).unsqueeze(-1)
        # nos pés: condução física
        dirc = torch.where(self.moving[bi, o].unsqueeze(-1), self.moveDir[bi, o], self.facing[bi, o])
        dist = self.r[o] + CFG['BALL_R'] + CFG['CARRY_DIST']
        drib = self.stance[bi, o] == ST_DRIB
        # proteger do adversário mais próximo (postura de drible)
        oppmask = self.team.view(1, P) != self.team[o].view(B, 1)
        dopp = torch.where(oppmask, (self.pos - opos.view(B, 1, 2)).norm(dim=-1), torch.full_like(self.stamina, 1e9))
        nd, ni = dopp.min(dim=1)
        away = opos - self.pos[bi, ni]
        away = away / away.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        shield = drib & (nd < 140)
        dirc = torch.where(shield.unsqueeze(-1), dirc * 0.4 + away * 0.6, dirc)
        dirc = dirc / dirc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dist = torch.where(drib, dist - 2, dist)
        target = opos + dirc * dist.unsqueeze(-1)
        rel = (target - self.bpos) * CFG['CARRY_K']
        rl = rel.norm(dim=-1, keepdim=True)
        rel = torch.where(rl > CFG['CARRY_MAX_REL'], rel * (CFG['CARRY_MAX_REL'] / rl.clamp(min=1e-6)), rel)
        k = min(1.0, CFG['CARRY_LERP'] * DT)
        nv = self.bvel + (ovel + rel - self.bvel) * k
        npos = self.bpos + nv * DT
        dd = (npos - opos).norm(dim=-1, keepdim=True)
        minD = (self.r[o] + CFG['BALL_R']).unsqueeze(-1)
        n = torch.where(dd > 1e-6, (npos - opos) / dd.clamp(min=1e-6), dirc)
        npos = torch.where(dd < minD, opos + n * minD, npos)
        npos = self.clamp_arena(npos, CFG['BALL_R'])
        lost = has & ~oheld & ((npos - opos).norm(dim=-1) > self.r[o] + CFG['BALL_R'] + CFG['CARRY_LOSE'])
        feetPos = npos; feetVel = nv
        # livre
        fpos, fvel, fspin = self._integrate_free(self.bpos, self.bvel, self.bspin, DT, walls=True)
        self.bpos = torch.where(has.unsqueeze(-1), torch.where(oheld.unsqueeze(-1), handsPos, feetPos), fpos)
        self.bvel = torch.where(has.unsqueeze(-1), torch.where(oheld.unsqueeze(-1), ovel, feetVel), fvel)
        self.bspin = torch.where(has, torch.zeros_like(self.bspin), fspin)
        self.owner = torch.where(lost, -1, self.owner)
        self.cd['grab'][bi[lost], o[lost]] = 0.1

    def _integrate_free(self, pos, vel, spin, dt, walls):
        s = vel.norm(dim=-1)
        ang = spin * dt * (s / 500).clamp(max=1)
        c, si = torch.cos(ang), torch.sin(ang)
        vel = torch.stack([vel[:, 0] * c - vel[:, 1] * si, vel[:, 0] * si + vel[:, 1] * c], -1)
        spin = spin * math.exp(-CFG['SPIN_DECAY'] * dt)
        spin = torch.where(spin.abs() < 0.01, torch.zeros_like(spin), spin)
        s = vel.norm(dim=-1)
        vel = torch.where((s > CFG['BALL_MAX']).unsqueeze(-1), vel * (CFG['BALL_MAX'] / s.clamp(min=1e-6)).unsqueeze(-1), vel)
        s = vel.norm(dim=-1)
        s2 = (s * math.exp(-CFG['BALL_DRAG'] * dt) - CFG['BALL_DECEL'] * dt).clamp(min=0)
        vel = torch.where((s > 0).unsqueeze(-1), vel * (s2 / s.clamp(min=1e-6)).unsqueeze(-1), vel)
        pos = pos + vel * dt
        if walls:
            pos, vel = self._walls(pos, vel)
        return pos, vel, spin

    @staticmethod
    def _walls(pos, vel):
        W2, H2, r, Bc = CFG['FIELD_W'] / 2, CFG['FIELD_H'] / 2, CFG['BALL_R'], CFG['BALL_BOUNCE']
        x, y, vx, vy = pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1]
        mouth = y.abs() < CFG['GOAL_W'] / 2 - r
        # esquerda
        hitL = (x - r < -W2) & ~mouth & (x > -W2)
        x = torch.where(hitL, torch.full_like(x, -W2 + r), x); vx = torch.where(hitL & (vx < 0), -vx * Bc, vx)
        netL = (x - r < -W2 - CFG['GOAL_D'])
        x = torch.where(netL, torch.full_like(x, -W2 - CFG['GOAL_D'] + r), x); vx = torch.where(netL & (vx < 0), -vx * Bc, vx)
        hitR = (x + r > W2) & ~mouth & (x < W2)
        x = torch.where(hitR, torch.full_like(x, W2 - r), x); vx = torch.where(hitR & (vx > 0), -vx * Bc, vx)
        netR = (x + r > W2 + CFG['GOAL_D'])
        x = torch.where(netR, torch.full_like(x, W2 + CFG['GOAL_D'] - r), x); vx = torch.where(netR & (vx > 0), -vx * Bc, vx)
        inGoal = x.abs() > W2
        lim = CFG['GOAL_W'] / 2 - r
        hy = torch.where(inGoal, y > lim, y + r > H2)
        ly = torch.where(inGoal, y < -lim, y - r < -H2)
        y = torch.where(hy, torch.where(inGoal, torch.full_like(y, lim), torch.full_like(y, H2 - r)), y)
        y = torch.where(ly, torch.where(inGoal, torch.full_like(y, -lim), torch.full_like(y, -H2 + r)), y)
        vy = torch.where(hy & (vy > 0), -vy * Bc, vy); vy = torch.where(ly & (vy < 0), -vy * Bc, vy)
        pos = torch.stack([x, y], -1); vel = torch.stack([vx, vy], -1)
        # traves
        for sx in (-1, 1):
            for sy in (-1, 1):
                post = torch.tensor([sx * W2, sy * CFG['GOAL_W'] / 2], device=pos.device)
                dv = pos - post
                dd = dv.norm(dim=-1)
                mn = r + CFG['POST_R']
                hit = (dd < mn) & (dd > 1e-6)
                n = dv / dd.clamp(min=1e-6).unsqueeze(-1)
                pos = torch.where(hit.unsqueeze(-1), post + n * mn, pos)
                vn = (vel * n).sum(-1)
                refl = vel - 2 * vn.unsqueeze(-1) * n
                vel = torch.where((hit & (vn < 0)).unsqueeze(-1), refl * Bc, vel)
        return pos, vel

    def _walls_free(self, loose):
        pos, vel = self._walls(self.bpos, self.bvel)
        self.bpos = torch.where(loose.unsqueeze(-1), pos, self.bpos)
        self.bvel = torch.where(loose.unsqueeze(-1), vel, self.bvel)

    # ------------------------------------------------------------ prioridade e contatos
    def _resolve_locks(self):
        B, P = self.B, self.P
        loose = self.owner < 0
        cand = (self.qKind != Q_NONE) & self.reach & loose.view(B, 1)
        if not cand.any():
            self.lock = torch.where(loose, -1, self.lock)
            return
        toBall = self.bpos.view(B, 1, 2) - self.pos
        dist = toBall.norm(dim=-1)
        approach = (self.vel * toBall).sum(-1) / dist.clamp(min=1e-6)
        score = dist - approach * 0.2
        score = torch.where(self.isKeeper & self.in_own_box(), score - 40, score)
        pidx = torch.arange(P, device=self.dev).view(1, P)
        score = torch.where(self.lock.view(B, 1) == pidx, score - 8, score)
        score = torch.where(cand, score, torch.full_like(score, 1e9))
        best = score.argmin(dim=1)
        hasC = cand.any(dim=1)
        losers = cand & (pidx != best.view(B, 1))
        self.qKind = torch.where(losers, Q_NONE, self.qKind)
        self.recover = torch.where(losers, torch.maximum(self.recover, torch.full_like(self.recover, CFG['LOCK_LOSE_RECOVER'])), self.recover)
        self.lock = torch.where(hasC, best, torch.where(loose, -1, self.lock))

    def _contacts(self, play):
        B, P = self.B, self.P
        pidx = torch.arange(P, device=self.dev).view(1, P)
        has, o, oteam, oheld, immune, odrib = self._owner_info()
        notOwner = (self.owner.view(B, 1) != pidx) & play.view(B, 1)
        d = (self.pos - self.bpos.view(B, 1, 2)).norm(dim=-1)
        hb = self._ball_hitbox()
        R = hb + CFG['BALL_R']
        free = (self.fallen <= 0) & (self.getup <= 0) & (self.cd['grab'] <= 0)
        # bola dominada por outro: corpo só desvia (postura defensiva: solta a bola do adversário)
        owned = notOwner & has.view(B, 1) & ~oheld.view(B, 1) & (d < R)
        n = (self.bpos.view(B, 1, 2) - self.pos)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        blockers = self._one_per_match(owned & (self.stance == ST_DEF) & (self.team.view(1, P) != oteam.view(B, 1)) & free & (self.act == ACT_NONE) & ~immune.view(B, 1) & ~odrib.view(B, 1))
        if blockers.any():
            bi = torch.arange(B, device=self.dev)
            bb = blockers.any(dim=1)
            self.cd['grab'][bi[bb], o[bb]] = 0.35
            who = blockers.float().argmax(dim=1)
            self.owner = torch.where(bb, -1, self.owner)
            self.bvel = torch.where(bb.unsqueeze(-1), n[bi, who] * 60 + self.vel[bi, o] * 0.5, self.bvel)
            self.lastTouch = torch.where(bb, who, self.lastTouch); self.lastTeam = torch.where(bb, self.team[who], self.lastTeam)
        pushOut = owned & ~blockers
        if pushOut.any():
            pb = self._one_per_match(pushOut)
            bi = torch.arange(B, device=self.dev)
            who = pb.float().argmax(dim=1)
            anyb = pb.any(dim=1)
            newp = self.pos[bi, who] + n[bi, who] * (hb[bi, who] + CFG['BALL_R'] + 0.5).unsqueeze(-1)
            self.bpos = torch.where(anyb.unsqueeze(-1), newp, self.bpos)
        # bola solta
        has = self.owner >= 0
        loose = ~has
        through = (self.lastTouch.view(B, 1) == pidx) & (self.cd['through'] > 0)
        cand = notOwner & loose.view(B, 1) & ~through
        # trajeto no tick para o toque travado
        seg = self._seg_dist(self.pos, self.bprev, self.bpos)
        canQ = cand & (self.qKind != Q_NONE) & (self.fallen <= 0) & (self.getup <= 0) & ((self.act == ACT_NONE) | (self.act == ACT_DASH)) & (seg < R)
        canQ = self._one_per_match(canQ)
        self._fire_queued(canQ, self.prev_inp_for_kick())
        has = self.owner >= 0; loose = ~has
        cand = cand & loose.view(B, 1) & (d < R) & ~canQ
        speed = self.bvel.norm(dim=-1).view(B, 1)
        # outro tem a trava: só desvia
        lockOther = (self.lock.view(B, 1) >= 0) & (self.lock.view(B, 1) != pidx)
        canGrab = cand & free & ((self.act == ACT_NONE) | (self.act == ACT_DASH)) & ~lockOther
        hands = self.isKeeper & self.in_own_box() & (self.lastTeam.view(B, 1) != self.team.view(1, P))
        limit = torch.where(hands, torch.where(self.stance == ST_DEF, float('inf'), CFG['GK_PARRY_SPEED']),
                            torch.where(self.stance == ST_DEF, CFG['CONTROL_MAX_DEF'], CFG['CONTROL_MAX'])) * torch.ones_like(d)
        take = self._one_per_match(canGrab & (speed <= limit))
        self._take(take, hands & take)
        self.events['control'] = self.events.get('control', torch.zeros_like(take)) | take
        # rebate no corpo
        deflect = cand & ~take & ~take.any(dim=1).view(B, 1)
        if deflect.any():
            db = self._one_per_match(deflect)
            bi = torch.arange(B, device=self.dev)
            who = db.float().argmax(dim=1)
            anyb = db.any(dim=1)
            nn = n[bi, who]
            newp = self.pos[bi, who] + nn * (hb[bi, who] + CFG['BALL_R'] + 0.5).unsqueeze(-1)
            rel = self.bvel - self.vel[bi, who]
            vn = (rel * nn).sum(-1)
            refl = rel - 2 * vn.unsqueeze(-1) * nn
            newv = torch.where((vn < 0).unsqueeze(-1), refl * CFG['DEFLECT_BOUNCE'] + self.vel[bi, who], self.bvel)
            self.bpos = torch.where(anyb.unsqueeze(-1), newp, self.bpos)
            self.bvel = torch.where(anyb.unsqueeze(-1), newv, self.bvel)
            self.bspin = torch.where(anyb, self.bspin * 0.3, self.bspin)
            self.lastTouch = torch.where(anyb, who, self.lastTouch); self.lastTeam = torch.where(anyb, self.team[who], self.lastTeam)
            grabbers = db & canGrab
            self.cd['grab'] = torch.where(grabbers, torch.full_like(self.cd['grab'], CFG['DEFLECT_COOLDOWN']), self.cd['grab'])

    def prev_inp_for_kick(self):
        return self._last_inp

    @staticmethod
    def _seg_dist(p, a, b):
        """distância de p [B,P,2] ao segmento a->b [B,2]"""
        a = a.view(-1, 1, 2); b = b.view(-1, 1, 2)
        ab = b - a
        t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)
        t = t.clamp(0, 1)
        proj = a + ab * t.unsqueeze(-1)
        return (p - proj).norm(dim=-1)

    # ------------------------------------------------------------ colisões, goleiro
    def _collide_players(self):
        B, P = self.B, self.P
        dv = self.pos.view(B, P, 1, 2) - self.pos.view(B, 1, P, 2)   # i - j
        dd = dv.norm(dim=-1)
        minD = self.r.view(1, P, 1) + self.r.view(1, 1, P)
        eye = torch.eye(P, dtype=torch.bool, device=self.dev).view(1, P, P)
        hit = (dd < minD) & (dd > 1e-6) & ~eye
        n = dv / dd.clamp(min=1e-6).unsqueeze(-1)
        push = ((minD - dd) / 2).clamp(min=0) * hit
        self.pos = self.pos + (n * push.unsqueeze(-1)).sum(dim=2)
        relv = ((self.vel.view(B, 1, P, 2) - self.vel.view(B, P, 1, 2)) * n).sum(-1)   # vj - vi ao longo de n (n aponta de j para i)
        # JS: n = norm(c - a) (a=i, c=j) => n_js = -n; rel = dot(vj - vi, n_js) < 0 -> a.vel += n_js*rel*0.5
        rel_js = -relv
        adj = (rel_js < 0) & hit
        self.vel = self.vel + ((-n) * (rel_js * 0.5 * adj).unsqueeze(-1)).sum(dim=2)

    def _keeper_repel(self):
        B, P = self.B, self.P
        pidx = torch.arange(P, device=self.dev).view(1, P)
        k = self.isKeeper & self.held & (self.owner.view(B, 1) == pidx)
        if not k.any():
            return
        bi = torch.arange(B, device=self.dev)
        kb = k.any(dim=1); kp = k.float().argmax(dim=1)
        kpos = self.pos[bi, kp]
        R = self.r[kp] + CFG['GK_REPEL']
        opp = kb.view(B, 1) & (self.team.view(1, P) != self.team[kp].view(B, 1))
        dv = self.pos - kpos.view(B, 1, 2)
        dd = dv.norm(dim=-1)
        inside = opp & (dd < R.view(B, 1) + self.r.view(1, P))
        n = dv / dd.clamp(min=1e-6).unsqueeze(-1)
        gap = (R.view(B, 1) + self.r.view(1, P) - dd).clamp(min=0)
        mv = torch.minimum(gap, torch.full_like(gap, CFG['GK_REPEL_PUSH'] * DT))
        self.pos = self.pos + torch.where(inside.unsqueeze(-1), n * mv.unsqueeze(-1), torch.zeros_like(self.pos))
        inward = (self.vel * n).sum(-1)
        self.vel = self.vel - torch.where((inside & (inward < 0)).unsqueeze(-1), n * inward.unsqueeze(-1), torch.zeros_like(self.vel))
        stop = inside & ((self.act == ACT_TACKLE) | (self.act == ACT_SLIDE))
        self.act = torch.where(stop, ACT_NONE, self.act); self.getup = torch.where(stop, torch.full_like(self.getup, 0.3), self.getup)

    def _clamp_keeper_box(self):
        B, P = self.B, self.P
        pidx = torch.arange(P, device=self.dev).view(1, P)
        k = self.held & (self.owner.view(B, 1) == pidx)
        if not k.any():
            return
        W2 = CFG['FIELD_W'] / 2
        r = self.r.view(1, P)
        lo = torch.where(self.team.view(1, P) == 0, -W2 + r, W2 - CFG['BOX_W'] + r)
        hi = torch.where(self.team.view(1, P) == 0, -W2 + CFG['BOX_W'] - r, W2 - r)
        x = self.pos[..., 0].clamp(min=lo, max=hi) if False else torch.minimum(torch.maximum(self.pos[..., 0], lo), hi)
        y = self.pos[..., 1].clamp(-CFG['BOX_H'] / 2 + CFG['PLAYER_R'], CFG['BOX_H'] / 2 - CFG['PLAYER_R'])
        self.pos = torch.where(k.unsqueeze(-1), torch.stack([x, y], -1), self.pos)

    def _gloves(self, play):
        B, P = self.B, self.P
        inBox = self.in_own_box() & self.active_mask() & (self.fallen <= 0) & play.view(B, 1)
        gx = torch.where(self.team.view(1, P) == 0, -CFG['FIELD_W'] / 2, CFG['FIELD_W'] / 2)
        dist = (self.pos[..., 0] - gx).abs()
        for t in (0, 1):
            tm = self.team.view(1, P) == t
            keeper = self.isKeeper & tm
            hasK = keeper.any(dim=1)
            kp = keeper.float().argmax(dim=1)
            bi = torch.arange(B, device=self.dev)
            kOut = hasK & ~self.in_own_box()[bi, kp] & (self.cd['gloves'][bi, kp] <= 0)
            candM = inBox & tm & ~self.isKeeper
            candD = torch.where(candM, dist, torch.full_like(dist, 1e9))
            best = candD.argmin(dim=1)
            hasC = candM.any(dim=1)
            assign = hasC & (~hasK | kOut)
            # tira as luvas do antigo
            old = assign & hasK
            self.isKeeper[bi[old], kp[old]] = False
            self.held[bi[old], kp[old]] = False
            self.isKeeper[bi[assign], best[assign]] = True
            self.cd['gloves'][bi[assign], best[assign]] = 1.5

    def active_mask(self):
        return torch.ones(self.B, self.P, dtype=torch.bool, device=self.dev)

    def _check_goal(self, play):
        B = self.B
        W2 = CFG['FIELD_W'] / 2
        inMouth = self.bpos[:, 1].abs() < CFG['GOAL_W'] / 2
        g1 = play & inMouth & (self.bpos[:, 0] < -W2 - CFG['BALL_R'])   # gol do time 1
        g0 = play & inMouth & (self.bpos[:, 0] > W2 + CFG['BALL_R'])
        goal = g0 | g1
        if not goal.any():
            return
        team = torch.where(g0, 0, 1)
        self.score[goal, team[goal]] += 1
        self.kickoffTeam = torch.where(goal, 1 - team, self.kickoffTeam)
        self.state = torch.where(goal, torch.full_like(self.state, M_GOAL), self.state)
        self.stateT = torch.where(goal, torch.full_like(self.stateT, CFG['GOAL_PAUSE']), self.stateT)
        self.owner = torch.where(goal, -1, self.owner); self.bvel = torch.where(goal.unsqueeze(-1), torch.zeros_like(self.bvel), self.bvel)
        self.chKind = torch.where(goal.view(B, 1), CH_NONE, self.chKind)
        ev = torch.zeros(B, 2, dtype=torch.bool, device=self.dev)
        ev[goal, team[goal]] = True
        self.events['goal'] = ev
        mercy = goal & ((self.score[:, 0] - self.score[:, 1]).abs() >= CFG['MERCY'])
        self.state = torch.where(mercy, torch.full_like(self.state, M_END), self.state)

    # ------------------------------------------------------------ API
    def act_from_discrete(self, a):
        """a: dict de tensores [B,P]: move (0..8), aim (0..15 setor relativo ao jogador), curve (0..2),
        shoot, pas, sprint, stance, special, tackle (0/1). Devolve o input do step."""
        B, P = self.B, self.P
        md = MOVE_DIRS.to(self.dev)[a['move']]                                   # [B,P,2]
        ang = a['aim'].float() * (2 * math.pi / AIM_SECTORS)
        aimv = torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        # referencial do time: ataque em +x -> espelha x para o time 1
        flip = self.dir.view(1, P, 1)
        md = md * torch.cat([flip, torch.ones_like(flip)], -1)
        aimv = aimv * torch.cat([flip, torch.ones_like(flip)], -1)
        inp = dict(mx=md[..., 0], my=md[..., 1], aim=self.pos + aimv * 600, curve=(a['curve'].float() - 1),
                   shoot=a['shoot'].bool(), pas=a['pas'].bool(), sprint=a['sprint'].bool(), stance=a['stance'].bool(),
                   special=a['special'].bool(), tackle=a['tackle'].bool())
        self._last_inp = inp
        return inp

    def stats(self):
        return dict(score=self.score.clone(), time=self.time.clone(), state=self.state.clone())
