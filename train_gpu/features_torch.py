"""Observação vetorizada [B,P,135], na MESMA ordem e normalização de js/features.js,
para que os pesos treinados aqui rodem no navegador sem conversão."""
import math
import torch
from sim_torch import CFG, ACT_NONE, ST_DRIB, ST_DEF, CH_SHOT, Q_NONE, M_PLAY

MAX_MATES, MAX_OPPS = 4, 5
SELF, BALL, MATE, OPP, GOALS, MISC = 35, 14, 12, 14, 14, 14
SIZE = SELF + BALL + MAX_MATES * MATE + MAX_OPPS * OPP + GOALS + MISC   # 135
GX, GY = 12, 7


def pitch_control(sim):
    """dono (jogador mais próximo) de cada célula da grade -> count [B,P], owner [B,GX*GY]"""
    B, P, d = sim.B, sim.P, sim.dev
    W2, H2 = CFG['FIELD_W'] / 2, CFG['FIELD_H'] / 2
    cx = -W2 + (torch.arange(GX, device=d) + 0.5) * CFG['FIELD_W'] / GX
    cy = -H2 + (torch.arange(GY, device=d) + 0.5) * CFG['FIELD_H'] / GY
    cells = torch.stack(torch.meshgrid(cx, cy, indexing='xy'), -1).reshape(-1, 2)   # [GX*GY,2] (j*GX+i)
    dd = ((cells.view(1, -1, 1, 2) - sim.pos.view(B, 1, P, 2)) ** 2).sum(-1)      # [B,C,P]
    owner = dd.argmin(dim=-1)                                                        # [B,C]
    count = torch.zeros(B, P, device=d).scatter_add_(1, owner, torch.ones_like(owner, dtype=torch.float))
    return count, owner, cells


def build(sim):
    B, P, d = sim.B, sim.P, sim.dev
    T = sim.T
    W2, H2 = CFG['FIELD_W'] / 2, CFG['FIELD_H'] / 2
    POS, VEL, DIST = 1 / W2, 1 / 300, 1 / 1000
    dirp = sim.dir.view(1, P, 1)                       # [1,P,1] direção de ataque
    team = sim.team.view(1, P)
    pidx = torch.arange(P, device=d).view(1, P)
    hasBall = (sim.owner.view(B, 1) == pidx) & ~sim.held
    held = (sim.owner.view(B, 1) == pidx) & sim.held
    inBox = sim.in_own_box()
    count, cellOwner, cells = pitch_control(sim)
    ncell = GX * GY
    out = []
    put = out.append
    flipx = torch.cat([dirp, torch.ones_like(dirp)], -1)          # multiplica x por dir

    def relpos(q):   # q [B,P,2] -> [B,P,2]
        return (q - sim.pos) * flipx * POS

    def vel(v):
        return v * flipx * VEL

    # ---- eu (33) ----
    put(sim.pos[..., 0:1] * dirp * POS); put(sim.pos[..., 1:2] / H2)
    put(relpos(sim.home.view(1, P, 2).expand(B, -1, -1)))
    for i in range(5):
        put((sim.idx.view(1, P) == i).float().unsqueeze(-1).expand(B, -1, -1))
    put(vel(sim.vel))
    put(sim.facing * flipx)
    put((sim.stamina / CFG['STAMINA_MAX']).unsqueeze(-1)); put(sim.exhausted.float().unsqueeze(-1))
    put(sim.effortBar.unsqueeze(-1)); put((sim.effortT > 0).float().unsqueeze(-1))
    put(sim.isKeeper.float().unsqueeze(-1))
    put(hasBall.float().unsqueeze(-1)); put(held.float().unsqueeze(-1))
    put(inBox.float().unsqueeze(-1))
    put((sim.stance == ST_DRIB).float().unsqueeze(-1)); put((sim.stance == ST_DEF).float().unsqueeze(-1))
    put((sim.cd['tackle'] > 0).float().unsqueeze(-1)); put((sim.cd['slide'] > 0).float().unsqueeze(-1)); put((sim.cd['dribble'] > 0).float().unsqueeze(-1))
    put((sim.act != ACT_NONE).float().unsqueeze(-1))
    put(((sim.recover > 0) | (sim.fallen > 0) | (sim.getup > 0)).float().unsqueeze(-1))
    put((sim.dribbleLag > 0).float().unsqueeze(-1))
    chMax = torch.where(sim.chKind == CH_SHOT, CFG['CHARGE_MAX'], CFG['PASS_CHARGE'])
    put(torch.where(sim.chKind != 0, (sim.chT / chMax).clamp(max=1), torch.zeros_like(sim.chT)).unsqueeze(-1))
    put(torch.where(sim.qKind != Q_NONE, (sim.qAge / CFG['LOCK_MAX']).clamp(max=1), torch.zeros_like(sim.qAge)).unsqueeze(-1))
    put((count / ncell * 4).unsqueeze(-1))
    # espaço livre à frente: adversário mais próximo num cone
    rel = sim.pos.view(B, 1, P, 2) - sim.pos.view(B, P, 1, 2)          # [B,p,q]: q - p
    dx = rel[..., 0] * dirp.view(1, P, 1); dy = rel[..., 1].abs()
    opp = team.view(1, P, 1) != team.view(1, 1, P)
    cone = opp & (dx > 0) & (dy < dx * 0.8 + 40)
    dq = rel.norm(dim=-1)
    free = torch.where(cone, dq, torch.full_like(dq, 600.0)).min(dim=-1).values.clamp(max=600)
    put((free / 600).unsqueeze(-1))
    put((sim.cd['dash'] > 0).float().unsqueeze(-1)); put((sim.cd['dive'] > 0).float().unsqueeze(-1))

    # ---- bola (12) ----
    bp = sim.bpos.view(B, 1, 2).expand(B, P, 2)
    put(relpos(bp)); put(((bp - sim.pos).norm(dim=-1) * DIST).unsqueeze(-1))
    bv = sim.bvel.view(B, 1, 2).expand(B, P, 2)
    put(vel(bv)); put((bv.norm(dim=-1) * DIST).unsqueeze(-1))
    o = sim.owner.view(B, 1)
    has = o >= 0
    oteam = torch.where(has, sim.team[o.clamp(min=0)], torch.full_like(o, -1))
    put((~has).float().expand(B, P).unsqueeze(-1))
    put((o == pidx).float().unsqueeze(-1))
    put((has & (o != pidx) & (oteam == team)).float().unsqueeze(-1))
    put((has & (oteam != team)).float().unsqueeze(-1))
    put(sim.reach.float().unsqueeze(-1))
    put(((sim.lock.view(B, 1) >= 0) & (sim.lock.view(B, 1) != pidx)).float().unsqueeze(-1))
    kdrag = CFG['BALL_DRAG']
    spd = sim.bvel.norm(dim=-1)
    tstop = (torch.log((spd / 40).clamp(min=1)) / kdrag).clamp(0, 3)
    fst = (1 - torch.exp(-kdrag * tstop)) / kdrag
    stopPt = sim.bpos + sim.bvel * fst.unsqueeze(-1)
    stopPt = torch.stack([stopPt[:, 0].clamp(-W2 + 10, W2 - 10), stopPt[:, 1].clamp(-H2 + 10, H2 - 10)], -1)
    put(relpos(stopPt.view(B, 1, 2).expand(B, P, 2)))

    # ---- companheiros (4×8) e adversários (5×8), por distância ----
    same = (team.view(1, P, 1) == team.view(1, 1, P)) & ~torch.eye(P, dtype=torch.bool, device=d).view(1, P, P)
    oppm = team.view(1, P, 1) != team.view(1, 1, P)
    qHas = (o.view(B, 1, 1) == torch.arange(P, device=d).view(1, 1, P)).float()   # [B,1,P] -> q tem a bola
    qK = sim.isKeeper.float().view(B, 1, P)
    qShare = (count / ncell * 4).view(B, 1, P)

    # por jogador q: distância ao adversário mais próximo DELE, adversários a <200 dele, distância à bola
    oppOfQ = (team.view(1, P, 1) != team.view(1, 1, P))                       # [1,q,o]
    dqo = torch.where(oppOfQ.expand(B, -1, -1), dq, torch.full_like(dq, 1000.0))
    qNearOpp = (dqo.min(dim=-1).values / 400).clamp(max=1)                    # [B,q]
    qOppsNear = ((dqo < 200).sum(-1).float() / 5)                              # [B,q]
    qBallD = ((sim.pos - sim.bpos.view(B, 1, 2)).norm(dim=-1) * DIST).clamp(max=1)   # [B,q]
    # linha de passe da bola até q livre (nenhum adversário de q a menos de r+26 da linha, fora as pontas)
    def seg_dist_pts(pt, a, bb):   # pt [B,1,P,2] (obstáculos), a [B,1,1,2], bb [B,P,1,2] -> [B,P,P]
        ab = bb - a
        t = ((pt - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)
        t = t.clamp(0, 1)
        proj = a + ab * t.unsqueeze(-1)
        return (pt - proj).norm(dim=-1)
    bpt = sim.bpos.view(B, 1, 1, 2)
    sd = seg_dist_pts(sim.pos.view(B, 1, P, 2), bpt, sim.pos.view(B, P, 1, 2))   # [B,q,o]
    dOA = (sim.pos.view(B, 1, P, 2) - bpt).norm(dim=-1)                          # [B,1,o] obstáculo -> bola
    dOB = (sim.pos.view(B, 1, P, 2) - sim.pos.view(B, P, 1, 2)).norm(dim=-1)     # [B,q,o] obstáculo -> q
    blocks = oppOfQ.expand(B, -1, -1) & (dOA.expand(B, P, P) > 30) & (dOB > 30) & (sd < CFG['PLAYER_R'] + 26)
    qLane = (~blocks.any(dim=-1)).float()                                         # [B,q]
    # espaço livre à frente de cada q (cone de ataque de q)
    relq = sim.pos.view(B, 1, P, 2) - sim.pos.view(B, P, 1, 2)                   # [B,q,o]: o - q
    dxq = relq[..., 0] * dirp.view(1, P, 1); dyq = relq[..., 1].abs()
    coneq = oppOfQ.expand(B, -1, -1) & (dxq > 0) & (dyq < dxq * 0.8 + 40)
    qFree = (torch.where(coneq, relq.norm(dim=-1), torch.full_like(dq, 600.0)).min(dim=-1).values.clamp(max=600) / 600)   # [B,q]
    qDown = ((sim.fallen > 0) | (sim.getup > 0)).float()
    qAct = (sim.act != ACT_NONE).float()
    qDef = (sim.stance == ST_DEF).float()
    # perigo: distância de q ao gol do OBSERVADOR (p) -> [B,p,q]
    ownGoalP = torch.stack([-sim.dir * W2, torch.zeros(P, device=d)], -1).view(1, P, 1, 2)
    qDanger = ((sim.pos.view(B, 1, P, 2) - ownGoalP).norm(dim=-1) * DIST).clamp(max=1)   # [B,p,q]
    # adversário na minha linha de chute (segmento eu -> gol adversário)
    goalPt = torch.stack([sim.dir * W2, torch.zeros(P, device=d)], -1).view(1, P, 1, 2).expand(B, -1, -1, -1)
    sdG = seg_dist_pts(sim.pos.view(B, 1, P, 2), sim.pos.view(B, P, 1, 2), goalPt)   # [B,p,o]: dist do o à linha p->gol
    inShotLane = (sdG < CFG['PLAYER_R'] + 30).float()                                # [B,p,o]

    def group(mask, nmax, extra):
        dist = torch.where(mask.expand(B, -1, -1), dq, torch.full_like(dq, 1e9))
        order = dist.argsort(dim=-1)[..., :nmax]                     # [B,P,nmax]
        valid = torch.gather(dist, 2, order) < 1e8
        idxe = order.unsqueeze(-1).expand(B, P, nmax, 2)
        qpos = torch.gather(sim.pos.view(B, 1, P, 2).expand(B, P, P, 2), 2, idxe)
        qvel = torch.gather(sim.vel.view(B, 1, P, 2).expand(B, P, P, 2), 2, idxe)
        rp = (qpos - sim.pos.view(B, P, 1, 2)) * flipx.view(1, P, 1, 2) * POS
        rv = qvel * flipx.view(1, P, 1, 2) * VEL
        hb = torch.gather(qHas.expand(B, P, P), 2, order)
        kk = torch.gather(qK.expand(B, P, P), 2, order)
        sh = torch.gather(qShare.expand(B, P, P), 2, order)
        cols = [valid.float().unsqueeze(-1), rp, rv, hb.unsqueeze(-1), kk.unsqueeze(-1), sh.unsqueeze(-1)]
        for e in extra:
            cols.append(torch.gather(e, 2, order).unsqueeze(-1))
        feat = torch.cat(cols, -1)
        feat = feat * valid.unsqueeze(-1)
        return feat.reshape(B, P, nmax * (8 + len(extra)))

    put(group(same, MAX_MATES, [qNearOpp.view(B, 1, P).expand(B, P, P), qOppsNear.view(B, 1, P).expand(B, P, P), qLane.view(B, 1, P).expand(B, P, P), qFree.view(B, 1, P).expand(B, P, P)]))
    put(group(oppm, MAX_OPPS, [qBallD.view(B, 1, P).expand(B, P, P), inShotLane, qDown.view(B, 1, P).expand(B, P, P), qAct.view(B, 1, P).expand(B, P, P), qDef.view(B, 1, P).expand(B, P, P), qDanger]))

    # ---- gols (6) ----
    oppGoal = torch.stack([sim.dir * W2, torch.zeros(P, device=d)], -1).view(1, P, 2).expand(B, -1, -1)
    ownGoal = -oppGoal
    put(relpos(oppGoal)); put(((oppGoal - sim.pos).norm(dim=-1) * DIST).unsqueeze(-1))
    put(relpos(ownGoal)); put(((ownGoal - sim.pos).norm(dim=-1) * DIST).unsqueeze(-1))
    # linha livre da bola para cada canto (bloqueadores: adversários de linha)
    blockersM = oppm.expand(B, -1, -1) & ~sim.isKeeper.view(B, 1, P)
    for sgn in (-1.0, 1.0):
        corner = torch.stack([sim.dir * W2, torch.full((P,), sgn * (CFG['GOAL_W'] / 2 - 30), device=d)], -1).view(1, P, 1, 2).expand(B, -1, -1, -1)
        sdc = seg_dist_pts(sim.pos.view(B, 1, P, 2), bpt, corner)            # [B,p,o]
        blk = blockersM & (sdc < CFG['PLAYER_R'] + 30)
        put((~blk.any(dim=-1)).float().unsqueeze(-1))
    # goleiro adversário relativo ao gol dele
    gkm = oppm.expand(B, -1, -1) & sim.isKeeper.view(B, 1, P)
    hasGk = gkm.any(dim=-1)
    gki = gkm.float().argmax(dim=-1)                                           # [B,p]
    gkpos = torch.gather(sim.pos.view(B, 1, P, 2).expand(B, P, P, 2), 2, gki.view(B, P, 1, 1).expand(B, P, 1, 2)).squeeze(2)
    put(hasGk.float().unsqueeze(-1))
    put((hasGk.float() * (gkpos[..., 0] - sim.dir.view(1, P) * W2) * sim.dir.view(1, P) * DIST).unsqueeze(-1))
    put((hasGk.float() * gkpos[..., 1] / (CFG['GOAL_W'] / 2)).unsqueeze(-1))
    # espaço livre à esquerda e à direita
    for sgn in (-1.0, 1.0):
        dys = rel[..., 1] * sgn; dxs = rel[..., 0].abs()
        cone_s = opp & (dys > 0) & (dxs < dys * 0.8 + 40)
        put((torch.where(cone_s, dq, torch.full_like(dq, 600.0)).min(dim=-1).values.clamp(max=600) / 600).unsqueeze(-1))
    put(torch.zeros(B, P, 1, device=d))   # reservado

    # ---- diversos (12) ----
    keeperTeam = torch.zeros(B, 2, dtype=torch.bool, device=d)
    keeperTeam[:, 0] = (sim.isKeeper & (team == 0)).any(dim=1); keeperTeam[:, 1] = (sim.isKeeper & (team == 1)).any(dim=1)
    put(torch.gather(keeperTeam.float(), 1, team.expand(B, -1)).unsqueeze(-1))
    put(torch.gather(keeperTeam.float(), 1, (1 - team).expand(B, -1)).unsqueeze(-1))
    sd = (torch.gather(sim.score, 1, team.expand(B, -1)) - torch.gather(sim.score, 1, (1 - team).expand(B, -1))).float()
    put((sd / 3).clamp(-1, 1).unsqueeze(-1))
    put((sim.time / CFG['MATCH_TIME']).view(B, 1, 1).expand(B, P, 1))
    put(((H2 - sim.pos[..., 1].abs()) * DIST).unsqueeze(-1))
    put(((W2 - sim.pos[..., 0].abs()) * DIST).unsqueeze(-1))
    near = dq < 220
    put(((near & same).sum(-1).float() / 4).unsqueeze(-1))
    put(((near & oppm).sum(-1).float() / 5).unsqueeze(-1))
    minOpp = torch.where(oppm.expand(B, -1, -1), dq, torch.full_like(dq, 1000.0)).min(dim=-1).values
    put((minOpp / 400).clamp(max=1).unsqueeze(-1))
    # controle de campo do time: total, terço de ataque, terço de defesa
    ownerTeam = sim.team[cellOwner]                                   # [B,C]
    cx = cells[:, 0].view(1, 1, -1) * dirp.view(1, P, 1) / W2         # [1,P,C] x no referencial de cada jogador
    mine = (ownerTeam.view(B, 1, -1) == team.view(1, P, 1)).float()  # [B,P,C]
    put(mine.mean(dim=-1, keepdim=True))
    att = (cx >= 1 / 3).float().expand(B, -1, -1); dfn = (cx <= -1 / 3).float().expand(B, -1, -1)
    put(((mine * att).sum(-1) / att.sum(-1).clamp(min=1)).unsqueeze(-1))
    put(((mine * dfn).sum(-1) / dfn.sum(-1).clamp(min=1)).unsqueeze(-1))
    # contra-ataque: adversários de linha mais perto do meu gol do que meu último defensor de linha
    xs = sim.pos[..., 0].view(B, 1, P) * dirp.view(1, P, 1)                     # [B,p,q] x de q no referencial de p
    mineField = (team.view(1, P, 1) == team.view(1, 1, P)) & ~sim.isKeeper.view(B, 1, P)   # inclui o próprio jogador (como no JS)
    lastX = torch.where(mineField.expand(B, -1, -1), xs, torch.full_like(xs, 1e9)).min(dim=-1).values
    lastX = torch.where(lastX > 1e8, torch.zeros_like(lastX), lastX)
    oppField = oppm.expand(B, -1, -1) & ~sim.isKeeper.view(B, 1, P)
    put(((oppField & (xs < lastX.unsqueeze(-1))).sum(-1).float() / 5).unsqueeze(-1))
    put((sim.state == M_PLAY).float().view(B, 1, 1).expand(B, P, 1))
    x = torch.cat(out, -1)
    assert x.shape[-1] == SIZE, x.shape
    return x
