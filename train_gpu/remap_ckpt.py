"""Remapeia um checkpoint PPO treinado com a observação antiga (135) para a nova (157):
copia as colunas das entradas existentes e zera as novas. Uso: python train_gpu/remap_ckpt.py"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
import features_torch as FT
from ppo import Policy

OLD = dict(SELF=33, BALL=14, MATE=12, OPP=14, GOALS=14, MISC=14, MM=4, MO=5)
NEW = dict(SELF=FT.SELF, BALL=FT.BALL, MATE=FT.MATE, OPP=FT.OPP, GOALS=FT.GOALS, MISC=FT.MISC, MM=FT.MAX_MATES, MO=FT.MAX_OPPS)


def mapping():
    m = []   # (old_idx, new_idx)
    o = n = 0
    for k in range(OLD['SELF']):
        m.append((o + k, n + k))
    o += OLD['SELF']; n += NEW['SELF']
    for k in range(OLD['BALL']):
        m.append((o + k, n + k))
    o += OLD['BALL']; n += NEW['BALL']
    for i in range(OLD['MM']):
        for k in range(OLD['MATE']):
            m.append((o + i * OLD['MATE'] + k, n + i * NEW['MATE'] + k))
    o += OLD['MM'] * OLD['MATE']; n += NEW['MM'] * NEW['MATE']
    for i in range(OLD['MO']):
        for k in range(OLD['OPP']):
            m.append((o + i * OLD['OPP'] + k, n + i * NEW['OPP'] + k))
    o += OLD['MO'] * OLD['OPP']; n += NEW['MO'] * NEW['OPP']
    for k in range(OLD['GOALS']):
        m.append((o + k, n + k))
    o += OLD['GOALS']; n += NEW['GOALS']
    for k in range(OLD['MISC']):
        m.append((o + k, n + k))
    return m


def main():
    path = os.path.join(os.path.dirname(__file__), 'ckpt.pt')
    ck = torch.load(path, map_location='cpu')
    old = ck['pol']
    if old['l1.weight'].shape[1] == FT.SIZE:
        print('checkpoint já está na observação nova'); return
    torch.save(ck, path.replace('ckpt.pt', 'ckpt_obs135.pt'))
    pol = Policy()
    sd = pol.state_dict()
    for k, v in old.items():
        if k in ('l1.weight', 'v1.weight') and v.shape[1] != FT.SIZE:
            w = torch.zeros_like(sd[k])
            for oi, ni in mapping():
                w[:, ni] = v[:, oi]
            sd[k] = w
        else:
            sd[k] = v
    pol.load_state_dict(sd)
    ck['pol'] = pol.state_dict()
    ck.pop('opt', None)   # otimizador tem formas antigas: recomeça o Adam
    torch.save(ck, path)
    print('remapeado', old['l1.weight'].shape[1], '->', FT.SIZE, 'iteração', ck.get('it'))


if __name__ == '__main__':
    main()
