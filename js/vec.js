'use strict';
// Utilitários de vetor 2D (objetos {x, y} imutáveis por convenção).
const V = {
  v: (x, y) => ({ x, y }),
  add: (a, b) => ({ x: a.x + b.x, y: a.y + b.y }),
  sub: (a, b) => ({ x: a.x - b.x, y: a.y - b.y }),
  mul: (a, s) => ({ x: a.x * s, y: a.y * s }),
  dot: (a, b) => a.x * b.x + a.y * b.y,
  len: (a) => Math.hypot(a.x, a.y),
  dist: (a, b) => Math.hypot(a.x - b.x, a.y - b.y),
  norm: (a) => {
    const l = Math.hypot(a.x, a.y);
    return l > 1e-9 ? { x: a.x / l, y: a.y / l } : { x: 0, y: 0 };
  },
  rot: (a, ang) => {
    const c = Math.cos(ang), s = Math.sin(ang);
    return { x: a.x * c - a.y * s, y: a.x * s + a.y * c };
  },
  perp: (a) => ({ x: -a.y, y: a.x }),
  angle: (a) => Math.atan2(a.y, a.x),
  fromAngle: (ang) => ({ x: Math.cos(ang), y: Math.sin(ang) }),
  lerp: (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }),
  clamp: (x, lo, hi) => Math.max(lo, Math.min(hi, x)),
  angleDiff: (a, b) => {
    let d = b - a;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    return d;
  },
  reflect: (v, n) => {
    const d = v.x * n.x + v.y * n.y;
    return { x: v.x - 2 * d * n.x, y: v.y - 2 * d * n.y };
  },
  // distância do ponto p ao segmento ab
  segDist: (p, a, b) => {
    const ab = V.sub(b, a);
    const t = V.clamp(V.dot(V.sub(p, a), ab) / Math.max(1e-9, V.dot(ab, ab)), 0, 1);
    return V.dist(p, V.add(a, V.mul(ab, t)));
  },
};
