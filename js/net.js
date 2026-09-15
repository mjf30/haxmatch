'use strict';
// Rede P2P via PeerJS (WebRTC). O host roda a simulação; convidados mandam
// inputs e recebem snapshots. O servidor público do PeerJS só faz a
// apresentação inicial; depois o tráfego é direto entre os navegadores.
const NetUtil = {
  PREFIX: 'haxmatch-v1-',
  makeCode() {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let c = '';
    for (let i = 0; i < 5; i++) c += chars[Math.floor(Math.random() * chars.length)];
    return c;
  },
  normalizeCode(c) { return (c || '').trim().toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 5); },
};

class NetHost {
  // handlers: onReady(code), onError(msg), onJoin(conn, name) -> pid|null, onLeave(pid), onInput(pid, input)
  constructor(code, handlers) {
    this.code = code;
    this.h = handlers;
    this.conns = new Map();   // conn -> pid
    this.peer = new Peer(NetUtil.PREFIX + code, { debug: 1 });
    this.peer.on('open', () => this.h.onReady(code));
    this.peer.on('error', (e) => this.h.onError(e.type === 'unavailable-id' ? 'Código de sala já em uso, tente outro.' : `Erro de rede: ${e.type || e}`));
    this.peer.on('connection', (conn) => {
      conn.on('data', (d) => {
        if (!d || typeof d !== 'object') return;
        const pid = this.conns.has(conn) ? this.conns.get(conn) : null;
        if (d.t === 'hello' && pid === null) {
          const np = this.h.onJoin(conn, String(d.name || '').slice(0, 12));
          if (np === null) { conn.send({ t: 'full' }); setTimeout(() => conn.close(), 200); return; }
          this.conns.set(conn, np);
          conn.send({ t: 'welcome', pid: np, teamSize: this.h.teamSize, seed: this.h.seed });
        } else if (d.t === 'input' && pid !== null) {
          this.h.onInput(pid, d.i);
        } else if (d.t === 'pong' && pid !== null && typeof d.ts === 'number') {
          this.h.onPing(pid, performance.now() - d.ts);
        } else if (d.t === 'switch' && pid !== null) {
          const np = this.h.onSwitch(pid);
          if (np !== null && np !== pid) { this.conns.set(conn, np); conn.send({ t: 'assign', pid: np }); }
        }
      });
      const bye = () => { if (this.conns.has(conn)) { const pid = this.conns.get(conn); this.conns.delete(conn); this.h.onLeave(pid); } };
      conn.on('close', bye);
      conn.on('error', bye);
    });
    this.pingTimer = setInterval(() => this.broadcast({ t: 'ping', ts: performance.now() }), 1000);
  }
  broadcast(msg) {
    for (const conn of this.conns.keys()) if (conn.open) { try { conn.send(msg); } catch (e) { /* conexão caindo */ } }
  }
  get count() { return this.conns.size; }
  destroy() { clearInterval(this.pingTimer); try { this.peer.destroy(); } catch (e) { /* já fechado */ } }
}

class NetGuest {
  // handlers: onWelcome(msg), onState(msg), onClose(reason), onError(msg)
  constructor(code, name, handlers) {
    this.h = handlers;
    this.conn = null;
    this.peer = new Peer({ debug: 1 });
    this.peer.on('error', (e) => this.h.onError(e.type === 'peer-unavailable' ? 'Sala não encontrada. Confira o código.' : `Erro de rede: ${e.type || e}`));
    this.peer.on('open', () => {
      const conn = this.peer.connect(NetUtil.PREFIX + code);
      this.conn = conn;
      conn.on('open', () => conn.send({ t: 'hello', name }));
      conn.on('data', (d) => {
        if (!d || typeof d !== 'object') return;
        if (d.t === 'welcome') this.h.onWelcome(d);
        else if (d.t === 'state') this.h.onState(d);
        else if (d.t === 'ping') conn.send({ t: 'pong', ts: d.ts });
        else if (d.t === 'assign') this.h.onAssign(d.pid);
        else if (d.t === 'full') this.h.onClose('Sala cheia.');
      });
      conn.on('close', () => this.h.onClose('Conexão encerrada pelo host.'));
      conn.on('error', () => this.h.onClose('Erro na conexão.'));
    });
  }
  send(msg) { if (this.conn && this.conn.open) { try { this.conn.send(msg); } catch (e) { /* ignora */ } } }
  destroy() { try { this.peer.destroy(); } catch (e) { /* já fechado */ } }
}
