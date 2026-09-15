'use strict';
// Loop principal + lobby. Modos: solo (bots), host (simula e envia estado) e
// convidado (envia inputs, recebe estado).
(function () {
  const canvas = document.getElementById('game');
  const renderer = new Renderer(canvas);
  const params = new URLSearchParams(location.search);
  const $ = (id) => document.getElementById(id);
  const lobby = $('lobby'), status = $('status');

  let mode = null;          // 'solo' | 'host' | 'guest'
  let game = null;
  let humanId = -1;
  let net = null;
  let paused = false;
  let teamSize = V.clamp(parseInt(params.get('n') || CFG.TEAM_SIZE, 10) || CFG.TEAM_SIZE, 3, 5);
  let seed = (Date.now() & 0xffff) || 1;
  const remoteQueues = new Map();   // pid -> [inputs]
  const remoteLast = new Map();     // pid -> último input
  const remoteNames = new Map();    // pid -> nome
  let netEvents = [];
  let tick = 0, seq = 0;
  let withBots = true;              // host: vagas sem humano têm bot (true) ou ficam vazias (false)

  // ---------- lobby ----------
  $('teamSize').value = String(teamSize);
  $('teamSize').addEventListener('change', (e) => { teamSize = parseInt(e.target.value, 10); });
  let savedName = '';
  try { savedName = localStorage.getItem('haxmatch-name') || ''; } catch (e) { /* sem storage */ }
  $('name').value = savedName;
  const myName = () => {
    const n = $('name').value.trim().slice(0, 12) || 'Jogador';
    try { localStorage.setItem('haxmatch-name', n); } catch (e) { /* sem storage */ }
    return n;
  };
  $('btnSolo').addEventListener('click', () => startSolo());
  $('btnHost').addEventListener('click', () => startHost());
  $('btnJoin').addEventListener('click', () => startGuest(NetUtil.normalizeCode($('code').value)));
  $('code').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('btnJoin').click(); });
  if (params.get('join')) { $('code').value = NetUtil.normalizeCode(params.get('join')); }
  const setStatus = (t, isError) => { status.textContent = t; status.className = isError ? 'err' : ''; };
  const hideLobby = () => { lobby.hidden = true; canvas.focus(); };
  const showLobby = () => { lobby.hidden = false; };

  function newGame() {
    game = new Game({ teamSize, seed });
    remoteQueues.clear(); remoteLast.clear(); remoteNames.clear();
    tick = 0; seq = 0; netEvents = [];
  }

  function startSolo() {
    mode = 'solo';
    newGame();
    humanId = teamSize - 1;
    game.players[humanId].human = true;
    game.players[humanId].name = myName();
    hideLobby();
  }

  function startHost() {
    if (typeof Peer === 'undefined') { setStatus('PeerJS não carregou (sem internet?).', true); return; }
    mode = 'host';
    withBots = $('withBots').checked;
    newGame();
    humanId = teamSize - 1;
    game.players[humanId].human = true;
    game.players[humanId].name = myName();
    applyBotsSetting();
    const code = NetUtil.makeCode();
    setStatus('Criando sala…');
    net = new NetHost(code, {
      teamSize, seed,
      onReady(c) {
        const link = location.origin.startsWith('http') ? `${location.origin}${location.pathname}?join=${c}` : null;
        $('roomInfo').hidden = false;
        $('roomCode').textContent = c;
        $('roomLink').textContent = link || '(abra o jogo hospedado em http(s) para ter um link compartilhável)';
        setStatus('Sala criada. Você já pode jogar; amigos entram a qualquer momento.');
        hideLobby();
        renderer.roomCode = c;
      },
      onError(msg) { setStatus(msg, true); },
      onJoin(conn, name) {
        const slot = freeSlot();
        if (slot === null) return null;
        const p = game.players[slot];
        p.human = true; p.active = true; p.name = name || `P${slot}`;
        if (game.state === 'play' && V.dist(p.pos, { x: 0, y: -CFG.FIELD_H }) < 1) p.pos = { x: p.home.x, y: p.home.y };
        remoteQueues.set(slot, []); remoteLast.set(slot, emptyInput()); remoteNames.set(slot, p.name);
        renderer.addFlash(`${p.name} entrou`, p.pos, '#fff');
        return slot;
      },
      onLeave(pid) {
        const p = game.players[pid];
        p.human = false; p.name = (p.team === 0 ? 'V' : 'A') + (p.idx + 1);
        if (!withBots) { p.active = false; if (game.ball.owner === p) game.ball.owner = null; p.pos = { x: 0, y: -CFG.FIELD_H }; }
        remoteQueues.delete(pid); remoteLast.delete(pid); remoteNames.delete(pid);
      },
      onInput(pid, inp) {
        const q = remoteQueues.get(pid);
        if (!q) return;
        if (q.length > 4) q.shift();
        q.push(sanitizeInput(inp));
      },
    });
  }

  // alterna times: 1º convidado no time azul, 2º no vermelho, …
  function freeSlot() {
    const order = [];
    for (let i = teamSize - 1; i >= 0; i--) { order.push(teamSize + i); order.push(i); }
    for (const id of order) if (!game.players[id].human) return id;
    return null;
  }

  function sanitizeInput(i) {
    const o = emptyInput();
    if (!i || typeof i !== 'object') return o;
    const num = (v) => (Number.isFinite(v) ? v : 0);
    o.mx = V.clamp(num(i.mx), -1, 1); o.my = V.clamp(num(i.my), -1, 1);
    o.aim = { x: V.clamp(num(i.aim && i.aim.x), -5000, 5000), y: V.clamp(num(i.aim && i.aim.y), -5000, 5000) };
    for (const k of ['shoot', 'pass', 'sprint', 'stance', 'special', 'tackle', 'throwBall', 'call']) o[k] = !!i[k];
    return o;
  }

  function startGuest(code) {
    if (typeof Peer === 'undefined') { setStatus('PeerJS não carregou (sem internet?).', true); return; }
    if (code.length !== 5) { setStatus('Código inválido (5 caracteres).', true); return; }
    mode = 'guest';
    game = null;
    setStatus('Conectando…');
    net = new NetGuest(code, myName(), {
      onWelcome(m) {
        teamSize = m.teamSize; seed = m.seed;
        newGame();
        humanId = m.pid;
        renderer.roomCode = code;
        hideLobby();
      },
      onState(s) {
        if (!game) return;
        if (s.seq < seq) return;       // snapshot antigo
        seq = s.seq;
        const evs = NetState.apply(game, s);
        for (const e of evs) onEvent(e);
      },
      onClose(reason) { leaveToLobby(reason); },
      onError(msg) { setStatus(msg, true); mode = null; },
    });
  }

  function leaveToLobby(reason) {
    if (net) { net.destroy(); net = null; }
    mode = null; game = null; humanId = -1;
    $('roomInfo').hidden = true;
    renderer.roomCode = null;
    setStatus(reason || '', !!reason);
    showLobby();
  }

  // ---------- input humano ----------
  const input = new HumanInput(canvas, {
    onKey(code) {
      if (!game) return;
      if (code === 'Tab' && mode === 'solo') switchPlayer();
      if (code === 'KeyR' && mode !== 'guest') { game.reset(); restoreHumans(); }
      if (code === 'KeyH') renderer.showHelp = !renderer.showHelp;
      if (code === 'KeyP' && mode === 'solo') paused = !paused;
      if (code === 'Enter') toggleFullscreen();
      if (code === 'Escape' && !document.fullscreenElement) leaveToLobby('');
    },
  });

  // reset recria os jogadores: devolve as vagas dos humanos (host e convidados)
  function restoreHumans() {
    const me = game.players[humanId];
    me.human = true; me.name = myName();
    for (const [pid, name] of remoteNames) { game.players[pid].human = true; game.players[pid].name = name; }
    if (mode === 'host') applyBotsSetting();
  }

  // sem bots, vagas sem humano ficam fora do campo
  function applyBotsSetting() {
    for (const p of game.players) {
      p.active = withBots || p.human;
      if (!p.active) p.pos = { x: 0, y: -CFG.FIELD_H };
    }
  }

  function switchPlayer() {
    const team = game.players[humanId].team;
    game.players[humanId].human = false;
    const teamIds = game.players.filter((p) => p.team === team).map((p) => p.id);
    const i = teamIds.indexOf(humanId);
    humanId = teamIds[(i + 1) % teamIds.length];
    game.players[humanId].human = true;
  }

  // Tela cheia + Keyboard Lock: em tela cheia o navegador (Chrome/Edge) passa a
  // entregar Ctrl+W, Ctrl+S, Tab etc. para a página em vez de executar o atalho.
  const LOCK_KEYS = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyE', 'KeyF', 'KeyC', 'KeyR', 'KeyH', 'KeyP', 'KeyQ',
    'Tab', 'Space', 'ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight',
    'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
  async function toggleFullscreen() {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen({ navigationUI: 'hide' });
        if (navigator.keyboard && navigator.keyboard.lock) await navigator.keyboard.lock(LOCK_KEYS);
      } else {
        await document.exitFullscreen();
      }
    } catch (e) { console.warn('tela cheia indisponível:', e); }
  }
  document.addEventListener('fullscreenchange', () => {
    renderer.fullscreen = !!document.fullscreenElement;
    if (!document.fullscreenElement && navigator.keyboard && navigator.keyboard.unlock) navigator.keyboard.unlock();
    renderer.resize();
  });
  renderer.keyboardLock = !!(navigator.keyboard && navigator.keyboard.lock);

  // ---------- loop ----------
  let last = performance.now();
  let acc = 0;

  function frame(now) {
    const frameDt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (game) {
      if (mode === 'guest') guestFrame(frameDt);
      else if (!paused) simFrame(frameDt);
      const human = game.players[humanId];
      renderer.updateCamera(game, human, frameDt);
      renderer.draw(game, human, frameDt);
    }
    requestAnimationFrame(frame);
  }

  function simFrame(frameDt) {
    acc += frameDt;
    while (acc >= CFG.DT) {
      const human = game.players[humanId];
      for (const p of game.players) {
        if (p === human) game.setInput(p.id, input.sample((sx, sy) => renderer.screenToWorld(sx, sy)));
        else if (p.human && mode === 'host') game.setInput(p.id, nextRemoteInput(p.id));
        else game.setInput(p.id, AI.think(p, game, CFG.DT));
      }
      game.step(CFG.DT);
      for (const e of game.events) onEvent(e);
      if (mode === 'host') {
        netEvents.push(...game.events);
        tick++;
        if (tick % 2 === 0) {
          if (net && net.count > 0) net.broadcast(NetState.encode(game, ++seq, netEvents));
          netEvents = [];
        }
      }
      acc -= CFG.DT;
    }
  }

  function nextRemoteInput(pid) {
    const q = remoteQueues.get(pid);
    if (q && q.length) { const i = q.shift(); remoteLast.set(pid, i); return i; }
    return remoteLast.get(pid) || emptyInput();
  }

  let guestAcc = 0;
  function guestFrame(frameDt) {
    const human = game.players[humanId];
    const inp = input.sample((sx, sy) => renderer.screenToWorld(sx, sy));
    human.input = inp;                     // para mira/preview locais
    guestAcc += frameDt;
    while (guestAcc >= CFG.DT) { net.send({ t: 'input', i: inp }); guestAcc -= CFG.DT; }
    if (game.state === 'play') NetState.extrapolate(game, frameDt);
  }

  function onEvent(e) {
    const p = e.p;
    switch (e.type) {
      case 'steal': renderer.addFlash('roubou!', p.pos, '#ffd'); break;
      case 'tackle-fail': if (e.victim) renderer.addFlash('escapou', e.victim.pos, '#ffb347'); break;
      case 'knockdown': if (e.victim) renderer.addFlash('carrinho!', e.victim.pos, '#ff8'); break;
      case 'save': renderer.addFlash('DEFESA!', p.pos, '#ffd84a'); break;
      case 'parry': renderer.addFlash('espalmou', p.pos, '#ffd84a'); break;
      case 'effort': renderer.addFlash('arrancada', p.pos, '#ffe66d'); break;
      case 'gloves': renderer.addFlash('luvas', p.pos, '#ffd84a'); break;
      case 'fake': renderer.addFlash('finta', p.pos, '#fff'); break;
      case 'first-touch': renderer.addFlash('de primeira!', p.pos, '#fff'); break;
    }
  }

  window.addEventListener('resize', () => renderer.resize());
  window.addEventListener('beforeunload', () => { if (net) net.destroy(); });
  requestAnimationFrame(frame);
})();
