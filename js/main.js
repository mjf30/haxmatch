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
  let botKind = 'script';           // 'script' (IA programada) ou 'nn' (rede neural treinada)
  const nnPolicy = (typeof NN_WEIGHTS !== 'undefined') ? (NN_WEIGHTS.kind === 'macro' ? MacroBot.fromExport(NN_WEIGHTS) : NNBot.fromExport(NN_WEIGHTS)) : null;
  const nnKind = (typeof NN_WEIGHTS !== 'undefined') ? NN_WEIGHTS.kind : null;
  const rawPolicy = (typeof NN_RAW_WEIGHTS !== 'undefined') ? RawBot.fromExport(NN_RAW_WEIGHTS) : null;
  const macroPpoPolicy = (typeof NN_MACRO_PPO !== 'undefined') ? MacroBot.fromExport(NN_MACRO_PPO) : null;
  const nnOffPolicy = nnPolicy ? Object.assign({}, nnPolicy, { offball: 'script' }) : null;   // rede com a bola, script sem a bola
  const controlPolicy = (typeof NN_GUIDED !== 'undefined') ? MacroBot.fromExport(NN_GUIDED) : null;   // híbrido PPO guiado pelo script (4v4)
  const msPolicy = (typeof NN_GUIDED_MS !== 'undefined') ? MacroBot.fromExport(NN_GUIDED_MS) : null;   // híbrido PPO guiado, treinado em 3v3/4v4/5v5
  if (!msPolicy) { const o = $('botKind').querySelector('option[value="nn_ms"]'); if (o) o.disabled = true; }
  for (const [k, pol] of [['nn_ppo', macroPpoPolicy], ['nn_zero', controlPolicy]]) if (!pol) { const o = $('botKind').querySelector(`option[value="${k}"]`); if (o) o.disabled = true; }
  if (!rawPolicy) { const o = $('botKind').querySelector('option[value="raw"]'); if (o) { o.disabled = true; o.textContent = 'rede PPO (sem pesos)'; } }
  if (!nnPolicy) { const o = $('botKind').querySelector('option[value="nn"]'); if (o) { o.disabled = true; o.textContent = 'rede neural (sem pesos)'; } }
  const botThink = (p, dt) => (botKind === 'raw' && rawPolicy ? RawBot.think(p, game, dt, rawPolicy)
    : botKind === 'nn_ppo' && macroPpoPolicy ? MacroBot.think(p, game, dt, macroPpoPolicy)
    : botKind === 'nn_off' && nnOffPolicy ? MacroBot.think(p, game, dt, nnOffPolicy)
    : botKind === 'nn_zero' && controlPolicy ? MacroBot.think(p, game, dt, controlPolicy)
    : botKind === 'nn_ms' && msPolicy ? MacroBot.think(p, game, dt, msPolicy)
    : botKind === 'nn' && nnPolicy ? (nnKind === 'macro' ? MacroBot.think(p, game, dt, nnPolicy) : NNBot.think(p, game, dt, nnPolicy)) : AI.think(p, game, dt));

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
  // duas telas: menu (lobby) e jogo (canvas). A captura de teclado só existe no jogo.
  const showGameScreen = () => { lobby.hidden = true; canvas.hidden = false; renderer.resize(); input.setEnabled(true); canvas.focus(); };
  const showMenuScreen = () => { input.setEnabled(false); canvas.hidden = true; lobby.hidden = false; };
  const hideLobby = () => showGameScreen();
  const showLobby = () => showMenuScreen();

  function newGame() {
    const st = $('botStyle') ? $('botStyle').value : 'balanced';
    game = new Game({ teamSize, seed, styles: [st, st] });
    remoteQueues.clear(); remoteLast.clear(); remoteNames.clear();
    tick = 0; seq = 0; netEvents = [];
  }

  function startSolo() {
    mode = 'solo';
    botKind = $('botKind').value;
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
    botKind = $('botKind').value;
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
      onPing(pid, rtt) { if (game && game.players[pid]) game.players[pid].ping = rtt; },
      onSwitch(pid) {
        const np = moveToOtherTeam(pid, remoteNames.get(pid) || game.players[pid].name);
        if (np === null) return null;
        for (const m of [remoteQueues, remoteLast, remoteNames]) { m.set(np, m.get(pid)); m.delete(pid); }
        return np;
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
    o.drag = { x: V.clamp(num(i.drag && i.drag.x), -300, 300), y: V.clamp(num(i.drag && i.drag.y), -300, 300) };
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
      onAssign(pid) { humanId = pid; },
      onClose(reason) { leaveToLobby(reason); },
      onError(msg) { setStatus(msg, true); mode = null; },
    });
  }

  // move o jogador da vaga pid para uma vaga livre do outro time; devolve a nova vaga
  let lastSwitch = -10;
  function moveToOtherTeam(pid, name) {
    const p = game.players[pid];
    const other = 1 - p.team;
    const free = game.players.filter((q) => q.team === other && !q.human).sort((a, b) => b.idx - a.idx);
    if (!free.length) return null;
    const q = free[0];
    // libera a vaga antiga
    p.human = false; p.name = (p.team === 0 ? 'V' : 'A') + (p.idx + 1);
    p.isKeeper = false; p.charge = null; p.queued = null;
    if (game.ball.owner === p) { game.ball.owner = null; p.held = false; }
    const botsHere = mode === 'solo' || withBots;
    if (!botsHere) { p.active = false; p.pos = { x: 0, y: -CFG.FIELD_H }; }
    // ocupa a nova
    q.human = true; q.name = name; q.isKeeper = false; q.stats = { goals: 0, assists: 0, steals: 0, saves: 0 };
    if (!q.active) { q.active = true; q.pos = { x: q.home.x, y: q.home.y }; }
    renderer.addFlash(`${name} → ${CFG.TEAM_NAMES[other]}`, q.pos, CFG.TEAM_COLORS[other]);
    return q.id;
  }

  function requestTeamSwitch() {
    if (!game || performance.now() - lastSwitch < 3000) return;
    lastSwitch = performance.now();
    if (mode === 'guest') { net.send({ t: 'switch' }); return; }
    const np = moveToOtherTeam(humanId, myName());
    if (np !== null) humanId = np;
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
      if (code === 'KeyQ' && mode === 'solo') switchPlayer();
      if (code === 'KeyR' && mode !== 'guest') { game.reset(); restoreHumans(); }
      if (code === 'KeyH') renderer.showHelp = !renderer.showHelp;
      if (code === 'KeyV') renderer.showValue = !renderer.showValue;   // camada: controle de campo x valor (EPV) e candidatos dos bots
      if (code === 'KeyP' && mode === 'solo') paused = !paused;
      if (code === 'Enter') toggleFullscreen();
      if (code === 'KeyT' && renderer.showScoreboard) requestTeamSwitch();
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
      const h0 = game.players[humanId];
      // mira travada enquanto o chute está segurado/travado (o arrasto vira efeito)
      input.setFrozen(!!((h0.charge && h0.charge.kind === 'shot') || (h0.queued && h0.queued.kind === 'shot')));
      if (mode === 'guest') guestFrame(frameDt);
      else if (!paused) simFrame(frameDt);
      const human = game.players[humanId];
      renderer.cursor = input.mouse; renderer.cursorFrozen = input.frozen;
      renderer.showScoreboard = input.down('Tab');
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
        if (p === human) game.setInput(p.id, renderer.showScoreboard ? Object.assign(emptyInput(), { aim: human.input.aim }) : input.sample((sx, sy) => renderer.screenToWorld(sx, sy)));
        else if (p.human && mode === 'host') game.setInput(p.id, nextRemoteInput(p.id));
        else game.setInput(p.id, botThink(p, CFG.DT));
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
    const inp = renderer.showScoreboard ? Object.assign(emptyInput(), { aim: human.input.aim }) : input.sample((sx, sy) => renderer.screenToWorld(sx, sy));
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
      case 'block': renderer.addFlash('bloqueou', p.pos, '#8cf'); break;
      case 'call': renderer.addFlash('bola!', p.pos, '#fff'); break;
      case 'lost-prio': renderer.addFlash('perdeu a prioridade', p.pos, '#f88'); break;
    }
  }

  canvas.addEventListener('mousedown', (e) => {
    if (game && !renderer.showScoreboard) input.lockPointer();   // mira virtual (Rematch): cursor preso ao jogo
    if (!game || !renderer.showScoreboard || e.button !== 0 || !renderer.switchBtn) return;
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top, b = renderer.switchBtn;
    if (x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) requestTeamSwitch();
  });
  window.addEventListener('resize', () => renderer.resize());
  window.addEventListener('beforeunload', () => { if (net) net.destroy(); });
  requestAnimationFrame(frame);
})();
