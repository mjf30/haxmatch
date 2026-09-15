'use strict';
// Todas as constantes de gameplay. Unidades: pixels de mundo e segundos.
const CFG = {
  FIELD_W: 2000, FIELD_H: 1150,     // área jogável entre as paredes
  GOAL_W: 300, GOAL_D: 80,          // boca do gol e profundidade da rede
  BOX_W: 360, BOX_H: 640,           // área do goleiro (onde pode usar as mãos)
  PLAYER_R: 15, BALL_R: 8,
  TEAM_SIZE: 4,                     // 3, 4 ou 5
  MATCH_TIME: 6 * 60,
  MERCY: 4,                         // diferença de gols que encerra
  DT: 1 / 60,

  // ---- movimento ----
  SPEED: 135, SPRINT: 210, EXTRA_EFFORT: 1.22, ACCEL: 1000, DECEL: 1300,
  SPEED_BALL: 110,                  // andando com a bola nos pés
  SPRINT_BALL: 165,                 // correndo com a bola nos pés (gasta stamina)
  MUL_DRIBBLE: 0.7, MUL_DEF: 0.8, MUL_CHARGE: 0.55, MUL_RECOVER: 0.5,
  KEEPER_HOLD_SPEED: 95,

  // ---- stamina ----
  STAMINA_MAX: 60, STAMINA_SPRINT: 12, STAMINA_REGEN: 16,   // ~5 s de sprint, recupera em ~4 s
  STAMINA_REGEN_DELAY: 1.5,         // ao zerar, espera isso antes de começar a regenerar
  EXHAUST_RECOVER: 0.35,            // fração da barra para sair do estado exausto
  MUL_EXHAUSTED: 0.75,              // velocidade enquanto exausto
  COST_TACKLE: 5, COST_SLIDE: 14, COST_SLIDE_MISS: 12, COST_DASH: 6, COST_DRIBBLE: 10,
  EFFORT_RECHARGE: 12, EFFORT_RECHARGE_FULL: 5, EFFORT_DUR: 0.8, DOUBLE_TAP: 0.3,   // recarga da arrancada (s): normal / com a stamina cheia

  // ---- bola ----
  BALL_DRAG: 0.7, BALL_DECEL: 35, BALL_BOUNCE: 0.72, BALL_MAX: 1300,
  SPIN_DECAY: 1.3, SPIN_MAX: 1.7,
  CARRY_K: 14, CARRY_MAX_REL: 260, CARRY_LERP: 18, CARRY_DIST: 3, CARRY_LOSE: 34,   // condução física da bola
  PASS_THROUGH: 0.25,               // bola atravessa o corpo de quem acabou de chutar
  POST_R: 6,

  // ---- domínio ----
  ACTION_RADIUS: 70,                // alcance do "alvo": bola solta a essa distância permite travar chute/passe/push de primeira
  LOCK_SNAP_SPEED: 210,             // velocidade do snap até a bola com ação travada (= sprint, nunca mais rápido que correr)
  LOCK_LOSE_RECOVER: 0.25,          // quem perde a prioridade numa disputa fica esse tempo com controle reduzido
  LOCK_MAX: 0.8,                    // trava expira se a ação não sair nesse tempo (janela do Rematch é 0,3–0,6 s)
  CONTROL_MAX: 600,                 // acima disso a bola rebate no corpo
  CONTROL_MAX_DEF: 850,            // postura defensiva domina bolas mais fortes
  HITBOX_MUL: 1.0,                  // sem postura defensiva, a hitbox para a bola é só o corpo
  GRAB_MARGIN: 0, GRAB_MARGIN_DEF: 14,   // postura defensiva: alcance bem maior para dominar/bloquear
  DEFLECT_BOUNCE: 0.4,
  KICK_COOLDOWN: 0.35, PUSH_COOLDOWN: 0.14, DEFLECT_COOLDOWN: 0.2,

  // ---- chute ----
  CHARGE_MAX: 0.62, SHOT_MIN: 480, SHOT_MAX: 1200,
  SPIN_GAIN: 3.2, SPIN_POWER_FADE: 0.6,
  // ---- passe ----
  PASS_CHARGE: 0.4, PASS_MIN: 320, PASS_MAX: 620, PASS_ASSIST_DEG: 9,
  // ---- push ball ----
  PUSH_SPEED: 300, PUSH_WALK_SPEED: 175,   // velocidade absoluta do toque à frente (correndo / andando)
  // ---- tackle em pé ----
  TACKLE_DUR: 0.28, TACKLE_SPEED: 270, TACKLE_REACH: 12, TACKLE_CD: 0.8,
  TACKLE_MISS_RECOVER: 0.45, TACKLE_CHANCE: 0.85, TACKLE_CHANCE_DRIBBLE: 0.4,
  // ---- carrinho ----
  SLIDE_DUR: 0.5, SLIDE_SPEED: 340, SLIDE_REACH: 14, SLIDE_CD: 1.5,
  SLIDE_RECOVER: 0.6, SLIDE_MISS_RECOVER: 0.9, SLIDE_KNOCK: 380, FALL_DUR: 0.85,
  // ---- dash (postura defensiva) e drible (postura de drible) ----
  DASH_DUR: 0.16, DASH_MUL: 2.6, DASH_CD: 0.6,
  DRIBBLE_DUR: 0.14, DRIBBLE_MUL: 2.3,                      // drible na postura: passo curto
  DRIBBLE_CHAIN_WINDOW: 0.45,       // após o 1º drible, tempo para encadear o 2º ("roleta"); toque durante o 1º fica bufferizado
  DRIBBLE_LAG: 0.55,                // lag após o 2º drible (cancelável com chute, passe ou push)
  DRIBBLE_LAG_MUL: 0.35,            // velocidade durante o lag
  DRIBBLE_CD: 1.5, DRIBBLE_CD_SINGLE: 0.9,   // cooldown após a sequência de 2 / após um drible só
  // ---- goleiro ----
  GK_DIVE_DUR: 0.42, GK_DIVE_SPEED: 350, GK_DIVE_REACH: 10, GK_DIVE_RECOVER: 0.6, GK_DIVE_CD: 1.0,
  GK_PARRY_SPEED: 900, GK_HOLD_MAX: 5, GK_THROW: 560, GK_THROW_ASSIST_DEG: 20,
  GK_REPEL: 80, GK_REPEL_PUSH: 420,  // zona de repulsão com a bola nas mãos (raio além do corpo, px/s de empurrão)

  BOT_REACTION: 0.28,               // tempo de reação dos bots a mudanças de posse (s, com variação aleatória)
  BOT_REACTION_GK: 0.22,
  KICKOFF_FREEZE: 1.2, GOAL_PAUSE: 2.2,
  TEAM_COLORS: ['#e94b3c', '#3c8ee9'],
  TEAM_NAMES: ['Vermelho', 'Azul'],
};

// Posições iniciais em fração de meio campo (x: -1 = própria linha de fundo,
// y: -1..1 = largura) para o time que ataca para +x; o outro é espelhado.
// O índice 0 nasce dentro da área e recebe as luvas.
const FORMATIONS = {
  3: [{ x: -0.9, y: 0 }, { x: -0.4, y: -0.35 }, { x: -0.4, y: 0.35 }],
  4: [{ x: -0.9, y: 0 }, { x: -0.5, y: 0 }, { x: -0.27, y: -0.5 }, { x: -0.27, y: 0.5 }],
  5: [{ x: -0.9, y: 0 }, { x: -0.55, y: -0.4 }, { x: -0.55, y: 0.4 }, { x: -0.25, y: -0.55 }, { x: -0.25, y: 0.55 }],
};
