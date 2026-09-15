'use strict';
// Todas as constantes de gameplay. Unidades: pixels de mundo e segundos.
const CFG = {
  FIELD_W: 1800, FIELD_H: 1000,     // área jogável entre as paredes
  GOAL_W: 280, GOAL_D: 80,          // boca do gol e profundidade da rede
  BOX_W: 330, BOX_H: 600,           // área do goleiro (onde pode usar as mãos)
  PLAYER_R: 17, BALL_R: 9,
  TEAM_SIZE: 4,                     // 3, 4 ou 5
  MATCH_TIME: 6 * 60,
  MERCY: 4,                         // diferença de gols que encerra
  DT: 1 / 60,

  // ---- movimento ----
  SPEED: 165, SPRINT: 255, EXTRA_EFFORT: 1.22, ACCEL: 1200, DECEL: 1500,
  SPEED_BALL: 135,                  // andando com a bola nos pés
  SPRINT_BALL: 200,                 // correndo com a bola nos pés (gasta stamina)
  MUL_DRIBBLE: 0.7, MUL_DEF: 0.8, MUL_CHARGE: 0.55, MUL_RECOVER: 0.5,
  KEEPER_HOLD_SPEED: 115,

  // ---- stamina ----
  STAMINA_MAX: 100, STAMINA_SPRINT: 12, STAMINA_REGEN: 9,
  COST_TACKLE: 6, COST_SLIDE: 18, COST_SLIDE_MISS: 15, COST_DASH: 8,
  EFFORT_RECHARGE: 12, EFFORT_DUR: 0.8, DOUBLE_TAP: 0.3,

  // ---- bola ----
  BALL_DRAG: 0.7, BALL_DECEL: 35, BALL_BOUNCE: 0.72, BALL_MAX: 1050,
  SPIN_DECAY: 1.3, SPIN_MAX: 1.7,
  CARRY_K: 14, CARRY_MAX_REL: 260, CARRY_LERP: 18, CARRY_DIST: 3, CARRY_LOSE: 34,   // condução física da bola
  PASS_THROUGH: 0.25,               // bola atravessa o corpo de quem acabou de chutar
  POST_R: 6,

  // ---- domínio ----
  ACTION_RADIUS: 46,                // zona de ação: bola solta a essa distância do corpo permite agendar chute/passe/push de primeira
  ACTION_ZONE_SPEED: 1.3,           // multiplicador de velocidade indo até a bola com ação agendada
  CONTROL_MAX: 600,                 // acima disso a bola rebate no corpo
  CONTROL_MAX_DEF: 850,            // postura defensiva domina bolas mais fortes
  GRAB_MARGIN: 4, GRAB_MARGIN_DEF: 10,
  DEFLECT_BOUNCE: 0.4,
  KICK_COOLDOWN: 0.35, PUSH_COOLDOWN: 0.14, DEFLECT_COOLDOWN: 0.2,

  // ---- chute ----
  CHARGE_MAX: 0.62, SHOT_MIN: 400, SHOT_MAX: 950,
  SPIN_GAIN: 3.2, SPIN_POWER_FADE: 0.6,
  // ---- passe ----
  PASS_CHARGE: 0.4, PASS_MIN: 320, PASS_MAX: 620, PASS_ASSIST_DEG: 9,
  // ---- push ball ----
  PUSH_SPEED: 360, PUSH_WALK_SPEED: 210,   // velocidade absoluta do toque à frente (correndo / andando)
  // ---- tackle em pé ----
  TACKLE_DUR: 0.28, TACKLE_SPEED: 320, TACKLE_REACH: 12, TACKLE_CD: 0.8,
  TACKLE_MISS_RECOVER: 0.45, TACKLE_CHANCE: 0.85, TACKLE_CHANCE_DRIBBLE: 0.4,
  // ---- carrinho ----
  SLIDE_DUR: 0.5, SLIDE_SPEED: 400, SLIDE_REACH: 14, SLIDE_CD: 1.5,
  SLIDE_RECOVER: 0.6, SLIDE_MISS_RECOVER: 0.9, SLIDE_KNOCK: 380, FALL_DUR: 0.85,
  // ---- dash (postura defensiva) e drible (postura de drible) ----
  DASH_DUR: 0.16, DASH_MUL: 2.6, DASH_CD: 0.6,
  DRIBBLE_DUR: 0.2, DRIBBLE_MUL: 2.4, DRIBBLE_CD: 0.7,
  // ---- goleiro ----
  GK_DIVE_DUR: 0.42, GK_DIVE_SPEED: 450, GK_DIVE_REACH: 14, GK_DIVE_RECOVER: 0.5, GK_DIVE_CD: 0.9,
  GK_PARRY_SPEED: 800, GK_HOLD_MAX: 5, GK_THROW: 560, GK_THROW_ASSIST_DEG: 20,
  GK_REPEL: 80, GK_REPEL_PUSH: 420,  // zona de repulsão com a bola nas mãos (raio além do corpo, px/s de empurrão)

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
