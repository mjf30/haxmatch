# Mecânicas do Rematch → adaptação 2D (HaxMatch)

Resumo do que foi pesquisado sobre o Rematch (Sloclap) e como cada mecânica foi
traduzida para um jogo 2D visto de cima, sem altura (sem lobs, cabeçadas, voleios
ou chapéus).

## Regras gerais do Rematch

- Arena fechada por paredes em todos os lados: a bola quica, não existe lateral,
  escanteio ou tiro de meta.
- Sem faltas, sem impedimento, sem cartões, sem árbitro. O jogo nunca para,
  exceto após gol (reinício no centro).
- Partidas de 6 minutos, modos 3v3 / 4v4 / 5v5, cada pessoa controla 1 jogador.
  Regra de misericórdia: 4 gols de vantagem encerra.
- O primeiro jogador do time a entrar na própria área após o kickoff recebe as
  "luvas" e vira goleiro. Ele pode sair da área (vira líbero) mas só usa as mãos
  dentro dela. Goleiro tem stamina infinita no próprio campo.

## Controles padrão no PC (referência)

| Ação | Tecla |
|---|---|
| Mover | WASD |
| Mirar / câmera | Mouse |
| Sprint | Shift (segurar) |
| Extra Effort (arrancada) | Shift duas vezes, com a barra pequena cheia |
| Chutar | LMB (segurar = mais força; mover o mouse durante a carga = efeito) |
| Passe rasteiro ("Tap") | RMB (segurar = mais força; não é guiado, vai na direção mirada) |
| Empurrar a bola (push ball) | Espaço, com a bola nos pés enquanto corre |
| Postura de drible / defensiva | Ctrl esquerdo (segurar; muda conforme ter ou não a bola) |
| Drible (na postura de drible) | Espaço + direção |
| Dash (na postura defensiva) | Espaço + direção |
| Tackle em pé | E |
| Carrinho | E enquanto corre (Shift) |
| Mergulho (goleiro) / dive de linha | Espaço |
| Pedir bola | Botão do meio |

## Mecânica por mecânica

### Domínio da bola
- Ao encostar na bola ela gruda no pé automaticamente (não existe "primeiro toque"
  manual).
- Com a bola nos pés o jogador é mais lento. Por isso o jogo exige empurrar a bola
  para frente e correr atrás dela.
- Chutes de força alta NÃO são dominados: quicam no corpo de quem está na frente.
  A postura defensiva permite dominar bolas mais fortes.

**2D:** contato círculo-círculo gruda a bola em um ponto à frente do jogador
(na direção do movimento). Velocidade com bola reduzida. Bola acima de um limiar
de velocidade rebate no corpo em vez de grudar; na postura defensiva o limiar é
maior; goleiro dentro da área pega qualquer bola.

### Zona de ação (ação de primeira)
- Quando a bola solta chega perto o bastante do jogador ela brilha. Apertar chute,
  passe ou push nesse momento não executa nada de imediato: agenda a ação. O
  jogador acelera um pouco em direção à bola e, no instante do primeiro toque,
  executa a ação sem dominar.

**2D:** bola solta a até `ACTION_RADIUS` do corpo brilha; LMB/RMB/Espaço criam uma
ação agendada (chute continua carregando enquanto o botão está segurado); o
movimento é redirecionado para a bola com velocidade ×1,3; no contato a ação
dispara. Sair da zona ou a bola ser dominada por outro cancela.

### Push ball (empurrar e correr)
- Espaço enquanto corre com a bola: o personagem toca a bola para frente, ela sai
  dos pés e ele corre atrás em velocidade de sprint. Correndo o toque vai mais
  longe do que andando.
- Enquanto a bola não está nos pés, o portador não pode sofrer tackle em pé (a
  bola está "solta"); qualquer um pode disputá-la.
- Pode empurrar para o lado ou para trás (cutback/chop) para fugir de carrinho.

**2D:** a bola sai com velocidade fixa (maior correndo do que andando). O jogador
só pode recuperá-la após um pequeno cooldown. Correr (Shift) com a bola nos pés é
permitido, mais lento do que sem bola e gastando stamina. Extra Effort com a bola
nos pés empurra a bola automaticamente enquanto a arrancada dura.

### Chute
- Segurar LMB carrega força. Carga máxima ≈ 0,62 s (32 frames de carga + 5 de
  travamento). Chute forte é um compromisso: o jogador fica travado.
- A bola vai em direção à mira no momento do chute.
- Efeito (Magnus): mover o mouse para os lados enquanto segura o chute curva a
  bola para esse lado. Chutes mais fracos aceitam mais efeito.
- Chute também serve como passe longo.
- Fake shot: cancelar a carga vira finta.

**2D:** direção fixada no momento em que LMB é pressionado; o deslocamento angular
do mouse durante a carga vira spin lateral; força escala com o tempo segurado;
spin efetivo diminui com a força. Preview da trajetória curva é desenhado.

### Passe
- RMB = "tap", passe rasteiro na direção mirada; não é guiado automaticamente.
  Segurar dá mais força.

**2D:** passe na direção do mouse com força por tempo segurado. Pequena
assistência: se um companheiro está a poucos graus da mira, a força é ajustada
para chegar nele.

### Postura de drible (Ctrl com bola)
- Personagem agacha, mira vira um círculo, fica mais lento, mas mais difícil de
  sofrer tackle. Espaço + direção executa um drible curto naquela direção.
- Abusar deixa você lento e fácil de alcançar por trás/lados.

**2D:** velocidade 0,7×; bola mantida mais perto e "escondida" do adversário mais
próximo; tackle em pé tem chance reduzida de sucesso; Espaço + direção = passo
curto rápido com a bola grudada (invulnerável durante o passo).

### Postura defensiva (Ctrl sem bola)
- Personagem "enquadra" a bola, domina bolas fortes, e Espaço vira um dash curto
  lateral para interceptar.

**2D:** velocidade 0,8×; hitbox para a bola maior (sem a postura vale só o corpo); domina chutes fortes; Espaço + direção = dash curto. Encostar no
portador nunca rouba a bola: só tackle, carrinho ou disputar a bola quando ela
está solta (após um push, por exemplo). Exceção: com a postura defensiva ativa,
se a bola dominada pelo adversário for passar por dentro de você, ela sai do
domínio dele e fica solta (bloqueio de corpo).

### Tackle em pé (E)
- Investida curta para tirar a bola do pé do adversário. Baixo risco: recuperação
  rápida se errar, mas (desde a Season 4) errar deixa o controle mais lento por
  um instante.

**2D:** lunge de ~0,28 s; se o alcance toca a bola do adversário, rouba (chance
menor contra postura de drible, zero durante o drible/dash e com a bola solta à
frente após um push). Erro = recuperação com movimento reduzido.

### Carrinho (Shift + E)
- Cobre muita distância; se erra custa muita stamina e tempo (fica caído).
- Bom contra quem corre com a bola empurrada à frente.

**2D:** deslize de ~0,5 s; bola tocada é chutada para longe; o adversário só cai
se estiver com a bola dominada no pé (sem bola, o carrinho não derruba).
Recuperação longa, mais longa e cara se errar.

### Dive / jump block de linha
- No Rematch existe um pulo para bloquear bolas altas. Em 2D sem altura não faz
  sentido, então foi descartado: Espaço sem bola e sem postura não faz nada.

### Sprint, stamina e Extra Effort
- Shift gasta a barra grande. Carrinho/tackles perdidos também gastam.
- Barra pequena: quando cheia, Shift duas vezes dá arrancada curta e
  "prioridade" em disputas.

**2D:** implementado com barra grande (sprint) e barra pequena (extra effort com
recarga lenta). Goleiro não gasta stamina no próprio campo.

### Goleiro
- Luvas para o primeiro a entrar na área após o kickoff.
- Espaço = mergulho na direção segurada; hitbox foi reduzida em patches (menos
  "imã"). Pode defender com o corpo só andando na frente da bola. Segurar Ctrl =
  postura de defesa para bola que vem reta.
- Bola nas mãos dentro da área: ninguém pode tirar. Sair da área com a bola → ela
  vai para os pés e pode ser roubada. Existe limite de tempo com a bola nas mãos
  (Season 3) para forçar a saída.
- Distribuição: LMB chutão (com efeito), RMB passe rasteiro, arremesso, Espaço =
  soltar e conduzir. E = sair jogando/varrer (tackle/carrinho).

**2D:** luvas atribuídas ao primeiro na área; mergulho com raio de catch limitado;
bolas muito fortes são espalmadas em vez de agarradas; segurar até 5 s; posse nas
mãos intocável; sair da área solta para os pés. Com a bola nas mãos existe uma
zona de repulsão que empurra adversários para longe (não vale com a bola no pé,
como num recuo).

### Fora do escopo 2D (por decisão)
- Lob, cabeçada, voleio, chapéu (rainbow flick), jump block, bola por cima.

## Fontes consultadas

- https://gamerant.com/all-every-rematch-mechanic-explained/
- https://www.playrematch.com/post/feature-focus-shoot-mechanics
- https://www.destructoid.com/all-controls-for-rematch/
- https://www.dualshockers.com/rematch-how-to-play-defender/
- https://www.dualshockers.com/rematch-how-to-play-goalkeeper/
- https://www.thegamer.com/rematch-defense-pro-tips-goalkeeper-guide/
- https://gamerant.com/rematch-how-avoid-tackles-dribble/
- https://gameranx.com/features/id/544436/article/rematch-how-to-dribble-and-enter-dribble-stance/
- https://www.gamer.org/all-rematch-hidden-mechanics-you-should-know/
- https://www.gamer.org/rematch-season-4-patch-4-dribbling-tackles-and-goalkeeper-changes-explained/
- https://www.playrematch.com/post/patch-5-patch-notes-1-201-100
- https://www.playrematch.com/season-3
- https://www.playrematch.com/season-2
- https://insider-gaming.com/how-to-dribble-in-rematch/
- https://www.gamespot.com/reviews/rematch-review-unbelievable-tekkers/1900-6418378/
