# HaxMatch

Futebol 2D visto de cima no estilo Haxball, mas com as mecânicas do Rematch:
domínio automático, mira com o mouse, chute com carga e efeito, push ball,
tackle, carrinho, posturas de drible/defesa e goleiro com luvas.

Protótipo local: você controla um jogador, o resto são bots. A simulação é
determinística e separada do render/input, pronta para virar multiplayer com
servidor autoritativo depois.

## Rodar

Abra `index.html` direto no navegador (não precisa de servidor), ou:

```
npx serve .
```

No lobby: **Jogar sozinho** (bots), **Criar sala** (você vira host e recebe um
código de 5 letras) ou **Entrar na sala** com o código de um amigo. Quem entra
ocupa uma vaga, alternando os times. A opção **preencher vagas com bots** decide
se as vagas sem humano têm bot ou ficam vazias (por exemplo, 1v1 puro). Esc
volta ao lobby.

Parâmetros de URL: `?n=3|4|5` (tamanho dos times), `?join=CODIGO` (preenche o
código da sala).

## Jogar com amigos pela internet (GitHub Pages)

O jogo é só HTML/JS estático, então o GitHub Pages hospeda de graça com HTTPS
(necessário para o WebRTC). A conexão entre os jogadores é direta (PeerJS /
WebRTC); o servidor público do PeerJS só faz a apresentação inicial.

1. Crie um repositório no GitHub (por exemplo `haxmatch`) e envie esta pasta:
   ```
   git init
   git add .
   git commit -m "HaxMatch"
   git branch -M main
   git remote add origin https://github.com/SEU_USUARIO/haxmatch.git
   git push -u origin main
   ```
2. No repositório: **Settings → Pages → Build and deployment → Source: Deploy
   from a branch → Branch: main / (root) → Save**.
3. Em 1 a 2 minutos o jogo fica em `https://SEU_USUARIO.github.io/haxmatch/`.
4. Abra, clique em **Criar sala** e mande o link mostrado (`...?join=CODIGO`)
   para o amigo. Cada novo push no `main` atualiza o site sozinho.

Segurança: não há servidor seu nem dados armazenados. Como em qualquer WebRTC,
os participantes da sala veem o IP um do outro. O host é a única fonte da
simulação; convidados só enviam inputs (validados antes de usar).

Arquitetura de rede: `js/net.js` (PeerJS host/convidado), `js/netstate.js`
(snapshot compacto a 30 Hz + extrapolação no convidado). A simulação em
`js/game.js` roda só no host.

Testes sem navegador:

```
node test/headless.js 180 4 7   # bots x bots, procura erros e conta eventos
node test/netstate.js           # snapshot de rede: encode/apply reproduz o estado
```

## Controles

| Ação | Tecla |
|---|---|
| Mover | WASD ou setas |
| Mirar | Mouse |
| Correr | Shift (segurar), com ou sem bola (com bola é mais lento). Shift 2x com a barra pequena cheia = arrancada; com a bola, a arrancada empurra a bola sozinha |
| Chutar | Botão esquerdo (segurar = força; mover o mouse durante a carga = efeito) |
| Passar | Botão direito (segurar = força) |
| Push ball | Espaço com a bola nos pés (correndo vai mais longe) |
| Ação de primeira | Com a bola brilhando (zona de ação), LMB/RMB/Espaço agendam chute/passe/push: o jogador acelera até a bola e executa no toque |
| Postura | Ctrl ou C (drible com bola / defensiva sem bola) |
| Drible / dash | Espaço + direção dentro da postura |
| Tackle | E |
| Carrinho | Shift + E |
| Mergulho (goleiro) | Espaço sem bola dentro da própria área |
| Arremesso (goleiro) | F com a bola nas mãos |
| Pedir bola | Botão do meio |
| Placar e ping | Tab (segurar). Com o Tab aberto, T ou clique no botão troca de time |
| Trocar jogador (solo) | Q |
| Reiniciar / ajuda / pausa | R / H / P |

Atenção: em janela normal o navegador executa Ctrl+W (fecha a aba) antes de
o jogo ver a tecla. Aperte **Enter** para entrar em tela cheia: o jogo usa a
Keyboard Lock API (Chrome/Edge) e passa a receber Ctrl+W, Ctrl+S, Tab etc.
No Firefox isso não existe; use C em vez de Ctrl, ou as setas com o Ctrl direito.

Veja `docs/mecanicas.md` para a pesquisa das mecânicas do Rematch e como cada
uma foi adaptada.
