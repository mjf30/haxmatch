# Estado do projeto (16/09/2026) — treinos pausados

## Jogo
- Script v2.7 com estilos (equilibrado, posse curta, jogo longo, vertical, controle de campo/EPV), cruzamento, reflexos, mapa de valor do campo (js/value_map.js), controle de campo por tempo de chegada (Features.pitchControlTT), mapa de passe (Features.passMap), camada de análise (tecla V: território / perigo).
- Lobby: script (+estilo), ES50 + reflexos (js/nn_weights.js), ES nova geração 75 (js/nn_macro_ppo.js), PPO híbrido guiado 4v4 (js/nn_guided.js), PPO híbrido guiado 3v3/4v4/5v5 (js/nn_guided_ms.js), PPO controle total guiado (js/nn_raw_weights.js), ES50 com a bola + script sem a bola.

## Melhor ranking medido (torneio por tamanho, 45 partidas por versão)
ES50+reflexos 79 pts, PPO guiado 4v4 67, ES nova g75 52, script vertical 51, script equilibrado 46, PPO guiado 3-5 (it 200) 46.

## Treinos (GPU) — pausados; retomar com --resume
- Híbrido guiado 4v4: train_gpu/ckpt_guided.pt (guia 0.6 decaindo até it 1400, piso 0.05; liga com ES50 e ES75 fixas, fixed_share 0.5).
- Híbrido guiado 3v3/4v4/5v5: train_gpu/ckpt_guided_ms.pt (guia 0.6 decaindo até it 700; mesma liga).
- Controle total guiado: train_gpu/ckpt.pt (lr 2.5e-4, guia 1.0 decaindo até it 3000, peso 4x nos desarmes; v10 fixa na liga).
- Comandos exatos de retomada: ver o histórico do git (mensagens dos commits) e os logs em %TMP%\ppo_guided8.log, ppo_guided_ms6.log, ppo23.log.
- Parada suave: criar o arquivo <ckpt>.stop (código já no ppo.py; processos antigos não têm).

## Pendências
- Portar para torch: estilos, cruzamento, reflexos, controle de campo por tempo de chegada (o torch ainda executa o script v2.3 com as notas ajustadas).
- Padrões coordenados curtos (terceiro homem, corte para dentro, sobrecarga e inversão) com árbitro fuzzy e, depois, aprendido.
- Avaliação da ES mais ampla (mais partidas, adversários variados, três tamanhos) antes de novas rodadas.
