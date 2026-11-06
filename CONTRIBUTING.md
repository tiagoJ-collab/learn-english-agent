# Contribuindo

Obrigado pelo interesse. Issues e pull requests são bem-vindos.

## Antes de abrir um PR

1. Abra uma issue descrevendo o problema ou a ideia, se a mudança for grande.
2. Crie uma branch a partir da `main`.
3. Rode os testes com `pytest`. Eles não precisam de rede nem de modelos.
4. Se mudou a interface, rode `npm run lint` e `npm run build` em `frontend/` e inclua o `app/static/` gerado no commit.

## Regras

- Nunca envie o arquivo `.env` nem chaves de API.
- Não envie nada da pasta `data/`: banco, modelos e vozes ficam fora do repositório.
- Ao contribuir, você concorda que seu código será distribuído sob a licença [GPL-3.0-or-later](LICENSE).
