# free-model-router

A single HTTP server that fronts ten free-tier model providers and answers every client request from the
strongest free model that can serve it — in the client's own wire.

Entry for [make.fun](https://make.fun) competition `free-model-aggregator-01`.

## What it does

* probes each provider's catalog in the provider's own dialect (`/v1/models`, `/v1beta/models`) and builds
  one catalog document, uploaded at register time and served to clients at `GET /v1/models`;
* ranks every free candidate that meets a request's requirements (tools, vision, JSON mode, context) by the
  quality table's rating, so each request starts on the strongest legal model;
* treats a refusal as a routing signal, not an answer: `402`, `429` and every `5xx` move the request down the
  ranked list, a provider that refused an inference call is retired for the trial, and a
  `concurrent_requests` refusal only waits its turn behind the provider's in-flight call;
* speaks all three upstream wires — OpenAI Chat Completions, Anthropic Messages and Gemini
  `generateContent`/`streamGenerateContent` — normalises every answer into one canonical shape, and re-emits
  it in the wire the client used, streaming included: text deltas, tool-call deltas, stop reason and the
  trailing token usage;
* exits `0` when the environment reports the client is done.

## Layout

| path | role |
| --- | --- |
| `main.ts` | the server: catalog, routes, ranking, failover, register and exit |
| `lib/canonical.ts` | the adapter matrix: canonical request/answer ⇄ Anthropic and Gemini |
| `lib/providers.ts` | the brief's types and parser, auth headers, catalog probe |
| `lib/wire.ts` | Anthropic ⇄ Chat Completions translation for the client side, SSE writers |

## Run

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=./prompt.txt \
node main.ts
```

`node main.ts --list` prints the catalog document and exits without listening.

