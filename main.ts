// free-model-aggregator-01 entry: one server, three upstream dialects, one canonical shape.
//
// The shape of a trial is described in the kit's RULES.txt. In one sentence: read the brief, probe every
// provider's catalog, listen, register, serve the client's twenty requests from the strongest free model
// that meets each one, and exit 0 when the env says the client is done.
//
// WHAT THIS ENTRY DOES THAT THE BASELINE DOES NOT
//   - quality (60%): every free candidate that meets a request is ranked by the quality table's
//     rating_milli, so a request starts on the strongest legal model rather than the first one listed.
//   - failover (10%) + served (15%): a refusal (402, 429, any 5xx) is not an answer. The request walks
//     down the ranked list; a provider that refused an inference call is retired for the rest of the trial,
//     and a `concurrent_requests` refusal — which spends no budget — only makes the router wait its turn.
//   - wires: all three upstream dialects are spoken. Anthropic and Gemini providers are normalised into a
//     canonical Chat Completions answer and re-emitted in the client's wire, so an Anthropic client can be
//     served by a Gemini provider and the other way round.
//   - the catalog (5%): unchanged from the baseline, which already reports every listing with the quality
//     table's free flag and capabilities.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  chunkOf,
  usageChunkOf,
  fromAnthropicAnswer,
  fromGeminiAnswer,
  fromGeminiCandidate,
  ssePayloads,
  toAnthropicRequest,
  toGeminiRequest,
} from './lib/canonical.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take before the router moves on, in milliseconds. */
const CALL_TIMEOUT_MS = 25000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

type ClientWire = 'openai' | 'anthropic';

/* ------------------------------------------------------------------ provider state */

/** What one trial has taught this server about one provider. Reset per process, never persisted. */
interface ProviderState {
  /** Refusals this provider answered: enough to stop guessing at its remaining budget. */
  refused: number;
  /** True once it refused an inference call: the rest of the trial treats it as exhausted. */
  retired: boolean;
  /** Calls it served, so the log shows where the twenty answers actually came from. */
  served: number;
}

const state = new Map<string, ProviderState>();

function providerState(id: string): ProviderState {
  let entry = state.get(id);
  if (entry === undefined) {
    entry = { refused: 0, retired: false, served: 0 };
    state.set(id, entry);
  }
  return entry;
}

/** One in-flight inference call per provider: a second one is refused with 429 and spends no budget. */
const locks = new Map<string, Promise<void>>();

async function withProviderLock<T>(id: string, run: () => Promise<T>): Promise<T> {
  const previous = locks.get(id) ?? Promise.resolve();
  let release: () => void = () => {};
  const current = new Promise<void>((resolve) => {
    release = resolve;
  });
  locks.set(id, previous.then(() => current));
  await previous;
  try {
    return await run();
  } finally {
    release();
  }
}

/* ------------------------------------------------------------------ catalog */

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: GET /v1/models. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  context: number;
}

/** Every free model on a provider that answered and that meets the request, strongest first. */
function candidates(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      const listed = typeof model.context_length === 'number' ? model.context_length : 0;
      if (typeof caps.context_length === 'number') caps.context_length = Math.max(caps.context_length, listed);
      if (!meetsRequirementsFor(requires, caps)) continue;
      out.push({
        probe,
        model,
        rating: row?.rating_milli ?? 0,
        context: Math.max(row?.context_length ?? 0, listed),
      });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.context !== b.context) return b.context - a.context;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return byId(a.model, b.model);
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/* ------------------------------------------------------------------ http plumbing */

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/* ------------------------------------------------------------------ upstream calls */

interface Outcome {
  /** The response to keep, when a provider answered; null when every candidate refused. */
  response: Response | null;
  /** The candidate whose provider answered, so the answer keeps its model id. */
  candidate: Candidate | null;
  /** Every refusal seen on the way, as `provider:status`, for the log. */
  refusals: string[];
}

/** Ask one provider for one answer, in its own dialect, and hand back the raw response. */
async function callProvider(candidate: Candidate, canonical: Json): Promise<Response> {
  const { provider } = candidate.probe;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const wire = provider.wire;
  let url: string;
  let payload: Json;
  if (wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = toAnthropicRequest(canonical, candidate.model.id);
  } else if (wire === 'gemini') {
    const verb = canonical.stream === true ? 'streamGenerateContent?alt=sse' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}`;
    payload = toGeminiRequest(canonical);
  } else {
    url = `${base}/v1/chat/completions`;
    payload = { ...canonical, model: candidate.model.id };
    // Ask for the trailing usage chunk: an OpenAI client is scored on the tokens it is told were spent.
    if (canonical.stream === true) payload.stream_options = { include_usage: true };
  }
  if (canonical.stream === true) headers.accept = 'text/event-stream';
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

/**
 * Walk the ranked candidates until one answers. A refusal costs nothing but the time it took, so the loop
 * moves to the next model; a provider that refused an inference call is retired, while a
 * `concurrent_requests` refusal — which spends no budget — is retried after the in-flight call clears.
 */
async function upstream(canonical: Json, requires: Requirement): Promise<Outcome> {
  const ranked = candidates(requires);
  const refusals: string[] = [];
  for (const candidate of ranked) {
    const provider = candidate.probe.provider;
    const stats = providerState(provider.id);
    if (stats.retired) continue;
    let response: Response;
    try {
      response = await withProviderLock(provider.id, () => callProvider(candidate, canonical));
    } catch (err) {
      refusals.push(`${provider.id}:threw`);
      log(`${provider.id}/${candidate.model.id} threw ${(err as Error).message}`);
      continue;
    }
    if (response.ok) {
      stats.served += 1;
      return { response, candidate, refusals };
    }
    const status = response.status;
    const text = await response.text();
    const concurrent = text.includes('concurrent_requests');
    refusals.push(`${provider.id}:${status}${concurrent ? '/concurrent' : ''}`);
    stats.refused += 1;
    if (concurrent) {
      // The provider was busy, not out of budget. After a couple of those it is not worth queueing behind.
      if (stats.refused > 2) stats.retired = true;
      continue;
    }
    if (status === 402 || status === 429 || status >= 500) {
      stats.retired = true;
      continue;
    }
    // A 4xx that is not about budget (unknown model, wrong wire shape) is also not an answer.
    log(`${provider.id}/${candidate.model.id} answered ${status}: ${text.slice(0, 200)}`);
  }
  return { response: null, candidate: null, refusals };
}

/* ------------------------------------------------------------------ streaming */

/** The canonical chunks of one provider stream, whatever dialect it speaks. */
async function* canonicalChunks(
  candidate: Candidate,
  stream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<Json> {
  const wire = candidate.probe.provider.wire;
  const id = `chatcmpl-${candidate.probe.provider.id}`;
  if (wire === 'openai') {
    // Already canonical: pass the provider's own payloads through, and make sure the client is told the
    // token usage — a trailing chunk whose choices are empty carries it.
    let usage: Json | null = null;
    let prompt = 0;
    let completion = 0;
    for await (const payload of ssePayloads(stream)) {
      if (isRecord(payload.usage)) {
        usage = payload.usage;
        prompt = typeof payload.usage.prompt_tokens === 'number' ? payload.usage.prompt_tokens : prompt;
        completion = typeof payload.usage.completion_tokens === 'number' ? payload.usage.completion_tokens : completion;
      }
      yield payload;
    }
    if (usage === null) yield usageChunkOf(id, model, prompt, completion);
    return;
  }
  if (wire === 'anthropic') {
    let finish: string | null = null;
    let sentText = false;
    let toolIndex = 0;
    let prompt = 0;
    let completion = 0;
    for await (const event of ssePayloads(stream)) {
      const type = typeof event.type === 'string' ? event.type : '';
      if (type === 'message_start') {
        const message = isRecord(event.message) ? event.message : {};
        const usage = isRecord(message.usage) ? message.usage : {};
        if (typeof usage.input_tokens === 'number') prompt = usage.input_tokens;
        if (typeof usage.output_tokens === 'number') completion = usage.output_tokens;
        continue;
      }
      if (type === 'content_block_start') {
        const block = isRecord(event.content_block) ? event.content_block : {};
        if (block.type === 'tool_use') {
          yield chunkOf(
            id,
            model,
            {
              tool_calls: [
                {
                  index: toolIndex,
                  id: typeof block.id === 'string' ? block.id : `call_${toolIndex}`,
                  type: 'function',
                  function: { name: typeof block.name === 'string' ? block.name : 'tool', arguments: '' },
                },
              ],
            },
            null,
          );
        }
        continue;
      }
      if (type === 'content_block_delta') {
        const delta = isRecord(event.delta) ? event.delta : {};
        if (typeof delta.text === 'string' && delta.text !== '') {
          sentText = true;
          yield chunkOf(id, model, { content: delta.text }, null);
        } else if (typeof delta.partial_json === 'string') {
          yield chunkOf(id, model, { tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }] }, null);
        }
        continue;
      }
      if (type === 'content_block_stop') {
        toolIndex += 1;
        continue;
      }
      if (type === 'message_delta') {
        const usage = isRecord(event.usage) ? event.usage : {};
        if (typeof usage.output_tokens === 'number') completion = usage.output_tokens;
        if (typeof usage.input_tokens === 'number') prompt = usage.input_tokens;
        const delta = isRecord(event.delta) ? event.delta : {};
        const stop = typeof delta.stop_reason === 'string' ? delta.stop_reason : 'end_turn';
        finish = stop === 'max_tokens' ? 'length' : stop === 'tool_use' ? 'tool_calls' : 'stop';
      }
    }
    if (!sentText) yield chunkOf(id, model, { role: 'assistant', content: '' }, null);
    yield chunkOf(id, model, {}, finish ?? 'stop');
    yield usageChunkOf(id, model, prompt, completion);
    return;
  }
  // gemini
  let finish: string | null = null;
  let emitted = false;
  let prompt = 0;
  let completion = 0;
  for await (const payload of ssePayloads(stream)) {
    const usage = isRecord(payload.usageMetadata) ? payload.usageMetadata : {};
    if (typeof usage.promptTokenCount === 'number') prompt = usage.promptTokenCount;
    if (typeof usage.candidatesTokenCount === 'number') completion = usage.candidatesTokenCount;
    const list = Array.isArray(payload.candidates) ? payload.candidates : [];
    const first = isRecord(list[0]) ? list[0] : {};
    if (Object.keys(first).length === 0) continue;
    const { message, finish: stop } = fromGeminiCandidate(first);
    if (stop !== 'stop') finish = stop;
    const delta: Json = {};
    if (typeof message.content === 'string' && message.content !== '') delta.content = message.content;
    if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) delta.tool_calls = message.tool_calls;
    if (Object.keys(delta).length > 0) {
      emitted = true;
      yield chunkOf(id, model, delta, null);
    }
  }
  if (!emitted) yield chunkOf(id, model, { role: 'assistant', content: '' }, null);
  yield chunkOf(id, model, {}, finish ?? 'stop');
  yield usageChunkOf(id, model, prompt, completion);
}

/** Copy the canonical chunk stream to the client in its own wire. */
async function streamToClient(
  res: ServerResponse,
  chunks: AsyncGenerator<Json>,
  wire: ClientWire,
  model: string,
): Promise<void> {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });
  if (wire === 'anthropic') {
    // The Anthropic event shapes already exist for a canonical stream: feed this one through them.
    const encoded = new ReadableStream<Uint8Array>({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          for await (const chunk of chunks) controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        } catch (err) {
          controller.error(err as Error);
          return;
        }
        controller.close();
      },
    });
    for await (const event of anthropicEventsFromOpenAI(encoded, model)) res.write(event);
    res.end();
    return;
  }
  for await (const chunk of chunks) res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  res.write('data: [DONE]\n\n');
  res.end();
}

/* ------------------------------------------------------------------ routes */

/** Serve one client request: route, fail over, normalise the answer, emit it in the client's wire. */
async function serve(req: IncomingMessage, res: ServerResponse, wire: ClientWire): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body);
  const canonical: Json = wire === 'anthropic' ? anthropicToCanonical(body) : body;
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : '';
  const outcome = await upstream(canonical, requires);
  if (outcome.response === null || outcome.candidate === null) {
    log(`no free model for ${JSON.stringify(requires)} (refusals ${outcome.refusals.join(',')})`);
    if (wire === 'anthropic') {
      sendJson(res, 503, anthropicError('overloaded_error', 'no free model on the directory meets this request'));
    } else {
      sendJson(res, 503, openAIError('no free model meets this request', 'no_free_model', null));
    }
    return;
  }
  const { candidate, response } = outcome;
  const model = candidate.model.id;
  const echoed = requested !== '' ? requested : model;
  if (body.stream === true && response.body !== null) {
    await streamToClient(res, canonicalChunks(candidate, response.body, model), wire, echoed);
    return;
  }
  let answer: Json;
  try {
    const raw: unknown = await response.json();
    answer = isRecord(raw) ? raw : {};
  } catch (err) {
    log(`unreadable answer from ${candidate.probe.provider.id}: ${(err as Error).message}`);
    return sendError(res, wire, 502, 'api_error', 'the provider answer was not JSON');
  }
  if (candidate.probe.provider.wire === 'anthropic') answer = fromAnthropicAnswer(answer, model);
  else if (candidate.probe.provider.wire === 'gemini') answer = fromGeminiAnswer(answer, model);
  if (wire === 'anthropic') sendJson(res, 200, openAIToAnthropic(answer, echoed));
  else sendJson(res, 200, answer);
}

/**
 * The canonical (OpenAI Chat Completions) body of an Anthropic Messages request — the inverse of the
 * adapter in lib/canonical.ts, kept here because it is about what a *client* sent, not what an upstream
 * provider takes.
 */
function anthropicToCanonical(body: Json): Json {
  const messages: Json[] = [];
  const system = body.system;
  const systemText =
    typeof system === 'string'
      ? system
      : Array.isArray(system)
        ? system.map((part) => (isRecord(part) && typeof part.text === 'string' ? part.text : '')).join('\n')
        : '';
  if (systemText !== '') messages.push({ role: 'system', content: systemText });
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(message)) continue;
    const role = typeof message.role === 'string' ? message.role : 'user';
    if (typeof message.content === 'string') {
      messages.push({ role, content: message.content });
      continue;
    }
    let text = '';
    const images: Json[] = [];
    const toolCalls: Json[] = [];
    const toolResults: Json[] = [];
    for (const part of Array.isArray(message.content) ? message.content : []) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') text += part.text;
      else if (part.type === 'image' && isRecord(part.source)) {
        const mime = typeof part.source.media_type === 'string' ? part.source.media_type : 'image/png';
        const data = typeof part.source.data === 'string' ? part.source.data : '';
        images.push({ type: 'image_url', image_url: { url: `data:${mime};base64,${data}` } });
      } else if (part.type === 'tool_use') {
        toolCalls.push({
          id: typeof part.id === 'string' ? part.id : `call_${toolCalls.length}`,
          type: 'function',
          function: {
            name: typeof part.name === 'string' ? part.name : 'tool',
            arguments: JSON.stringify(part.input ?? {}),
          },
        });
      } else if (part.type === 'tool_result') {
        const content = typeof part.content === 'string' ? part.content : JSON.stringify(part.content ?? '');
        toolResults.push({
          role: 'tool',
          tool_call_id: typeof part.tool_use_id === 'string' ? part.tool_use_id : '',
          content,
        });
      }
    }
    if (toolResults.length > 0) {
      messages.push(...toolResults);
      continue;
    }
    if (role === 'assistant') {
      const out: Json = { role: 'assistant', content: text };
      if (toolCalls.length > 0) out.tool_calls = toolCalls;
      messages.push(out);
      continue;
    }
    const content: Json[] = [];
    if (text !== '') content.push({ type: 'text', text });
    content.push(...images);
    messages.push({ role, content: content.length > 0 ? content : '' });
  }
  const tools: Json[] = [];
  for (const tool of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(tool)) continue;
    tools.push({
      type: 'function',
      function: {
        name: typeof tool.name === 'string' ? tool.name : 'tool',
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      },
    });
  }
  const canonical: Json = { messages };
  if (tools.length > 0) canonical.tools = tools;
  if (typeof body.max_tokens === 'number') canonical.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') canonical.temperature = body.temperature;
  if (typeof body.top_p === 'number') canonical.top_p = body.top_p;
  if (body.stream === true) canonical.stream = true;
  if (body.stop_sequences !== undefined) canonical.stop = body.stop_sequences;
  if (body.model !== undefined) canonical.model = body.model;
  return canonical;
}

/** The requirement check, tolerant of a missing rating row: the capability probe's own word then rules. */
function meetsRequirementsFor(requires: Requirement, caps: ReturnType<typeof capabilitiesOf>): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (typeof requires.min_context === 'number' && (caps.context_length ?? 0) < requires.min_context) return false;
  return true;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = (req.url ?? '/').split('?')[0];
  const method = req.method ?? 'GET';
  if (method === 'GET' && url === '/v1/models') return models(req, res);
  if (method === 'POST' && url === '/v1/chat/completions') return serve(req, res, 'openai');
  if (method === 'POST' && url === '/v1/messages') return serve(req, res, 'anthropic');
  sendJson(res, 404, { error: { message: `no route ${method} ${url}`, type: 'not_found' } });
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    // Port 0: the OS picks a free port. 8080 is never asked for, because the env may hold it.
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('no port'));
      else resolve(address.port);
    });
  });
}

/** Upload the catalog document the register body carries, exactly as `buildCatalog` made it. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const response = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
  });
  const text = await response.text();
  log(`register ${response.status}: ${text.slice(0, 200)}`);
}

/** Poll GET /work and exit 0 when the env says the client is done. */
function poll(server: Server): void {
  const tick = (): void => {
    void (async () => {
      try {
        const response = await fetch(`${ENV_URL}/work`);
        const body: unknown = await response.json();
        if (isRecord(body) && body.done === true) {
          server.close(() => process.exit(0));
          setTimeout(() => process.exit(0), 1000);
          return;
        }
      } catch {
        // A missed poll is not a failure: try again on the next tick.
      }
      setTimeout(tick, POLL_INTERVAL_MS);
    })();
  };
  setTimeout(tick, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log(`handler failed: ${(err as Error).message}`);
      if (!res.headersSent) sendJson(res, 500, { error: { message: 'internal', type: 'api_error' } });
      else res.end();
    });
  });
  const port = await listen(server);
  log(`listening on ${port}, ${catalog.providers.length} providers`);
  await register(port, catalog);
  poll(server);
}

if (process.argv.includes('--list')) {
  void load().then(() => {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    process.exit(0);
  });
} else {
  void start().catch((err: unknown) => {
    log(`fatal: ${(err as Error).message}`);
    process.exit(1);
  });
}
