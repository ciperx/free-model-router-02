// The adapter layer: everything crossing a wire is turned into one canonical shape here, so the router
// never has to think in three dialects.
//
// The canonical request is an OpenAI Chat Completions body, and the canonical answer is an OpenAI Chat
// Completion (or the same thing arriving as `data:` chunks). Client wires are then only a matter of
// emitting the canonical answer in the shape the client asked for, and provider wires only a matter of
// asking for it in the provider's own dialect.
//
//   client openai    <- canonical                          (as is)
//   client anthropic <- canonical                          openAIToAnthropic / anthropicEventsFromOpenAI
//   provider openai  <- canonical                          (as is)
//   provider anthropic <- canonical                        toAnthropicRequest / fromAnthropic
//   provider gemini  <- canonical                          toGeminiRequest / fromGemini
//
// Nothing here reads the clock or a random source: the same request and the same provider answers always
// produce the same bytes, which is what the replayed export is checked against.

import { isRecord, type Json } from './providers.ts';

/* ------------------------------------------------------------------ small readings */

const asString = (value: unknown): string | null => (typeof value === 'string' ? value : null);

const asNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

/** The text of a canonical (OpenAI-shaped) message, flattening the string-or-block-parts form. */
export function textOfContent(content: unknown): string {
  const direct = asString(content);
  if (direct !== null) return direct;
  let out = '';
  for (const part of asArray(content)) {
    if (!isRecord(part)) continue;
    const text = asString(part.text);
    if (text !== null) out += text;
  }
  return out;
}

/** Every content part of a canonical message that carries an image, as {mime, data} in base64. */
function imagesOfContent(content: unknown): { mime: string; data: string }[] {
  const out: { mime: string; data: string }[] = [];
  for (const part of asArray(content)) {
    if (!isRecord(part)) continue;
    if (part.type === 'image_url' && isRecord(part.image_url)) {
      const url = asString(part.image_url.url) ?? '';
      const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
      if (match !== null) out.push({ mime: match[1], data: match[2] });
    }
  }
  return out;
}

/* ------------------------------------------------------------------ canonical -> anthropic request */

interface AnthropicBlock {
  type: string;
  text?: string;
  source?: { type: string; media_type: string; data: string };
  id?: string;
  name?: string;
  input?: unknown;
  tool_use_id?: string;
  content?: unknown;
}

interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: AnthropicBlock[];
}

function anthropicParts(content: unknown): AnthropicBlock[] {
  const text = asString(content);
  if (text !== null) return [{ type: 'text', text }];
  const blocks: AnthropicBlock[] = [];
  for (const part of asArray(content)) {
    if (!isRecord(part)) continue;
    if (part.type === 'text') {
      const value = asString(part.text) ?? '';
      if (value !== '') blocks.push({ type: 'text', text: value });
    } else if (part.type === 'image_url' && isRecord(part.image_url)) {
      const url = asString(part.image_url.url) ?? '';
      const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
      if (match !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: match[1], data: match[2] } });
    }
  }
  return blocks;
}

/** Translate one canonical (OpenAI Chat Completions) request into an Anthropic Messages request. */
export function toAnthropicRequest(body: Json, model: string): Json {
  const messages: AnthropicMessage[] = [];
  const systemParts: string[] = [];
  for (const message of asArray(body.messages)) {
    if (!isRecord(message)) continue;
    const role = asString(message.role) ?? 'user';
    if (role === 'system' || role === 'developer') {
      const text = textOfContent(message.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      // A tool result is a user turn that carries a tool_result block.
      const block: AnthropicBlock = {
        type: 'tool_result',
        tool_use_id: asString(message.tool_call_id) ?? '',
        content: textOfContent(message.content),
      };
      const last = messages[messages.length - 1];
      if (last !== undefined && last.role === 'user') last.content.push(block);
      else messages.push({ role: 'user', content: [block] });
      continue;
    }
    if (role === 'assistant') {
      const blocks = anthropicParts(message.content);
      for (const call of asArray(message.tool_calls)) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: unknown = {};
        const raw = asString(call.function.arguments) ?? '';
        if (raw !== '') {
          try {
            input = JSON.parse(raw);
          } catch {
            input = {};
          }
        }
        blocks.push({
          type: 'tool_use',
          id: asString(call.id) ?? 'call_0',
          name: asString(call.function.name) ?? 'tool',
          input,
        });
      }
      if (blocks.length === 0) continue;
      const last = messages[messages.length - 1];
      if (last !== undefined && last.role === 'assistant') last.content.push(...blocks);
      else messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks = anthropicParts(message.content);
    if (blocks.length === 0) continue;
    messages.push({ role: 'user', content: blocks });
  }

  const tools: Json[] = [];
  for (const tool of asArray(body.tools)) {
    if (!isRecord(tool)) continue;
    const fn = isRecord(tool.function) ? tool.function : tool;
    tools.push({
      name: asString(fn.name) ?? 'tool',
      description: asString(fn.description) ?? '',
      input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
    });
  }

  const maxTokens = asNumber(body.max_completion_tokens) ?? asNumber(body.max_tokens) ?? 1024;
  const out: Json = { model, max_tokens: Math.max(1, Math.trunc(maxTokens)), messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n\n');
  if (tools.length > 0) out.tools = tools;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (body.stream === true) out.stream = true;
  return out;
}

/* ------------------------------------------------------------------ anthropic -> canonical answer */

function canonicalFromAnthropicContent(content: unknown, index: number): Json {
  let text = '';
  const toolCalls: Json[] = [];
  let callIndex = 0;
  for (const block of asArray(content)) {
    if (!isRecord(block)) continue;
    if (block.type === 'text') {
      text += asString(block.text) ?? '';
    } else if (block.type === 'tool_use') {
      toolCalls.push({
        index: callIndex,
        id: asString(block.id) ?? `call_${callIndex}`,
        type: 'function',
        function: { name: asString(block.name) ?? 'tool', arguments: JSON.stringify(block.input ?? {}) },
      });
      callIndex += 1;
    }
  }
  const usage = isRecord(content) ? content : {};
  void usage;
  const out: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) out.tool_calls = toolCalls;
  if (index >= 0) out.index = index;
  return out;
}

const canonicalStop = (stop: unknown): string => {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  return 'stop';
};

/** Translate one Anthropic Messages answer into a canonical (OpenAI Chat Completion) answer. */
export function fromAnthropicAnswer(answer: Json, requestedModel: string): Json {
  const message = canonicalFromAnthropicContent(answer.content, -1);
  delete message.index;
  const usage = isRecord(answer.usage) ? answer.usage : {};
  return {
    id: asString(answer.id) ?? 'chatcmpl-anthropic',
    object: 'chat.completion',
    created: 0,
    model: asString(answer.model) ?? requestedModel,
    choices: [{ index: 0, message, finish_reason: canonicalStop(answer.stop_reason) }],
    usage: {
      prompt_tokens: asNumber(usage.input_tokens) ?? 0,
      completion_tokens: asNumber(usage.output_tokens) ?? 0,
      total_tokens: (asNumber(usage.input_tokens) ?? 0) + (asNumber(usage.output_tokens) ?? 0),
    },
  };
}

/* ------------------------------------------------------------------ canonical -> gemini request */

function geminiParts(content: unknown): Json[] {
  const parts: Json[] = [];
  const text = asString(content);
  if (text !== null) {
    if (text !== '') parts.push({ text });
    return parts;
  }
  for (const part of asArray(content)) {
    if (!isRecord(part)) continue;
    if (part.type === 'text') {
      const value = asString(part.text) ?? '';
      if (value !== '') parts.push({ text: value });
    } else if (part.type === 'image_url' && isRecord(part.image_url)) {
      const url = asString(part.image_url.url) ?? '';
      const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
      if (match !== null) parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
    }
  }
  for (const image of imagesOfContent(content)) parts.push({ inlineData: { mimeType: image.mime, data: image.data } });
  return parts;
}

/** Translate one canonical request into a Gemini generateContent request body. */
export function toGeminiRequest(body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  for (const message of asArray(body.messages)) {
    if (!isRecord(message)) continue;
    const role = asString(message.role) ?? 'user';
    if (role === 'system' || role === 'developer') {
      const text = textOfContent(message.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      let response: unknown = textOfContent(message.content);
      const raw = asString(message.content);
      if (raw !== null) {
        try {
          response = JSON.parse(raw);
        } catch {
          response = raw;
        }
      }
      const name = asString(message.name) ?? 'tool';
      contents.push({ role: 'function', parts: [{ functionResponse: { name, response: { result: response } } }] });
      continue;
    }
    if (role === 'assistant') {
      const parts = geminiParts(message.content);
      for (const call of asArray(message.tool_calls)) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: unknown = {};
        const raw = asString(call.function.arguments) ?? '';
        if (raw !== '') {
          try {
            args = JSON.parse(raw);
          } catch {
            args = {};
          }
        }
        parts.push({ functionCall: { name: asString(call.function.name) ?? 'tool', args } });
      }
      if (parts.length === 0) continue;
      contents.push({ role: 'model', parts });
      continue;
    }
    const parts = geminiParts(message.content);
    if (parts.length === 0) continue;
    contents.push({ role: 'user', parts });
  }

  const declarations: Json[] = [];
  for (const tool of asArray(body.tools)) {
    if (!isRecord(tool)) continue;
    const fn = isRecord(tool.function) ? tool.function : tool;
    const parameters = isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} };
    declarations.push({ name: asString(fn.name) ?? 'tool', description: asString(fn.description) ?? '', parameters });
  }

  const generationConfig: Json = {};
  const maxTokens = asNumber(body.max_completion_tokens) ?? asNumber(body.max_tokens);
  if (maxTokens !== null) generationConfig.maxOutputTokens = Math.max(1, Math.trunc(maxTokens));
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) generationConfig.stopSequences = body.stop.filter((s) => typeof s === 'string');
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') generationConfig.responseMimeType = 'application/json';

  const out: Json = { contents, generationConfig };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n\n') }] };
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  return out;
}

/* ------------------------------------------------------------------ gemini -> canonical answer */

const GEMINI_STOP: Record<string, string> = {
  STOP: 'stop',
  MAX_TOKENS: 'length',
  SAFETY: 'content_filter',
  RECITATION: 'content_filter',
  OTHER: 'stop',
};

/** The canonical message and finish reason of one Gemini candidate. */
export function fromGeminiCandidate(candidate: Json): { message: Json; finish: string } {
  const content = isRecord(candidate.content) ? candidate.content : {};
  let text = '';
  const toolCalls: Json[] = [];
  for (const part of asArray(content.parts)) {
    if (!isRecord(part)) continue;
    const value = asString(part.text);
    if (value !== null) text += value;
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        index: toolCalls.length,
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: { name: asString(part.functionCall.name) ?? 'tool', arguments: JSON.stringify(part.functionCall.args ?? {}) },
      });
    }
  }
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const raw = asString(candidate.finishReason) ?? 'STOP';
  const finish = toolCalls.length > 0 ? 'tool_calls' : GEMINI_STOP[raw] ?? 'stop';
  return { message, finish };
}

/** Translate one Gemini generateContent answer into a canonical answer. */
export function fromGeminiAnswer(answer: Json, requestedModel: string): Json {
  const candidates = asArray(answer.candidates);
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const { message, finish } = fromGeminiCandidate(first);
  const usage = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  const prompt = asNumber(usage.promptTokenCount) ?? 0;
  const completion = asNumber(usage.candidatesTokenCount) ?? 0;
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: 0,
    model: requestedModel,
    choices: [{ index: 0, message, finish_reason: finish }],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

/* ------------------------------------------------------------------ canonical streaming */

/** A never-changing chunk prefix: replay-stable, so `created` is a constant rather than the clock. */
const CREATED = 0;

/** One canonical delta chunk, in the shape every OpenAI client expects. */
export function chunkOf(id: string, model: string, delta: Json, finish: string | null): Json {
  return {
    id,
    object: 'chat.completion.chunk',
    created: CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

/** The last chunk: an empty delta carrying only usage, which OpenAI sends when it is asked to. */
export function usageChunkOf(id: string, model: string, prompt: number, completion: number): Json {
  return {
    id,
    object: 'chat.completion.chunk',
    created: CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

/**
 * Read a provider's SSE body as `data:` payloads. Gemini, Anthropic and OpenAI all frame their events the
 * same way at this level; only the payloads differ, so one reader serves all three.
 */
export async function* ssePayloads(stream: ReadableStream<Uint8Array>): AsyncGenerator<Json> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      for (;;) {
        const cut = buffer.indexOf('\n');
        if (cut < 0) break;
        const line = buffer.slice(0, cut).replace(/\r$/, '');
        buffer = buffer.slice(cut + 1);
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '' || payload === '[DONE]') continue;
        try {
          const parsed: unknown = JSON.parse(payload);
          if (isRecord(parsed)) yield parsed;
        } catch {
          // A half-written event is not an event: skip it and keep reading.
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
