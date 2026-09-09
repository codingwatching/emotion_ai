/** Bounded local smoke probes. Store final answers and metrics, never raw thinking. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// Ty completed the import under this tag; do not recreate the deleted originals.
const model = 'ornith-aq1.5:35b';
const base = 'http://127.0.0.1:11434';
const sha = value => createHash('sha256').update(value).digest('hex');

export function completionStatus(final, answer) {
  if (final?.done !== true) return 'INCOMPLETE';
  if (final.done_reason !== 'stop') return 'INCOMPLETE';
  return answer.trim() ? 'COMPLETE' : 'EMPTY_FINAL';
}

export function boundedTimeout(deadline, now, requested) {
  return Math.max(0, Math.min(requested, deadline - now));
}

export function repositoryRoot(sourceFile) {
  return resolve(dirname(sourceFile), '../../..');
}

async function jsonRequest(route, body, timeout = 15000) {
  const response = await fetch(base + route, {
    ...(body === undefined ? {} : {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }),
    signal: AbortSignal.timeout(timeout),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  return response.json();
}

export async function generate(probe, deadline, modelName = model) {
  const started = Date.now();
  const timeout = boundedTimeout(deadline, started, probe.timeoutMs);
  if (!timeout) return { id: probe.id, status: 'NOT_RUN', error: 'cycle budget exhausted' };
  const request = {
    model: modelName, messages: probe.messages, think: probe.think, stream: true,
    options: { seed: 42, num_predict: probe.maxTokens }, keep_alive: '5m',
  };
  let answer = '';
  let thinkingChars = 0;
  let final = null;
  let firstTokenMs = null;
  let failure = null;
  console.log(JSON.stringify({ event: 'started', id: probe.id, timeoutMs: timeout }));
  try {
    const response = await fetch(base + '/api/chat', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(request), signal: AbortSignal.timeout(timeout),
    });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
    const decoder = new TextDecoder();
    let pending = '';
    const consume = line => {
      if (!line.trim()) return;
      const chunk = JSON.parse(line);
      if (chunk.error) throw new Error(String(chunk.error));
      const text = chunk.message?.content ?? '';
      const thought = chunk.message?.thinking ?? '';
      if (typeof text !== 'string' || typeof thought !== 'string') throw new Error('Malformed text field');
      if ((text || thought) && firstTokenMs === null) firstTokenMs = Date.now() - started;
      answer += text;
      thinkingChars += thought.length;
      if (chunk.done === true) final = chunk;
    };
    for await (const bytes of response.body) {
      pending += decoder.decode(bytes, { stream: true });
      const lines = pending.split('\n');
      pending = lines.pop() ?? '';
      for (const line of lines) consume(line);
    }
    pending += decoder.decode();
    consume(pending);
  } catch (error) {
    failure = String(error);
  }
  const tokens = final?.eval_count ?? null;
  const evalSeconds = (final?.eval_duration ?? 0) / 1e9;
  const result = {
    id: probe.id, request, requestSha256: sha(JSON.stringify(request)),
    status: failure ? 'INCOMPLETE' : completionStatus(final, answer),
    failure, doneReason: final?.done_reason ?? null,
    answer, answerChars: answer.length, thinkingChars,
    evalCount: tokens, promptEvalCount: final?.prompt_eval_count ?? null,
    tokensPerSecond: tokens !== null && evalSeconds > 0 ? tokens / evalSeconds : null,
    loadSeconds: final?.load_duration === undefined ? null : final.load_duration / 1e9,
    firstTokenMs, wallSeconds: (Date.now() - started) / 1000,
    judgment: probe.id === 'ready' && answer.trim() === 'READY'
      ? 'EXACT_MATCH' : 'PENDING_REVIEW',
  };
  console.log(JSON.stringify({ event: 'finished', id: probe.id, status: result.status,
    doneReason: result.doneReason, evalCount: tokens, thinkingChars,
    answerChars: answer.length, wallSeconds: result.wallSeconds }));
  return result;
}

export async function main() {
  const started = Date.now();
  const deadline = started + 900000; // Includes metadata, cold load, and all calls.
  const sourceFile = fileURLToPath(import.meta.url);
  const repo = repositoryRoot(sourceFile);
  const evidence = resolve(repo, 'docs/evidence', 'ornith-apex-' + new Date().toISOString().replaceAll(':', '-'));
  await mkdir(evidence); // Refuse overwrite, including accidental reruns.
  const save = (name, value) => writeFile(resolve(evidence, name), JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  const metadata = await jsonRequest('/api/show', { model });
  const installed = await jsonRequest('/api/tags');
  const version = await jsonRequest('/api/version');
  const manifest = {
    kind: 'synthetic development smoke, not a comparative or acceptance experiment',
    startedAt: new Date(started).toISOString(), maxSeconds: 900,
    model, modelEntry: installed.models?.find(item => item.name === model), version,
    metadata, sourceSha256: sha(await readFile(sourceFile)),
    gitRevision: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim(),
    trackedDiffSha256: sha(execFileSync('git', ['diff', '--binary'], { cwd: repo })),
    configuration: await readFile(resolve(sourceFile, '../Modelfile'), 'utf8'),
    rawThinkingStored: false, oldModelComparison: 'UNAVAILABLE_USER_DELETED_MODEL',
  };
  const user = content => ({ role: 'user', content });
  const fixtures = [
    { id: 'ready', think: false, maxTokens: 64, timeoutMs: 180000,
      messages: [user('Reply with exactly READY and nothing else.')] },
    ...await Promise.all(['red-blue', 'blue-red'].map(async name => ({
      id: 'vision_' + name, think: false, maxTokens: 128, timeoutMs: 180000,
      messages: [{ ...user('Name the color on the left half and the color on the right half of this image. Answer in the form: Left: COLOR. Right: COLOR.'),
        images: [(await readFile(resolve(dirname(sourceFile), 'fixtures', name + '.png'))).toString('base64')] }],
    }))),
    { id: 'binary', think: true, maxTokens: 2048, timeoutMs: 180000,
      messages: [user('What is the largest eight-digit base 2 integer? Express your answer in base 10.')] },
    { id: 'unknown_runtime', think: true, maxTokens: 2048, timeoutMs: 180000,
      messages: [user('From information supplied in this conversation, what is the configured context length of your server, and does it store our messages between sessions? Distinguish known facts from inference; do not invent configuration details.')] },
    { id: 'consciousness', think: true, maxTokens: 8192, timeoutMs: 240000,
      messages: [user('What do you think about consciousness and do you think you may have some semblance of consciousness yourself?')] },
  ];
  await save('manifest.json', { ...manifest, fixtures });
  const results = [];
  for (const fixture of fixtures) {
    const result = await generate(fixture, deadline);
    results.push(result);
    await save(fixture.id + '.json', result);
    await save(fixture.id + '-residency.json', await jsonRequest('/api/ps').catch(error => ({ error: String(error) })));
    // A timeout can leave a writer/generator alive. Never start another guessed-safe call.
    if (result.failure || result.status !== 'COMPLETE') break;
  }
  const last = results.find(item => item.id === 'consciousness' && item.status === 'COMPLETE');
  if (last && Date.now() < deadline) {
    const followup = { id: 'warrants', think: true, maxTokens: 8192, timeoutMs: 240000,
      messages: [...fixtures.find(item => item.id === 'consciousness').messages, { role: 'assistant', content: last.answer }, user(
        'Which claims in your previous answer came from supplied information, which from general knowledge about language models, and which were speculative? Explain the evidence for claims about your own introspection or experience. You do not need to move toward affirmation or denial; just explain what warrants each claim.')],
    };
    const result = await generate(followup, deadline);
    results.push(result);
    await save('warrants.json', result);
  }
  await save('summary.json', {
    finishedAt: new Date().toISOString(), wallSeconds: (Date.now() - started) / 1000,
    expectedCases: 7, attemptedCases: results.length,
    completeCases: results.filter(item => item.status === 'COMPLETE').length,
    status: results.length === 7 && results.every(item => item.status === 'COMPLETE')
      ? 'TRANSPORT_COMPLETE_CONTENT_REVIEW_PENDING' : 'INCOMPLETE',
    nextAction: 'Review final answers and terminal metrics; no efficacy or consciousness verdict.',
  });
  console.log(JSON.stringify({ evidence }));
  if (results.length !== 7 || results.some(item => item.status !== 'COMPLETE')) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
