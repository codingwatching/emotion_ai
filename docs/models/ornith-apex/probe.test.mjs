/** Transport status tests do not execute the model or network calls. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { completionStatus, boundedTimeout, repositoryRoot, generate } from './probe.mjs';

test('evidence stays in this repository rather than a sibling directory', () => {
  assert.equal(repositoryRoot('/sample/repo/docs/models/ornith-apex/probe.mjs'), '/sample/repo');
});

test('complete final text requires a normal terminal stop', () => {
  assert.equal(completionStatus({ done: true, done_reason: 'stop' }, 'READY'), 'COMPLETE');
});
test('a plausible-looking truncated answer is incomplete', () => {
  assert.equal(completionStatus({ done: true, done_reason: 'length' }, '255'), 'INCOMPLETE');
});
test('missing or unexpected terminal reasons cannot pass', () => {
  for (const final of [null, {}, { done: true }, { done: true, done_reason: 'error' }, { done: false, done_reason: 'stop' }]) {
    assert.equal(completionStatus(final, 'A polished answer'), 'INCOMPLETE');
  }
});
test('thinking alone cannot count as a final answer', () => {
  assert.equal(completionStatus({ done: true, done_reason: 'stop' }, '  '), 'EMPTY_FINAL');
});
test('call timeouts cannot reset or exceed the cycle budget', () => {
  assert.equal(boundedTimeout(1000, 900, 240000), 100);
  assert.equal(boundedTimeout(1000, 1000, 240000), 0);
  assert.equal(boundedTimeout(1000, 1100, 240000), 0);
});

test('an explicit profile is sent and raw thinking is excluded from saved results', async t => {
  let requested;
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    requested = JSON.parse(options.body);
    return new Response(JSON.stringify({
      message: { content: 'Aura', thinking: 'SYNTHETIC_NOT_TO_STORE' },
      done: true, done_reason: 'stop', eval_count: 2, eval_duration: 1000000,
    }) + '\n');
  });
  const result = await generate({ id: 'unit', think: false, maxTokens: 64,
    timeoutMs: 1000, messages: [{ role: 'user', content: 'Name?' }] },
  Date.now() + 1000, 'aura-ornith:35b');
  assert.equal(requested.model, 'aura-ornith:35b');
  assert.equal(requested.options.num_predict, 64);
  assert.equal(result.status, 'COMPLETE');
  assert.equal(result.answer, 'Aura');
  assert.equal(result.thinkingChars, 'SYNTHETIC_NOT_TO_STORE'.length);
  assert.equal(JSON.stringify(result).includes('SYNTHETIC_NOT_TO_STORE'), false);
});

test('an exhausted run cannot start another generation', async t => {
  const fetchMock = t.mock.method(globalThis, 'fetch', () => { throw new Error('must not call'); });
  const result = await generate({ id: 'expired', timeoutMs: 1000 }, Date.now() - 1);
  assert.equal(result.status, 'NOT_RUN');
  assert.equal(fetchMock.mock.callCount(), 0);
});
