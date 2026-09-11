import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

test('conversation retries preserve identity; new sends get new identities', async () => {
  const saved = { window: globalThis.window, fetch: globalThis.fetch, setInterval: globalThis.setInterval };
  const bodies = [];
  globalThis.window = { location: { hostname: 'localhost', protocol: 'http:' } };
  globalThis.setInterval = () => 0;
  let fail = true;
  globalThis.fetch = async (url, options) => {
    if (url.endsWith('/health')) return Response.json({ status: 'operational' });
    bodies.push(JSON.parse(options.body));
    if (fail) { fail = false; return new Response('pending', { status: 503 }); }
    return Response.json({ response: 'Reply', emotional_state: {}, cognitive_state: {}, session_id: bodies.at(-1).session_id });
  };
  try {
    const source = await readFile(new URL('../../src/services/auraApi.ts', import.meta.url), 'utf8');
    const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 } });
    const { auraAPI } = await import(`data:text/javascript;base64,${Buffer.from(`${outputText}\n//# sourceURL=auraApi.test.js`).toString('base64')}`);
    // Replace only retry backoff time, not request construction or fetch logic.
    auraAPI.sleep = async () => {};
    const request = { user_id: 'test', message: 'Hello' };
    await auraAPI.sendMessage(request);
    assert.equal(bodies.length, 2);
    assert.ok(bodies[0].idempotency_key);
    assert.ok(bodies[0].session_id);
    assert.deepEqual(bodies[0], bodies[1]);
    await auraAPI.sendMessage(request); // explicit retry of the same send
    assert.deepEqual(bodies[2], bodies[0]);
    await auraAPI.sendMessage({ user_id: 'test', message: 'Hello' });
    assert.notEqual(bodies[3].idempotency_key, bodies[0].idempotency_key);
  } finally {
    Object.assign(globalThis, saved);
  }
});
