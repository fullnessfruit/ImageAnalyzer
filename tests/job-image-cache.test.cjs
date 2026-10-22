const test = require('node:test');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { JobImageCache, TimedImageResultCache } = require('../server/dist/job-image-cache');

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

test('identical bytes from different URLs are analyzed once per job', async () => {
  const cache = new JobImageCache();
  const firstUrl = 'https://pbs.twimg.com/media/example.jpg:orig';
  const secondUrl = 'https://pbs.twimg.com/media/example?format=jpg&name=orig';
  const bytes = Buffer.from('same image bytes');
  let calls = 0;
  const analyze = async () => { calls++; return [{ name: '大西亜玖璃' }]; };

  const first = await cache.getOrAnalyze(sha256(bytes), firstUrl, analyze);
  const second = await cache.getOrAnalyze(sha256(Buffer.from(bytes)), secondUrl, analyze);
  const third = await cache.getOrAnalyze(sha256(Buffer.from('different image bytes')), secondUrl, analyze);

  assert.equal(calls, 2);
  assert.deepEqual(second.result, first.result);
  assert.equal(second.reusedFromImageUrl, firstUrl);
  assert.equal(first.reusedFromImageUrl, undefined);
  assert.equal(third.reusedFromImageUrl, undefined);
});

test('failed analysis is retried for a duplicate image', async () => {
  const cache = new JobImageCache();
  const hash = sha256(Buffer.from('image'));
  let calls = 0;
  const analyze = async () => {
    calls++;
    if (calls === 1) throw new Error('model failed');
    return ['found'];
  };

  await assert.rejects(cache.getOrAnalyze(hash, 'first', analyze), /model failed/);
  assert.deepEqual((await cache.getOrAnalyze(hash, 'second', analyze)).result, ['found']);
  assert.equal(calls, 2);
});

test('URL results are reused for 24 hours only with the same analysis inputs', () => {
  let now = 1000;
  const cache = new TimedImageResultCache(24 * 60 * 60 * 1000, () => now);
  const url = 'https://pbs.twimg.com/media/example.jpg';
  const result = { found: ['大西亜玖璃'] };
  cache.set(url, 'search=A;threshold=0.5', result, 'hash-a', 123);

  assert.deepEqual(cache.get(url, 'search=A;threshold=0.5'), { result, imageSha256: 'hash-a', bytes: 123 });
  assert.equal(cache.get(url, 'search=B;threshold=0.5'), undefined);
  assert.equal(cache.get(url, 'search=A;threshold=0.6'), undefined);
  assert.equal(cache.get(`${url}:large`, 'search=A;threshold=0.5'), undefined);
  now += 24 * 60 * 60 * 1000;
  assert.equal(cache.get(url, 'search=A;threshold=0.5'), undefined);
});

test('expired URL results are pruned from memory', () => {
  let now = 0;
  const cache = new TimedImageResultCache(10, () => now);
  cache.set('url-a', 'query', ['a'], 'hash-a', 1);
  now = 5;
  cache.set('url-b', 'query', ['b'], 'hash-b', 2);
  now = 10;
  cache.prune();
  assert.equal(cache.get('url-a', 'query'), undefined);
  assert.deepEqual(cache.get('url-b', 'query')?.result, ['b']);
});
