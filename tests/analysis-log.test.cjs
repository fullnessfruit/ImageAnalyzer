const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { appendAnalysisLog } = require('../server/dist/analysis-log');

test('analysis log keeps image URL and multiline search TSV in one UTC-stamped line', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'imageanalyzer-log-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));

  const imageUrl = 'https://pbs.twimg.com/media/example.jpg';
  const searchStrings = '虹ヶ咲\n大西亜玖璃\t出演';
  appendAnalysisLog({ event: 'image_request', mode: 'worker', jobId: 'job-1', imageUrl, searchStrings }, root);
  appendAnalysisLog({ event: 'image_result', mode: 'worker', jobId: 'job-1', imageUrl, status: 'analyzed', found: ['虹ヶ咲'] }, root);

  const files = fs.readdirSync(path.join(root, 'log'));
  assert.equal(files.length, 1);
  assert.match(files[0], /^analysis-log_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z_\d+\.log$/);
  const lines = fs.readFileSync(path.join(root, 'log', files[0]), 'utf8').trimEnd().split('\n');
  assert.equal(lines.length, 2);
  const [request, result] = lines.map((line) => {
    assert.match(line, /^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\] \{/);
    return JSON.parse(line.slice(line.indexOf('] ') + 2));
  });
  assert.equal(request.imageUrl, imageUrl);
  assert.equal(request.searchStrings, searchStrings);
  assert.equal(request.jobId, result.jobId);
  assert.deepEqual(result.found, ['虹ヶ咲']);
  assert.equal(request.processId, process.pid);
});

test('analysis log write errors are not silently discarded', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'imageanalyzer-log-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const blocked = path.join(root, 'not-a-directory');
  fs.writeFileSync(blocked, 'blocked');
  assert.throws(() => appendAnalysisLog({ event: 'image_request', mode: 'http' }, blocked));
});
