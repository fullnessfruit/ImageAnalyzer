const test = require('node:test');
const assert = require('node:assert/strict');
const { selectNamedMatches, foundNames } = require('../server/dist/named-analysis');

test('an untyped character request is not satisfied by a same-named costume', () => {
  const costume = { name: '上原歩夢', score: 0.67, margin: 0.67 };
  const result = { faces: [], characters: [], costumes: [costume] };
  const registered = new Map([['上原歩夢', new Set(['character', 'costume'])]]);

  const costumeOnly = selectNamedMatches(result, ['上原歩夢'], registered);
  assert.deepEqual(costumeOnly.map(match => match.kind), ['costumes']);
  assert.deepEqual(foundNames(costumeOnly, ['上原歩夢'], registered), []);
  result.characters.push({ name: '上原歩夢', score: 0.85, margin: 0.1 });
  const both = selectNamedMatches(result, ['上原歩夢'], registered);
  assert.deepEqual(both.map(match => match.kind), ['characters', 'costumes']);
  assert.deepEqual(foundNames(both, ['上原歩夢'], registered), ['上原歩夢']);
  assert.deepEqual(foundNames(both, ['上原歩夢', 'costume:上原歩夢'], registered), ['上原歩夢', 'costume:上原歩夢']);
});

test('an explicit costume target works without a character match', () => {
  const result = { faces: [], characters: [], costumes: [{ name: '上原歩夢', score: 0.67, margin: 0.67 }] };
  const registered = new Map([['上原歩夢', new Set(['character', 'costume'])]]);
  const requested = ['costume:上原歩夢'];
  const matches = selectNamedMatches(result, requested, registered);

  assert.deepEqual(matches.map(match => match.kind), ['costumes']);
  assert.deepEqual(foundNames(matches, requested, registered), requested);
  assert.deepEqual(selectNamedMatches(result, ['character:上原歩夢'], registered), []);
});

test('a costume-only name can still be reported by the broker', () => {
  const outfit = '虹ヶ咲学園制服';
  const result = { faces: [], characters: [], costumes: [{ name: outfit, score: 0.7, margin: 0.1 }] };
  const registered = new Map([[outfit, new Set(['costume'])]]);

  const matches = selectNamedMatches(result, [outfit], registered);
  assert.deepEqual(matches.map(match => match.name), [outfit]);
  assert.deepEqual(foundNames(matches, [outfit], registered), [outfit]);
});
