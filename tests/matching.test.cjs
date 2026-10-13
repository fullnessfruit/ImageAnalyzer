const test = require('node:test');
const assert = require('node:assert/strict');
const { matchGallery } = require('../server/dist/matching');

test('costume variants are compared independently without weakening identity scores', () => {
  const side = Math.SQRT1_2;
  const gallery = [{
    name: 'outfit',
    vectors: [new Float32Array([1, 0, 0]), new Float32Array([0, 1, 0])],
    centroid: new Float32Array([side, side, 0]),
  }];
  const query = new Float32Array([1, 0, 0]);

  assert.equal(matchGallery(query, gallery, 0.9, 0), null);
  assert.equal(matchGallery(query, gallery, 0.9, 0, undefined, 'max')?.name, 'outfit');
});

test('costume exemplar scoring still rejects close runner-ups', () => {
  const gallery = [
    { name: 'first', vectors: [new Float32Array([1, 0])], centroid: new Float32Array([1, 0]) },
    { name: 'second', vectors: [new Float32Array([0.96, 0.28])], centroid: new Float32Array([0.96, 0.28]) },
  ];
  assert.equal(matchGallery(new Float32Array([1, 0]), gallery, 0.9, 0.05, undefined, 'max'), null);
});
