const test = require('node:test');
const assert = require('node:assert/strict');
const sharp = require('sharp');
const { cropWithMask } = require('../server/dist/inference');

test('a mask stays inside a crop clipped by the image boundary', async () => {
  const image = await sharp({ create: { width: 100, height: 100, channels: 3, background: '#ffffff' } })
    .png().toBuffer();
  const cropped = await cropWithMask(image, [80, 80, 50, 50], [[90, 90, 30, 30]]);
  const { data, info } = await sharp(cropped).raw().toBuffer({ resolveWithObject: true });
  const colorAt = (x, y) => [...data.subarray((y * info.width + x) * info.channels, (y * info.width + x) * info.channels + 3)];

  assert.equal(info.width, 20);
  assert.equal(info.height, 20);
  assert.deepEqual(colorAt(5, 5), [255, 255, 255]);
  assert.deepEqual(colorAt(15, 5), [128, 128, 128]);
});
