'use strict';

// Remove only reproducible installation artifacts. Reference data and models stay.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
try {
  const targets = ['node_modules', 'server/dist'].map(relative => path.join(root, relative));
  // Validate every target before removing the first one.
  for (const target of targets) {
    let stat;
    try { stat = fs.lstatSync(target); } catch (error) {
      if (error.code === 'ENOENT') continue;
      throw error;
    }
    if (stat.isSymbolicLink() || !stat.isDirectory()) {
      throw new Error(`Refusing to remove a link or non-directory: ${target}`);
    }
  }
  for (const target of targets) fs.rmSync(target, { recursive: true, force: true });
  process.stdout.write('Removed node_modules and server/dist. Source, settings, data, models, and db were preserved.\n');
} catch (error) {
  process.stderr.write(`ImageAnalyzer uninstall failed: ${error.message}\n`);
  process.exitCode = 1;
}
