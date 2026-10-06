'use strict';

// Keep the storage format identical to VoiceAnalyzer/app/shared_secret.py.
// Reading never creates a key; only an explicit installation may do that.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { execFileSync } = require('node:child_process');
const ENV_NAME = 'OCR_BROKER_SECRET';

function secretPath() {
  return path.join(process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config'),
    'announcement-analyzers', 'auth.json');
}

function powershell(command, env = process.env) {
  return execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
    "$ErrorActionPreference = 'Stop'; [Console]::OutputEncoding = [Text.UTF8Encoding]::new(); " + command
  ], { encoding: 'utf8', env, windowsHide: true }).replace(/^\uFEFF/, '');
}

function storedSecret() {
  if (process.platform === 'win32') {
    return powershell(`[Console]::Write([Environment]::GetEnvironmentVariable('${ENV_NAME}', 'User'))`);
  }
  let data;
  try { data = fs.readFileSync(secretPath(), 'utf8'); } catch (error) {
    if (error.code === 'ENOENT') return '';
    throw error;
  }
  let value;
  try { value = JSON.parse(data)[ENV_NAME]; } catch (_) {
    throw new Error('Invalid shared analyzer key file');
  }
  if (typeof value !== 'string' || !value) throw new Error('Invalid shared analyzer key file');
  return value;
}

function readSecret() {
  return process.env[ENV_NAME] || storedSecret();
}

function ensureSecret() {
  // The persisted value wins during installation, even in a stale terminal.
  const existing = storedSecret();
  if (existing) return existing;
  const secret = process.env[ENV_NAME] || crypto.randomBytes(32).toString('hex');
  if (process.platform === 'win32') {
    powershell(`[Environment]::SetEnvironmentVariable('${ENV_NAME}', $env:ANALYZER_SETUP_SECRET, 'User')`,
      { ...process.env, ANALYZER_SETUP_SECRET: secret });
    return storedSecret();
  }
  const target = secretPath();
  fs.mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 });
  const temporary = `${target}.${crypto.randomBytes(12).toString('hex')}.tmp`;
  const fd = fs.openSync(temporary, 'wx', 0o600);
  try {
    fs.writeFileSync(fd, JSON.stringify({ [ENV_NAME]: secret }) + '\n');
    fs.fsyncSync(fd);
  } finally { fs.closeSync(fd); }
  try {
    // Publish a complete file without replacing a concurrent installer's key.
    try { fs.linkSync(temporary, target); } catch (error) {
      if (error.code !== 'EEXIST') throw error;
    }
  } finally { fs.unlinkSync(temporary); }
  return storedSecret();
}

function deleteSecret() {
  // This operation is separate from uninstalling either analyzer.
  if (process.platform === 'win32') {
    powershell(`[Environment]::SetEnvironmentVariable('${ENV_NAME}', $null, 'User')`);
  } else {
    fs.rmSync(secretPath(), { force: true });
  }
}

if (require.main === module) {
  try {
    if (process.argv[2] === 'ensure') {
      ensureSecret();
      process.stdout.write('Shared OCR_BROKER_SECRET is ready; existing keys are preserved.\n');
    } else if (process.argv[2] === 'show') {
      const secret = storedSecret() || readSecret();
      if (!secret) throw new Error('OCR_BROKER_SECRET is not configured');
      process.stdout.write(secret + '\n');
    } else if (process.argv[2] === 'delete') {
      deleteSecret();
      process.stdout.write('Deleted the persisted shared OCR_BROKER_SECRET. Reconfigure both analyzers and their clients together.\n');
      process.stdout.write('Running services and terminals retain their old environment until restarted.\n');
    } else throw new Error('Usage: shared-secret.cjs ensure|show|delete');
  } catch (error) {
    process.stderr.write(`Shared analyzer key setup failed: ${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = { readSecret, ensureSecret, deleteSecret };
