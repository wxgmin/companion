// Point AIRI at the local companion stack by writing its persisted settings.
//
// AIRI is an Electron app whose settings live in Chromium's Local Storage
// LevelDB - not in any config file, and not in the neighbouring JSON files
// (those hold window geometry, language and MCP servers). There is no settings
// file to edit and no CLI, so the choice is to click through the UI or to write
// the store directly. This does the latter, which is also the only way to make
// setup reproducible for anyone cloning the repo.
//
//   node configure_airi.mjs [--voice-url URL] [--llm-model NAME] [--dry-run]
//
// AIRI MUST BE CLOSED. It holds the store open and rewrites it on exit, so a
// write made underneath a running instance is lost. A timestamped backup of the
// whole leveldb directory is taken first regardless.
//
// Encoding notes, discovered by dumping a real store:
//   keys   : "_<origin>\x00<enc><name>"   (origin here is "file://")
//   values : 0x00 => UTF-16LE, 0x01 => UTF-8/Latin-1
// Getting the 0x01 byte wrong makes the value unreadable to the app.

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'
import { execSync } from 'node:child_process'
import { ClassicLevel } from 'classic-level'

const ORIGIN = 'file://'
const KEY_PREFIX = `_${ORIGIN}\x00\x01`

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`)
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback
}
const DRY_RUN = process.argv.includes('--dry-run')
const VOICE_URL = arg('voice-url', 'http://127.0.0.1:8092/v1')
const LLM_MODEL = arg('llm-model', 'qwen38-local:latest')

const USERDATA = path.join(os.homedir(), 'AppData', 'Roaming', 'ai.moeru.airi')
const LEVELDB = path.join(USERDATA, 'Local Storage', 'leveldb')

// The settings that actually decide whether she talks and where.
//
// speech/hearing use `openai-audio`, NOT `openai-compatible`: the latter
// declares tasks:['chat'] and can never be selected for voice, despite the name
// suggesting otherwise. `openai-audio` builds both a speech and a transcription
// client from one baseUrl, which is why our server exposes both routes.
//
// stage/model is a built-in preset id. preset-vrm-1 is AvatarSample_A, the
// bundled VRM avatar - chosen over the default preset-live2d-1 because it is a
// 3D model with the blend shapes that lip-sync drives.
const SETTINGS = {
  'settings/speech/active-provider': 'openai-audio',
  'settings/speech/active-model': 'chatterbox',
  'settings/speech/voice': 'her',
  'settings/hearing/active-provider': 'openai-audio',
  'settings/hearing/active-model': 'whisper-1',
  'settings/consciousness/active-provider': 'ollama',
  'settings/consciousness/active-model': LLM_MODEL,
  'settings/stage/model': 'preset-vrm-1',
  'settings/audio/input/enabled': 'true',
  'settings/audio/input': 'default',
  'settings/language': 'en',
}

// Provider instances. `added` just marks them present in the catalog;
// `configured` holds the real instance, whose shape is InferenceServiceProvider
// from @proj-airi/stage-ui/src/libs/providers/types.ts.
const PROVIDERS = {
  'openai-audio': {
    id: 'openai-audio',
    definitionId: 'openai-audio',
    config: { apiKey: 'local', baseUrl: VOICE_URL },
    status: 'configured',
    configuredBy: 'user',
  },
  'ollama': {
    id: 'ollama',
    definitionId: 'ollama',
    config: { baseUrl: 'http://127.0.0.1:11434' },
    status: 'configured',
    configuredBy: 'user',
  },
}
const ADDED = { 'openai-audio': true, 'ollama': true }

function backupDir() {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const dest = path.join(USERDATA, `Local Storage.bak-${stamp}`)
  fs.cpSync(path.join(USERDATA, 'Local Storage'), dest, { recursive: true })
  return dest
}

function isAiriRunning() {
  // Deliberately NOT checking for the LOCK file: LevelDB leaves that behind
  // after a clean exit, so it reports "running" on every launch and trains you
  // to ignore the warning. Only a real process matters.
  try {
    const out = execSync(
      'tasklist /FI "IMAGENAME eq airi.exe" /NH',
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] },
    )
    return /airi\.exe/i.test(out)
  } catch {
    return false
  }
}

async function main() {
  if (!fs.existsSync(LEVELDB)) {
    console.error(`AIRI Local Storage not found at:\n  ${LEVELDB}`)
    console.error('Launch AIRI once so it creates its profile, then re-run.')
    return 1
  }

  if (isAiriRunning() && !DRY_RUN) {
    console.warn('WARNING: AIRI is running.')
    console.warn('         It will overwrite these settings when it exits.\n')
  }

  const db = new ClassicLevel(LEVELDB, {
    keyEncoding: 'binary',
    valueEncoding: 'binary',
  })
  try {
    await db.open({ createIfMissing: false })
  } catch (err) {
    // LevelDB takes an exclusive lock, so AIRI holding the store is the normal
    // reason this fails. Say so, because "Database failed to open" alone sends
    // people looking for a corrupt profile.
    if (isAiriRunning()) {
      console.error('AIRI is running, so its settings store is locked.')
      console.error('Close AIRI completely (check the tray), then re-run.')
      console.error('Nothing was changed.')
      return 1
    }
    throw err
  }

  // Back up only once the store is actually ours to write; a backup taken when
  // the open fails just litters the profile with copies we never touched.
  if (!DRY_RUN) {
    console.log(`backup: ${backupDir()}`)
  }

  const writes = { ...SETTINGS }
  writes['settings/providers/added'] = JSON.stringify(ADDED)
  writes['settings/providers/configured'] = JSON.stringify(PROVIDERS)

  let changed = 0
  for (const [name, value] of Object.entries(writes)) {
    const key = Buffer.from(KEY_PREFIX + name, 'latin1')
    // 0x01 marks a UTF-8 value; the app reads this byte to pick a decoder.
    const val = Buffer.concat([Buffer.from([0x01]), Buffer.from(value, 'utf8')])

    let current
    try {
      current = await db.get(key)
    } catch {
      current = undefined
    }

    const before = current ? current.subarray(1).toString('utf8') : '(unset)'
    if (before === value) {
      console.log(`  = ${name}`)
      continue
    }

    console.log(`  ~ ${name}`)
    console.log(`      was: ${before.slice(0, 90)}`)
    console.log(`      now: ${value.slice(0, 90)}`)
    if (!DRY_RUN) await db.put(key, val)
    changed += 1
  }

  await db.close()

  console.log()
  if (DRY_RUN) {
    console.log(`dry run: ${changed} setting(s) would change`)
  } else {
    console.log(`wrote ${changed} setting(s)`)
    console.log()
    console.log('AIRI will now use:')
    console.log(`  speech        chatterbox via ${VOICE_URL}`)
    console.log(`  hearing       whisper via ${VOICE_URL}`)
    console.log(`  consciousness ${LLM_MODEL} via ollama`)
    console.log('  stage         preset-vrm-1 (AvatarSample_A, 3D + lip-sync)')
  }
  return 0
}

main().then((c) => process.exit(c)).catch((err) => {
  console.error('FAILED:', err.message)
  process.exit(1)
})
