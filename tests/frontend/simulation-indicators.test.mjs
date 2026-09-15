import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import ts from 'typescript';

async function loadModule(path, dependencies = {}) {
  const source = await readFile(new URL(path, import.meta.url), 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  });
  const exports = {};
  vm.runInNewContext(outputText, {
    exports, require: name => {
      if (!(name in dependencies)) throw new Error(`Unexpected dependency: ${name}`);
      return dependencies[name];
    },
    document: { addEventListener() {} }, console: { log() {}, warn() {} },
  });
  return exports;
}

const renderer = await loadModule('../../src/services/simulationDisplay.ts');
const { AuraUIManager } = await loadModule('../../index.tsx', {
  marked: {}, './src/services/auraApi': {}, './src/services/simulationDisplay': renderer,
});

function element() {
  const properties = {};
  return { textContent: '', title: '', className: '', properties,
    style: { setProperty: (key, value) => { properties[key] = value; } } };
}
function simulation(band = 'Alpha', dominant = 'serotonin_like') {
  return {
    scope_id: 'Ty', revision: 4, disposition: 'committed',
    pre_state: { valence: 0, curiosity: 0, affiliation: 0, load: 0 },
    post_state: { valence: 0.1, arousal: 0.3, novelty: 0, curiosity: 0.6, affiliation: 0.65, control: 0.75, load: 0.1 },
    policy: { warmth: 'warm', energy: 'balanced' }, causes: [],
    channels: { dopamine_like: 0.575, norepinephrine_like: 0.15, acetylcholine_like: 0.1,
      serotonin_like: 0.75, gaba_like: 0.75, cortisol_like: 0.1 },
    display: { basis: 'published_affect_state', brainwave: band, activation: 0.3, dominant_channel: dominant, emotion: { name: 'Calm', intensity: 'Low', description: 'Near resting state.' } },
  };
}
function ui(api = {}) {
  const manager = new AuraUIManager(api);
  for (const name of ['emotionStatusElement', 'emotionDetailsElement', 'emotionIntensityElement',
    'brainwaveValueElement', 'wavePatternElement', 'ntValueElement', 'chemicalLevelElement',
    'simRevisionBadge', 'simPolicyDesc', 'simValence', 'simCuriosity', 'simAffiliation', 'simLoad', 'simCauses', 'simChannels']) {
    manager[name] = element();
  }
  manager.updateHeaderEmotionalState = () => {};
  manager.userName = 'Ty';
  manager.backendConnected = true;
  return manager;
}

test('invalid tone classification does not clear valid controller indicators', () => {
  const manager = ui();
  manager.updateEmotionalState({ name: 'Unknown', intensity: 'Unknown', brainwave: '', neurotransmitter: '',
    description: 'Emotion analysis could not be validated.', simulation: simulation() });
  assert.equal(manager.emotionStatusElement.textContent, 'Calm');
  assert.equal(manager.brainwaveValueElement.textContent, 'Alpha 30%');
  assert.equal(manager.wavePatternElement.className, 'wave-pattern wave-alpha');
  assert.equal(manager.ntValueElement.textContent, 'Serotonin 75%');
  assert.equal(manager.chemicalLevelElement.properties['--chemical-intensity'], '75%');
  assert.match(manager.ntValueElement.title, /Dopamine-like: 57%/);
  assert.equal(manager.simValence.textContent, '0.10'); // published state, not pre-state
});

test('changes in the controller update both indicators and clear missing data', () => {
  const manager = ui();
  const changed = simulation('Gamma', 'cortisol_like');
  changed.channels.cortisol_like = 0.95;
  changed.display.activation = 0.9;
  manager.updateSimulation(changed);
  assert.equal(manager.brainwaveValueElement.textContent, 'Gamma 90%');
  assert.equal(manager.ntValueElement.textContent, 'Cortisol 95%');
  manager.updateSimulation(null);
  assert.equal(manager.wavePatternElement.className, 'wave-pattern wave-unknown');
  assert.equal(manager.chemicalLevelElement.properties['--chemical-intensity'], '0%');
  assert.equal(manager.simRevisionBadge.textContent, 'No saved state');
});

test('restoring a saved state populates the header without a conversation call', async () => {
  const manager = ui({ savedSimulation: async user => {
    assert.equal(user, 'Ty');
    const saved = simulation();
    saved.disposition = 'restored';
    delete saved.causes; // older replay responses omitted this field
    return { simulation: saved };
  } });
  await manager.restoreSimulation();
  assert.equal(manager.ntValueElement.textContent, 'Serotonin 75%');
  assert.equal(manager.simRevisionBadge.textContent, 'Rev 4 (restored)');
  assert.equal(manager.emotionStatusElement.textContent, 'Calm');
  assert.match(manager.simChannels.textContent, /Dopamine: 57.5%/);
});

test('malformed channels cannot produce invented or invalid meter values', () => {
  const manager = ui();
  const malformed = simulation();
  malformed.channels.serotonin_like = NaN;
  manager.updateSimulation(malformed);
  assert.equal(manager.ntValueElement.textContent, 'Unknown');
  assert.equal(manager.chemicalLevelElement.properties['--chemical-intensity'], '0%');
});


test('state changes remain visible while the band and dominant chemical stay the same', () => {
  const manager = ui();
  manager.updateSimulation(simulation());
  const changed = simulation();
  changed.display.activation = 0.4;
  changed.display.emotion = { name: 'Curious', intensity: 'Medium', description: 'Exploring' };
  changed.channels.dopamine_like = 0.635;
  changed.channels.norepinephrine_like = 0.29;
  manager.updateSimulation(changed);
  assert.equal(manager.emotionStatusElement.textContent, 'Curious');
  assert.equal(manager.brainwaveValueElement.textContent, 'Alpha 40%');
  assert.equal(manager.ntValueElement.textContent, 'Serotonin 75%');
  assert.match(manager.simChannels.textContent, /Dopamine: 63.5%/);
  assert.match(manager.simChannels.textContent, /Norepinephrine: 29.0%/);
});
