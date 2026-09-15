import type { AffectSimulationState } from './auraApi';

const channelNames = {
  dopamine_like: 'Dopamine', norepinephrine_like: 'Norepinephrine',
  acetylcholine_like: 'Acetylcholine', serotonin_like: 'Serotonin',
  gaba_like: 'GABA', cortisol_like: 'Cortisol',
};

export interface SimulationIndicators {
  brainwave: HTMLElement;
  wave: HTMLElement;
  chemical: HTMLElement;
  level: HTMLElement;
  channels?: HTMLElement | null;
}

/** Render actual controller readouts; an unavailable tone label cannot erase them. */
export function renderSimulationIndicators(
  simulation: AffectSimulationState | null | undefined, elements: SimulationIndicators,
): void {
  const display = simulation?.display;
  const channels = simulation?.channels;
  const channel = display?.dominant_channel;
  const valid = display?.basis === 'published_affect_state'
    && ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'].includes(display.brainwave)
    && Number.isFinite(display.activation) && display.activation >= 0 && display.activation <= 1
    && channel && Object.prototype.hasOwnProperty.call(channelNames, channel) && channels
    && Object.keys(channelNames).every(key => {
      const value = channels[key as keyof typeof channelNames];
      return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
    });
  if (!valid) {
    elements.brainwave.textContent = 'Unknown';
    elements.brainwave.title = 'No saved simulation state is available yet.';
    elements.wave.className = 'wave-pattern wave-unknown';
    elements.chemical.textContent = 'Unknown';
    elements.chemical.title = 'No simulated channel values are available yet.';
    elements.level.style.setProperty('--chemical-intensity', '0%');
    if (elements.channels) elements.channels.textContent = 'No saved channel values';
    return;
  }
  const level = Math.round(channels[channel] * 100);
  elements.brainwave.textContent = `${display.brainwave} ${Math.round(display.activation * 100)}%`;
  elements.brainwave.title = `Simulated rhythm: ${Math.round(display.activation * 100)}% activation. Visual analogy, not EEG.`;
  elements.wave.className = `wave-pattern wave-${display.brainwave.toLowerCase()}`;
  elements.chemical.textContent = `${channelNames[channel]} ${level}%`;
  elements.chemical.title = Object.entries(channelNames)
    .map(([key, label]) => `${label}-like: ${Math.round(channels[key as keyof typeof channelNames] * 100)}%`)
    .join('\n');
  elements.level.style.setProperty('--chemical-intensity', `${level}%`);
  if (elements.channels) {
    elements.channels.textContent = Object.entries(channelNames)
      .map(([key, label]) => `${label}: ${(channels[key as keyof typeof channelNames] * 100).toFixed(1)}%`)
      .join(' · ');
  }
}
