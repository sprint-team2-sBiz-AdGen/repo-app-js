// All 8 strategies (for future use)
export const STRATEGY_META = {
  hero:      { apiId: 1, apiName: "Hero Dish Focus" },
  seasonal:  { apiId: 2, apiName: "Seasonal / Limited" },
  bts:       { apiId: 3, apiName: "Behind-the-Scenes / Authenticity" },
  lifestyle: { apiId: 4, apiName: "Lifestyle Integration" },
  ugc:       { apiId: 5, apiName: "UGC / Social Proof" },
  minimalist:{ apiId: 6, apiName: "Minimalist Branding" },
  comfort:   { apiId: 7, apiName: "Emotion / Comfort" },
  retro:     { apiId: 8, apiName: "Retro / Vintage / Storytelling" },
};

// For MVP, only expose these 4 to the user
export const ACTIVE_STRATEGIES = ["hero", "seasonal", "comfort", "retro"];

export function getStrategyMeta(strategyId) {
  const meta = STRATEGY_META[strategyId];
  if (!meta) {
    throw new Error(`Unsupported strategy id: ${strategyId}`);
  }
  return meta;
}
