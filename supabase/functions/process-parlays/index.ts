import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import * as XLSX from "npm:xlsx@0.18.5";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// ============================================================================
// UTILITY FUNCTIONS (Direct port from Python)
// ============================================================================

function americanToImplied(odds: number): number {
  if (odds < 0) {
    return -odds / (-odds + 100.0);
  } else {
    return 100.0 / (odds + 100.0);
  }
}

function impliedToDecimal(p: number): number {
  if (p <= 0.0) return Infinity;
  return 1.0 / p;
}

function decimalToAmerican(d: number): string {
  if (d <= 1.0) return "-INF";
  if (d >= 2.0) {
    return `+${Math.round((d - 1.0) * 100.0)}`;
  }
  return `-${Math.round(100.0 / (d - 1.0))}`;
}

function parseSide(proposition: string): string | null {
  if (!proposition) return null;
  const s = proposition.toLowerCase();
  if (s.includes("over")) return "OVER";
  if (s.includes("under")) return "UNDER";
  return null;
}

function parseProbabilityValue(val: any): number {
  if (val === null || val === undefined || val === "") return NaN;
  
  if (typeof val === "string") {
    let s = val.trim();
    if (s.endsWith("%")) {
      s = s.slice(0, -1).trim();
      return parseFloat(s) / 100.0;
    }
    val = parseFloat(s);
  } else {
    val = parseFloat(val);
  }
  
  if (isNaN(val)) return NaN;
  
  // If > 1, assume it's a percent (e.g., 69)
  if (val > 1.0) {
    val = val / 100.0;
  }
  
  if (val < 0.0 || val > 1.0) {
    throw new Error(`Probability out of bounds: ${val}`);
  }
  return val;
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

function ensurePositiveDefinite(mat: number[][], eps: number = 1e-6): number[][] {
  const n = mat.length;
  // Simple positive definite enforcement by ensuring diagonal dominance
  const result = mat.map(row => [...row]);
  
  for (let i = 0; i < n; i++) {
    result[i][i] = 1.0;
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        result[i][j] = Math.max(-0.999, Math.min(0.999, result[i][j]));
      }
    }
  }
  
  return result;
}

function choleskyDecomposition(matrix: number[][]): number[][] {
  const n = matrix.length;
  const L: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      
      if (j === i) {
        for (let k = 0; k < j; k++) {
          sum += L[j][k] * L[j][k];
        }
        L[j][j] = Math.sqrt(Math.max(matrix[j][j] - sum, 1e-10));
      } else {
        for (let k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
        L[i][j] = (matrix[i][j] - sum) / Math.max(L[j][j], 1e-10);
      }
    }
  }
  
  return L;
}

// Normal CDF approximation
function normalCDF(x: number): number {
  const t = 1.0 / (1.0 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2.0);
  const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1.0 - prob : prob;
}

// Beta distribution sampling (Box-Muller inspired approximation)
function betaSample(alpha: number, beta: number): number {
  // Simple beta approximation using normal approximation for large alpha, beta
  const mean = alpha / (alpha + beta);
  const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
  const std = Math.sqrt(variance);
  
  // Generate normal random variable
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  
  let sample = mean + std * z;
  sample = Math.max(0.001, Math.min(0.999, sample));
  return sample;
}

// ============================================================================
// HIERARCHICAL MINUTES ESTIMATION
// ============================================================================

function hierarchicalMinutesEstimate(
  data: any[],
  defaultMinutes: number = 25.0,
  shrinkageStrength: number = 10.0
): any[] {
  const lastMinutesData = data.map(row => {
    const lm = row.LastMinutes;
    if (lm === null || lm === undefined || lm === "") return NaN;
    if (typeof lm === "string" && lm.includes(",")) {
      const parts = lm.split(",").map((p: string) => parseFloat(p.trim())).filter((n: number) => !isNaN(n));
      return parts.length > 0 ? parts.reduce((a: number, b: number) => a + b, 0) / parts.length : NaN;
    }
    return parseFloat(lm);
  });
  
  const validLastMinutes = lastMinutesData.filter(v => !isNaN(v));
  const globalMean = validLastMinutes.length > 0 
    ? validLastMinutes.reduce((a, b) => a + b, 0) / validLastMinutes.length 
    : defaultMinutes;
  
  // Calculate team means
  const teamGroups: { [key: string]: number[] } = {};
  data.forEach((row, i) => {
    const team = row.Team || "UNKNOWN";
    if (!teamGroups[team]) teamGroups[team] = [];
    if (!isNaN(lastMinutesData[i])) {
      teamGroups[team].push(lastMinutesData[i]);
    }
  });
  
  const teamMeans: { [key: string]: number } = {};
  Object.keys(teamGroups).forEach(team => {
    const vals = teamGroups[team];
    teamMeans[team] = vals.length > 0 
      ? vals.reduce((a, b) => a + b, 0) / vals.length 
      : globalMean;
  });
  
  return data.map((row, i) => {
    const pmin = lastMinutesData[i];
    const team = row.Team || "UNKNOWN";
    const teamMean = teamMeans[team] || globalMean;
    
    let estMinutes: number;
    if (!isNaN(pmin)) {
      const wPlayer = 1.0;
      const wTeam = shrinkageStrength;
      const wGlobal = 1.0;
      estMinutes = (wPlayer * pmin + wTeam * teamMean + wGlobal * globalMean) / 
                   (wPlayer + wTeam + wGlobal);
    } else {
      estMinutes = teamMean;
    }
    
    return { ...row, EstMinutes: estMinutes };
  });
}

// ============================================================================
// PROBABILITY MODELING
// ============================================================================

const HISTORY_COLS = [
  "Last 5 Hit Rate",
  "Last 10 Hit Rate",
  "Last 20 Hit Rate",
  "2025 Hit Rate",
];

function computeModelProbabilities(data: any[], confidenceInModel: number = 0.35): any[] {
  return data.map(row => {
    // Weighted historical probability
    const histProb = Math.max(0.001, Math.min(0.999,
      0.10 * (parseProbabilityValue(row["Last 5 Hit Rate"]) || 0) +
      0.45 * (parseProbabilityValue(row["Last 10 Hit Rate"]) || 0) +
      0.25 * (parseProbabilityValue(row["Last 20 Hit Rate"]) || 0) +
      0.20 * (parseProbabilityValue(row["2025 Hit Rate"]) || 0)
    ));
    
    // Projection-based probability
    let projProb = NaN;
    const proj = parseFloat(row.Projection);
    const line = parseFloat(row.Line);
    
    if (!isNaN(proj) && !isNaN(line)) {
      const std = !isNaN(parseFloat(row.StdDev)) 
        ? parseFloat(row.StdDev) 
        : Math.max(0.1, Math.abs(proj) * 0.18);
      const z = (proj - line) / std;
      const pOver = normalCDF(z);
      const side = parseSide(row.Proposition);
      projProb = side === "UNDER" ? 1.0 - pOver : pOver;
    }
    
    // Combine historical and projection
    let modelProb = !isNaN(projProb) ? projProb : histProb;
    modelProb = Math.max(0.001, Math.min(0.999, modelProb));
    
    // Market implied probability
    let impliedProb = parseProbabilityValue(row["Implied Probability"]);
    if (isNaN(impliedProb)) {
      impliedProb = americanToImplied(parseFloat(row.Odds));
    }
    impliedProb = Math.max(0.001, Math.min(0.999, impliedProb));
    
    // Blend market with model
    const finalProb = Math.max(0.001, Math.min(0.999,
      (1.0 - confidenceInModel) * impliedProb + confidenceInModel * modelProb
    ));
    
    return {
      ...row,
      histProb,
      projProb,
      modelProb,
      impliedProb,
      finalProb,
      nEff: 8.0
    };
  });
}

// ============================================================================
// CORRELATION MATRIX
// ============================================================================

function scaleCorrelationByPace(baseCorr: number, rowI: any, rowJ: any): number {
  const pi = parseFloat(rowI.Pace);
  const pj = parseFloat(rowJ.Pace);
  
  let mult = 1.0;
  if (!isNaN(pi) && !isNaN(pj)) {
    const avgPace = 0.5 * (pi + pj);
    const baseline = 100.0;
    mult += 0.004 * (avgPace - baseline);
    mult = Math.max(0.6, Math.min(1.6, mult));
  }
  return baseCorr * mult;
}

function getPositionCorr(posI: string, posJ: string, defaultSameTeam: number = 0.30): number {
  if (!posI || !posJ) return defaultSameTeam;
  
  const pi = posI.toUpperCase();
  const pj = posJ.toUpperCase();
  
  if ((pi.includes("C") && pj.includes("C")) || (pi.includes("PF") && pj.includes("PF"))) {
    return 0.45;
  }
  if ((pi.includes("PG") && (pj.includes("SG") || pj.includes("SF") || pj.includes("PG"))) ||
      (pj.includes("PG") && (pi.includes("SG") || pi.includes("SF") || pi.includes("PG")))) {
    return 0.40;
  }
  if ((pi.includes("PG") && pj.includes("C")) || (pj.includes("PG") && pi.includes("C"))) {
    return 0.20;
  }
  if (((pi.includes("SF") || pi.includes("SG")) && (pj.includes("PF") || pj.includes("C"))) ||
      ((pj.includes("SF") || pj.includes("SG")) && (pi.includes("PF") || pi.includes("C")))) {
    return 0.25;
  }
  return defaultSameTeam;
}

function buildConditionalCorrMatrix(
  data: any[],
  rhoSameTeam: number = 0.30,
  rhoOppTeam: number = 0.10,
  rhoSamePlayer: number = 0.95
): number[][] {
  const n = data.length;
  const corr: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    corr[i][i] = 1.0;
  }
  
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let baseR = 0.0;
      
      if (data[i].Game === data[j].Game) {
        if (data[i].Team === data[j].Team) {
          baseR = getPositionCorr(data[i].Position, data[j].Position, rhoSameTeam);
          
          if (data[i].Player === data[j].Player) {
            baseR = Math.max(baseR, rhoSamePlayer);
          }
        } else {
          baseR = rhoOppTeam;
        }
      }
      
      const r = scaleCorrelationByPace(baseR, data[i], data[j]);
      corr[i][j] = Math.max(-0.999, Math.min(0.999, r));
      corr[j][i] = corr[i][j];
    }
  }
  
  return ensurePositiveDefinite(corr);
}

// ============================================================================
// MONTE CARLO SIMULATION
// ============================================================================

function betaParamsFromMeanVar(mean: number, variance: number): [number, number] {
  mean = Math.max(1e-6, Math.min(1.0 - 1e-6, mean));
  const maxVar = mean * (1.0 - mean) - 1e-8;
  variance = Math.max(1e-8, Math.min(maxVar, variance));
  const m = (mean * (1.0 - mean) / variance) - 1.0;
  const alpha = Math.max(1e-6, mean * m);
  const beta = Math.max(1e-6, (1.0 - mean) * m);
  return [alpha, beta];
}

function monteCarloCopula(
  data: any[],
  corrMatrix: number[][],
  trials: number,
  seed: number = 42
): number[][] {
  const n = data.length;
  
  // Prepare Beta parameters
  const betaParams: [number, number][] = data.map(row => {
    const mean = row.finalProb;
    const nEff = row.nEff;
    const variance = Math.max(1e-6, mean * (1.0 - mean) / (nEff + 1.0));
    return betaParamsFromMeanVar(mean, variance);
  });
  
  const L = choleskyDecomposition(corrMatrix);
  
  // Preallocate output more efficiently
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    out.push(new Array(trials));
  }
  
  // Set random seed (simple approach)
  let rngState = seed;
  const seededRandom = () => {
    rngState = Math.sin(rngState) * 10000;
    return rngState - Math.floor(rngState);
  };
  
  // Process trials in batches to reduce memory pressure
  const batchSize = 1000;
  for (let batch = 0; batch < trials; batch += batchSize) {
    const currentBatchSize = Math.min(batchSize, trials - batch);
    
    for (let t = 0; t < currentBatchSize; t++) {
      const trialIdx = batch + t;
      
      // Generate correlated normals
      const Z: number[] = [];
      for (let i = 0; i < n; i++) {
        const u1 = seededRandom();
        const u2 = seededRandom();
        Z.push(Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
      }
      
      // Apply Cholesky: correlated = L * Z
      const correlated: number[] = new Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          correlated[i] += L[i][j] * Z[j];
        }
      }
      
      // Convert to uniforms and sample
      for (let i = 0; i < n; i++) {
        const u = normalCDF(correlated[i]);
        const [alpha, beta] = betaParams[i];
        const pSample = betaSample(alpha, beta);
        out[i][trialIdx] = u < pSample ? 1 : 0;
      }
    }
  }
  
  return out;
}

// ============================================================================
// PARLAY EVALUATION
// ============================================================================

function kellyFromPAndDecimal(p: number, decimal: number, cap: number = 0.10): number {
  const b = decimal - 1.0;
  if (b <= 0.0) return 0.0;
  const q = 1.0 - p;
  const f = (b * p - q) / b;
  if (f <= 0.0) return 0.0;
  return Math.min(f, cap);
}

function evaluateParlay(
  indices: number[],
  outMatrix: number[][],
  data: any[],
  kellyCap: number
): any | null {
  // Check if all legs hit in each trial
  const trials = outMatrix[0].length;
  let hits = 0;
  
  for (let t = 0; t < trials; t++) {
    let allHit = true;
    for (const idx of indices) {
      if (outMatrix[idx][t] === 0) {
        allHit = false;
        break;
      }
    }
    if (allHit) hits++;
  }
  
  const hitProb = hits / trials;
  if (hitProb <= 0.0) return null;
  
  const legs = indices.map(i => data[i]);
  const impliedProbs = legs.map(leg => leg.impliedProb);
  const impliedDecimals = impliedProbs.reduce((prod, p) => prod * impliedToDecimal(p), 1.0);
  const ev = hitProb * impliedDecimals - 1.0;
  const kelly = kellyFromPAndDecimal(hitProb, impliedDecimals, kellyCap);
  
  return {
    indices,
    legs,
    hitProb,
    impliedDecimal: impliedDecimals,
    ev,
    kelly
  };
}

function* combinations<T>(array: T[], k: number): Generator<T[]> {
  if (k === 0) {
    yield [];
    return;
  }
  
  for (let i = 0; i <= array.length - k; i++) {
    for (const combo of combinations(array.slice(i + 1), k - 1)) {
      yield [array[i], ...combo];
    }
  }
}

function searchTopParlays(
  data: any[],
  outMatrix: number[][],
  config: any
): any[] {
  // Score candidates
  const scored = data.map((row, i) => ({
    index: i,
    score: row.finalProb + 0.5 * (row.finalProb - row.impliedProb)
  }));
  
  scored.sort((a, b) => b.score - a.score);
  const candidates = scored.slice(0, config.poolLimit).map(s => s.index);
  
  const results: any[] = [];
  let combinationsChecked = 0;
  const maxCombinations = 5000; // Hard limit to prevent timeout
  
  for (let r = 2; r <= config.maxLegs; r++) {
    for (const combo of combinations(candidates, r)) {
      if (combinationsChecked++ > maxCombinations) break;
      
      if (config.avoidSamePlayer) {
        const players = combo.map(i => data[i].Player);
        if (new Set(players).size !== players.length) continue;
      }
      
      const stats = evaluateParlay(combo, outMatrix, data, config.kellyCap);
      if (stats && stats.ev > 0.0) {
        results.push(stats);
      }
    }
    if (combinationsChecked > maxCombinations) break;
  }
  
  results.sort((a, b) => b.ev - a.ev);
  return results.slice(0, config.topK);
}

// ============================================================================
// DATA PREPROCESSING
// ============================================================================

function preprocessPropsData(rows: any[]): any[] {
  const required = ["Team", "Player", "Game", "Proposition", "Line", "Odds"];
  const missing = required.filter(col => !rows[0].hasOwnProperty(col));
  if (missing.length > 0) {
    throw new Error(`Missing required columns: ${missing.join(", ")}`);
  }
  
  return rows.map(row => {
    const processed: any = { ...row };
    
    // Parse numeric fields
    processed.Line = parseFloat(row.Line);
    processed.Odds = parseFloat(row.Odds);
    
    if (isNaN(processed.Odds)) {
      throw new Error("Invalid Odds value found");
    }
    
    return processed;
  });
}

// ============================================================================
// MAIN ENGINE PIPELINE
// ============================================================================

function runParlayEngine(data: any[], config: any): any {
  console.log("Processing props data...");
  let processedData = preprocessPropsData(data);
  
  console.log("Estimating minutes...");
  processedData = hierarchicalMinutesEstimate(processedData);
  
  console.log("Computing model probabilities...");
  processedData = computeModelProbabilities(processedData, config.confidenceInModel);
  
  console.log("Building correlation matrix...");
  const corr = buildConditionalCorrMatrix(
    processedData,
    config.rhoSameTeam,
    config.rhoOppTeam,
    config.rhoSamePlayer
  );
  
  console.log(`Running Monte Carlo: ${config.trials} trials...`);
  const outMatrix = monteCarloCopula(processedData, corr, config.trials, config.randomSeed);
  
  console.log("Searching for +EV parlays...");
  const parlays = searchTopParlays(processedData, outMatrix, config);
  
  console.log(`Found ${parlays.length} +EV parlays`);
  
  // Build JSON entries
  const parlayEntries = parlays.map((parlay, rank) => {
    const legs = parlay.indices.map((idx: number) => {
      const row = processedData[idx];
      return {
        row_index: idx,
        team: row.Team,
        player: row.Player,
        game: row.Game,
        proposition: row.Proposition,
        line: row.Line,
        odds: row.Odds,
        implied_probability: row.impliedProb,
        model_probability: row.finalProb,
      };
    });
    
    return {
      label: `top_ev_${rank + 1}`,
      num_legs: parlay.indices.length,
      ev: parlay.ev,
      hit_probability: parlay.hitProb,
      implied_decimal: parlay.impliedDecimal,
      implied_american: decimalToAmerican(parlay.impliedDecimal),
      kelly_fraction: parlay.kelly,
      legs
    };
  });
  
  return {
    generated_at: new Date().toISOString(),
    source_file: "uploaded_file.xlsx",
    engine_config: config,
    num_parlays: parlayEntries.length,
    parlays: parlayEntries
  };
}

// ============================================================================
// EDGE FUNCTION HANDLER
// ============================================================================

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      throw new Error('No file provided');
    }
    
    console.log(`Processing file: ${file.name}`);
    
    // Read Excel file
    const arrayBuffer = await file.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: 'array' });
    const sheetName = workbook.SheetNames[0];
    const worksheet = workbook.Sheets[sheetName];
    const jsonData = XLSX.utils.sheet_to_json(worksheet);
    
    console.log(`Loaded ${jsonData.length} rows from Excel`);
    
    // Engine configuration (heavily optimized for edge function limits)
    const config = {
      trials: 3000,  // Significantly reduced for CPU constraints
      batchSize: 1000,
      maxLegs: 4,
      poolLimit: 25,  // Reduced to limit combination searches
      topK: 15,
      confidenceInModel: 0.35,
      rhoSameTeam: 0.30,
      rhoOppTeam: 0.10,
      rhoSamePlayer: 0.95,
      kellyCap: 0.10,
      avoidSamePlayer: true,
      randomSeed: 42
    };
    
    // Run the parlay engine
    const result = runParlayEngine(jsonData, config);
    
    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
    
  } catch (error: any) {
    console.error('Error processing parlays:', error);
    return new Response(
      JSON.stringify({ error: error.message || 'Internal server error' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});
