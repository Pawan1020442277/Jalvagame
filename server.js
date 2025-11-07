/**
 * server.js
 * - Polls WinGo history (every 30s)
 * - Sends last-10 to 10 Gemini APIs (keys from env)
 * - Stores pending predictions for the next period
 * - When new actual arrives, compares and updates stats
 * - Exposes /api/status for frontend to read live state
 */

import express from "express";
import cors from "cors";
import axios from "axios";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 10000;
const WIN_GO_API_URL = process.env.WIN_GO_API_URL?.trim() || "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json";

// Load keys from env
const KEYS = [
  process.env.GOOGLE_KEY_1, process.env.GOOGLE_KEY_2, process.env.GOOGLE_KEY_3,
  process.env.GOOGLE_KEY_4, process.env.GOOGLE_KEY_5, process.env.GOOGLE_KEY_6,
  process.env.GOOGLE_KEY_7, process.env.GOOGLE_KEY_8, process.env.GOOGLE_KEY_9,
  process.env.GOOGLE_KEY_10
];

// Basic AI stats structure
const AI_STATS = KEYS.map((k, i) => ({
  id: i + 1,
  name: `AI-${i + 1}`,
  key: k,
  wins: 0,
  losses: 0,
  accuracy: 0,
  history: [],           // last outcomes (1=win,0=loss)
  lastPrediction: null,  // { color, size }
  lastResultCorrect: null // boolean for most recent comparison
}));

// Cache / state
let cachedHistory = [];        // most recent list from WinGo (newest-first)
let lastSeenIssue = null;      // issueNumber of latest known actual
let pendingPredictions = null; // { issueExpected: <issue>, predictions: [{id, color, size}] }
const POLL_MS = Number(process.env.WINGO_POLL_INTERVAL_MS || 30_000);

// ---------------- Helpers ----------------

// parse WinGo response to array of items (newest-first)
function parseWinGoData(respData) {
  // as per sample: { data: { list: [ { issueNumber, number, color, ...}, ... ] }, code:0 }
  try {
    if (!respData) return [];
    const maybeList = respData?.data?.list ?? respData?.list ?? respData?.data ?? respData;
    if (Array.isArray(maybeList)) return maybeList.slice(); // return copy
    // if object containing list property
    for (const k of Object.keys(respData)) {
      if (Array.isArray(respData[k])) return respData[k].slice();
    }
    return [];
  } catch (e) {
    return [];
  }
}

// extract digit (number) and color and issueNumber
function normalizeEntry(entry) {
  // entry.number is string in sample
  const number = Number(entry.number ?? entry.num ?? entry.value ?? entry.n ?? 0);
  const color = (entry.color ?? entry.colour ?? entry.c ?? "").toString().toLowerCase();
  const issue = entry.issueNumber ?? entry.issue ?? entry.id ?? null;
  return { issue, number: Number.isNaN(number) ? null : number, color: color || null };
}

// fetch history from WinGo API
async function fetchWinGoHistory(limit = 10) {
  try {
    const resp = await axios.get(WIN_GO_API_URL, { timeout: 9000 });
    const list = parseWinGoData(resp.data);
    if (!list || !list.length) return [];
    // parse each and ensure newest-first. Based on sample API, first element seems newest.
    const normalized = list.map(normalizeEntry).filter(x => x.number !== null);
    return normalized.slice(0, limit); // keep most recent N
  } catch (err) {
    console.error("WinGo fetch error:", err.message || err);
    return [];
  }
}

// Rule helpers
function sizeOf(n) { return (n >= 5 ? "Big" : "Small"); }
function colorNormalizedFromNumber(n) {
  // mapping as you provided earlier: 0,2,4,6,8 -> Red; 1,3,5,7 -> Green; 9 -> Violet
  if (n === 9) return "Violet";
  if ([0,2,4,6,8].includes(n)) return "Red";
  return "Green";
}

// Ask one Gemini key for prediction (returns {color, size})
async function askGeminiWithPrompt(apiKey, last10, aiName, aiIndex) {
  // fallback random if no key
  if (!apiKey) {
    const randNum = Math.floor(Math.random() * 10);
    return { color: colorNormalizedFromNumber(randNum), size: sizeOf(randNum) };
  }

  try {
    const aiClient = new GoogleGenerativeAI(apiKey);
    // Use a robust prompt that enforces JSON output and describes rules
    const model = aiClient.getGenerativeModel({ model: "gemini-1.5-flash" });

    // Provide last10 as numbers with their colors/sizes for context
    const historyForPrompt = last10.map((e, idx) => ({
      idx: idx + 1,
      issue: e.issue,
      number: e.number,
      color: e.color || colorNormalizedFromNumber(e.number),
      size: sizeOf(e.number)
    }));

    const prompt = `
You are ${aiName}, an expert at analyzing WinGo 30s results.
Rules:
- Numbers are digits 0–9.
- Size rule: number >= 5 => "Big", else => "Small".
- Color rule: 0,2,4,6,8 => "Red"; 1,3,5,7 => "Green"; 9 => "Violet".

Given the last 10 results (newest-first):
${JSON.stringify(historyForPrompt, null, 2)}

Task:
1) Analyze patterns (frequency, runs, alternation, parity, hot/cold etc.).
2) Predict the NEXT result's COLOR (Red/Green/Violet) and SIZE (Big/Small).
3) MUST output EXACTLY one JSON object and NOTHING else, like:
{"color":"Red","size":"Big"}

Do not include explanation or any text besides the JSON.
Respond now.
`;

    // call the model
    const response = await model.generateContent({ contents: prompt });
    // response may have .response.text() or .text — try both
    let txt = "";
    try {
      txt = (response?.response?.text?.() ?? response?.text ?? "").trim();
    } catch {
      txt = (response?.text ?? "") + "";
      txt = txt.trim();
    }
    console.log(`AI-${aiIndex + 1} raw:`, txt);

    // Try to parse JSON strictly
    try {
      const parsed = JSON.parse(txt);
      const color = (parsed.color ?? parsed.Color ?? parsed.COLOR ?? "").toString();
      const size = (parsed.size ?? parsed.Size ?? parsed.SIZE ?? "").toString();
      // normalize capitalization
      const colorNorm = color.charAt(0).toUpperCase() + color.slice(1).toLowerCase();
      const sizeNorm = size.charAt(0).toUpperCase() + size.slice(1).toLowerCase();
      // quick validation
      if (["Red", "Green", "Violet"].includes(colorNorm) && ["Big", "Small"].includes(sizeNorm)) {
        return { color: colorNorm, size: sizeNorm };
      }
    } catch (e) {
      // not valid JSON — try to extract tokens
    }

    // fallback: try to find words in txt
    const lower = txt.toLowerCase();
    const colorGuess = lower.includes("violet") ? "Violet" : lower.includes("green") ? "Green" : lower.includes("red") ? "Red" : null;
    const sizeGuess = lower.includes("big") ? "Big" : lower.includes("small") ? "Small" : null;
    if (colorGuess && sizeGuess) return { color: colorGuess, size: sizeGuess };

    // last fallback: random deterministic from text digits if any
    const m = txt.match(/\d/);
    if (m) {
      const d = Number(m[0]);
      return { color: colorNormalizedFromNumber(d), size: sizeOf(d) };
    }

    // final fallback random
    const rand = Math.floor(Math.random() * 10);
    return { color: colorNormalizedFromNumber(rand), size: sizeOf(rand) };
  } catch (err) {
    console.error("Gemini call error:", err.message || err);
    const rand = Math.floor(Math.random() * 10);
    return { color: colorNormalizedFromNumber(rand), size: sizeOf(rand) };
  }
}

// ---------------- Polling & prediction flow ----------------
//
// Flow:
// - On start: fetch history; set lastSeenIssue = latest issue; create initial pendingPredictions by asking AIs for next period.
// - Each POLL_MS:
//    - fetch history
//    - if newestIssue !== lastSeenIssue -> that means a new actual arrived
//         -> compare this actual with pendingPredictions (the ones we had earlier) and update wins/losses
//         -> set lastSeenIssue = newestIssue
//    - generate new pendingPredictions for the next period (ask Gemini with current history)
// - Expose /api/status to return live AI_STATS, cachedHistory, pendingPredictions, lastSeenIssue

async function generatePredictionsForNextPeriod(history) {
  // returns array of {id, color, size}
  const results = await Promise.all(
    AI_STATS.map(async (ai, i) => {
      const p = await askGeminiWithPrompt(ai.key, history, ai.name, i);
      return { id: ai.id, name: ai.name, color: p.color, size: p.size };
    })
  );
  // store as pending
  const nextPred = {
    generatedAt: Date.now(),
    predictions: results
  };
  pendingPredictions = nextPred;
  // update lastPrediction (for UI)
  results.forEach(r => {
    const idx = r.id - 1;
    AI_STATS[idx].lastPrediction = { color: r.color, size: r.size };
    AI_STATS[idx].lastResultCorrect = null; // unknown until actual arrives
  });
  return nextPred;
}

async function compareActualToPending(actualEntry) {
  // actualEntry: { issue, number, color }
  if (!pendingPredictions || !pendingPredictions.predictions) return;
  const actualColor = (actualEntry.color ?? colorNormalizedFromNumber(actualEntry.number)).toString();
  const actualSize = sizeOf(actualEntry.number);
  // compare each AI's lastPrediction
  pendingPredictions.predictions.forEach(pred => {
    const idx = pred.id - 1;
    const stat = AI_STATS[idx];
    const predicted = stat.lastPrediction || { color: pred.color, size: pred.size };
    const colorOk = (predicted.color || "").toString().toLowerCase() === (actualColor || "").toString().toLowerCase();
    const sizeOk = (predicted.size || "").toString().toLowerCase() === (actualSize || "").toString().toLowerCase();
    // Decide winning rule: you said you want both checks — we can treat as need BOTH match to count as a full win,
    // but it's more informative to track color and size separately. Here we'll treat "win" as BOTH match.
    const bothCorrect = colorOk && sizeOk;
    if (bothCorrect) stat.wins++;
    else stat.losses++;
    stat.history.unshift(bothCorrect ? 1 : 0);
    if (stat.history.length > 100) stat.history.length = 100;
    stat.total = (stat.wins + stat.losses);
    stat.accuracy = stat.total ? Math.round((stat.wins / stat.total) * 100) : 0;
    stat.lastResultCorrect = bothCorrect;
  });
  // clear pendingPredictions after comparison (we will generate fresh after)
  pendingPredictions.comparedAt = Date.now();
}

// ---------------- Main polling loop ----------------

async function pollLoopOnce() {
  try {
    // fetch current history (newest-first)
    const history = await fetchWinGoHistory(10);
    if (!history || history.length === 0) {
      // nothing — keep existing
      return;
    }
    cachedHistory = history.slice(); // newest-first
    const newest = cachedHistory[0];
    const newestIssue = newest.issue;

    // first-time init
    if (!lastSeenIssue) {
      lastSeenIssue = newestIssue;
      // generate initial predictions for next period (based on current history)
      await generatePredictionsForNextPeriod(cachedHistory);
      console.log("Initial predictions generated.");
      return;
    }

    // If new actual arrived (issue changed)
    if (newestIssue !== lastSeenIssue) {
      // newest is the most recent result (the actual that just arrived)
      console.log("New actual detected:", newestIssue, newest.number);
      // Compare actual with the pending predictions from previous cycle
      await compareActualToPending(newest);
      // update lastSeenIssue
      lastSeenIssue = newestIssue;
      // generate new predictions for next period (based on updated history)
      await generatePredictionsForNextPeriod(cachedHistory);
      console.log("Generated new predictions for next period after actual.");
    } else {
      // no new actual — ensure we still refresh pending predictions occasionally (optional)
      // If no pending predictions exist (null) create one
      if (!pendingPredictions) {
        await generatePredictionsForNextPeriod(cachedHistory);
      }
      // else: do nothing, wait for next poll
    }
  } catch (err) {
    console.error("Poll loop error:", err.message || err);
  }
}

// start poller
(async () => {
  // populate once immediately
  await pollLoopOnce();
  // then start interval
  setInterval(pollLoopOnce, POLL_MS);
})();

// ---------------- HTTP endpoints ----------------

// status: return AI_STATS, cachedHistory (most recent first), pendingPredictions, lastSeenIssue
app.get("/api/status", (req, res) => {
  const aiInfo = AI_STATS.map(s => ({
    id: s.id,
    name: s.name,
    wins: s.wins,
    losses: s.losses,
    accuracy: s.accuracy,
    lastPrediction: s.lastPrediction,
    lastResultCorrect: s.lastResultCorrect,
    history: s.history.slice(0, 20)
  }));
  res.json({
    success: true,
    lastSeenIssue,
    cachedHistory,
    pendingPredictions,
    aiInfo,
    serverTime: Date.now()
  });
});

// convenience endpoint to force refresh/generate predictions (admin)
app.post("/api/force", async (req, res) => {
  try {
    const history = await fetchWinGoHistory(10);
    const pred = await generatePredictionsForNextPeriod(history);
    res.json({ success: true, pendingPredictions: pred, history });
  } catch (e) {
    res.status(500).json({ success: false, error: String(e) });
  }
});

app.listen(PORT, () => console.log(`🚀 Server running on ${PORT} — polling ${WIN_GO_API_URL} every ${POLL_MS}ms`));