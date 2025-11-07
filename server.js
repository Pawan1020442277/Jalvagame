/**
 * server.js (production-ready)
 * - Polls WinGo history (via proxy fallback)
 * - Generates predictions using 10 Gemini keys
 * - Stores pending predictions and compares when a new actual arrives
 * - Exposes /api/status and /api/force for frontend
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
const WIN_GO_API_URL = (process.env.WIN_GO_API_URL || "https://api.allorigins.win/raw?url=https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json").trim();
const POLL_MS = Number(process.env.WINGO_POLL_INTERVAL_MS || 30000);

// Load keys
const KEYS = [
  process.env.GOOGLE_KEY_1, process.env.GOOGLE_KEY_2, process.env.GOOGLE_KEY_3,
  process.env.GOOGLE_KEY_4, process.env.GOOGLE_KEY_5, process.env.GOOGLE_KEY_6,
  process.env.GOOGLE_KEY_7, process.env.GOOGLE_KEY_8, process.env.GOOGLE_KEY_9,
  process.env.GOOGLE_KEY_10
];

// AI stats init
const AI_STATS = KEYS.map((k, i) => ({
  id: i + 1,
  name: `AI-${i + 1}`,
  key: k,
  wins: 0,
  losses: 0,
  total: 0,
  accuracy: 0,
  history: [],
  lastPrediction: null,
  lastResultCorrect: null
}));

// State
let cachedHistory = [];        // newest-first normalized entries [{issue,number,color}, ...]
let lastSeenIssue = null;
let pendingPredictions = null; // {generatedAt, predictions: [{id,name,color,size}]}

// ---------- Helpers ----------
function parseWinGoData(respData) {
  try {
    if (!respData) return [];
    const maybeList = respData?.data?.list ?? respData?.list ?? respData?.data ?? respData;
    if (Array.isArray(maybeList)) return maybeList.slice();
    for (const k of Object.keys(respData)) {
      if (Array.isArray(respData[k])) return respData[k].slice();
    }
    return [];
  } catch (e) {
    return [];
  }
}
function normalizeEntry(entry) {
  const number = Number(entry.number ?? entry.num ?? entry.value ?? entry.n ?? null);
  const color = (entry.color ?? entry.colour ?? entry.c ?? "").toString().toLowerCase();
  const issue = entry.issueNumber ?? entry.issue ?? entry.id ?? null;
  return { issue, number: Number.isNaN(number) ? null : number, color: color || null };
}
function sizeOf(n) { return (n >= 5 ? "Big" : "Small"); }
function colorNormalizedFromNumber(n) {
  if (n === 9) return "Violet";
  if ([0,2,4,6,8].includes(n)) return "Red";
  return "Green";
}

// ---------- Robust fetch with proxy fallback ----------
async function fetchWinGoHistory(limit = 10) {
  // try configured URL first (WIN_GO_API_URL) which may already be proxy
  const urlsToTry = [
    WIN_GO_API_URL,
    "https://api.allorigins.win/raw?url=" + encodeURIComponent("https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"),
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
  ];

  for (const url of urlsToTry) {
    if (!url) continue;
    try {
      const resp = await axios.get(url, {
        timeout: 12000,
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
          "Accept": "application/json, text/plain, */*",
          "Referer": "https://draw.ar-lottery01.com/"
        }
      });
      const list = parseWinGoData(resp.data);
      const normalized = Array.isArray(list) ? list.map(normalizeEntry).filter(x => x.number !== null) : [];
      if (normalized.length > 0) {
        console.log(`✅ WinGo fetched via ${url.includes("allorigins") ? "AllOrigins Proxy" : (url.includes("draw.ar-lottery01") ? "Direct" : "Configured URL")}:`, normalized.slice(0,3));
        return normalized.slice(0, limit);
      } else {
        console.warn(`⚠️ WinGo returned empty for ${url}`);
      }
    } catch (err) {
      console.warn(`⚠️ Fetch failed for ${url}:`, err.message || err);
    }
  }

  console.error("🚫 All WinGo fetch attempts failed or returned empty.");
  return [];
}

// ---------- Gemini prompt + call ----------
async function askGeminiWithPrompt(apiKey, last10, aiName, aiIndex) {
  if (!apiKey) {
    const rand = Math.floor(Math.random() * 10);
    return { color: colorNormalizedFromNumber(rand), size: sizeOf(rand) };
  }
  try {
    const aiClient = new GoogleGenerativeAI(apiKey);
    const model = aiClient.getGenerativeModel({ model: "gemini-1.5-flash" });

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

    const response = await model.generateContent({ contents: prompt });
    let txt = "";
    try { txt = (response?.response?.text?.() ?? response?.text ?? "").trim(); } catch { txt = (response?.text ?? "").toString().trim(); }
    console.log(`AI-${aiIndex + 1} raw:`, txt);

    try {
      const parsed = JSON.parse(txt);
      const color = (parsed.color ?? parsed.Color ?? "").toString();
      const size = (parsed.size ?? parsed.Size ?? "").toString();
      const colorNorm = color.charAt(0).toUpperCase() + color.slice(1).toLowerCase();
      const sizeNorm = size.charAt(0).toUpperCase() + size.slice(1).toLowerCase();
      if (["Red", "Green", "Violet"].includes(colorNorm) && ["Big", "Small"].includes(sizeNorm)) {
        return { color: colorNorm, size: sizeNorm };
      }
    } catch (e) {
      // ignore JSON parse error, fallback next
    }

    const lower = txt.toLowerCase();
    const colorGuess = lower.includes("violet") ? "Violet" : lower.includes("green") ? "Green" : lower.includes("red") ? "Red" : null;
    const sizeGuess = lower.includes("big") ? "Big" : lower.includes("small") ? "Small" : null;
    if (colorGuess && sizeGuess) return { color: colorGuess, size: sizeGuess };

    const m = txt.match(/\d/);
    if (m) {
      const d = Number(m[0]);
      return { color: colorNormalizedFromNumber(d), size: sizeOf(d) };
    }

    const rand = Math.floor(Math.random() * 10);
    return { color: colorNormalizedFromNumber(rand), size: sizeOf(rand) };
  } catch (err) {
    console.error("Gemini call error:", err.message || err);
    const rand = Math.floor(Math.random() * 10);
    return { color: colorNormalizedFromNumber(rand), size: sizeOf(rand) };
  }
}

// ---------- Prediction generation + compare ----------
async function generatePredictionsForNextPeriod(history) {
  const results = await Promise.all(
    AI_STATS.map(async (ai, i) => {
      const p = await askGeminiWithPrompt(ai.key, history, ai.name, i);
      return { id: ai.id, name: ai.name, color: p.color, size: p.size };
    })
  );
  pendingPredictions = { generatedAt: Date.now(), predictions: results };
  results.forEach(r => {
    const idx = r.id - 1;
    AI_STATS[idx].lastPrediction = { color: r.color, size: r.size };
    AI_STATS[idx].lastResultCorrect = null;
  });
  return pendingPredictions;
}

async function compareActualToPending(actualEntry) {
  if (!pendingPredictions || !pendingPredictions.predictions) return;
  const actualColor = (actualEntry.color ?? colorNormalizedFromNumber(actualEntry.number)).toString();
  const actualSize = sizeOf(actualEntry.number);
  pendingPredictions.predictions.forEach(pred => {
    const idx = pred.id - 1;
    const stat = AI_STATS[idx];
    const predicted = stat.lastPrediction || { color: pred.color, size: pred.size };
    const colorOk = (predicted.color || "").toString().toLowerCase() === (actualColor || "").toString().toLowerCase();
    const sizeOk = (predicted.size || "").toString().toLowerCase() === (actualSize || "").toString().toLowerCase();
    const bothCorrect = colorOk && sizeOk;
    if (bothCorrect) stat.wins++; else stat.losses++;
    stat.history.unshift(bothCorrect ? 1 : 0);
    if (stat.history.length > 200) stat.history.length = 200;
    stat.total = (stat.wins + stat.losses);
    stat.accuracy = stat.total ? Math.round((stat.wins / stat.total) * 100) : 0;
    stat.lastResultCorrect = bothCorrect;
  });
  pendingPredictions.comparedAt = Date.now();
}

// ---------- Poll loop ----------
async function pollLoopOnce() {
  try {
    const history = await fetchWinGoHistory(10);
    if (!history || history.length === 0) {
      console.log("No history fetched this cycle.");
      return;
    }
    cachedHistory = history.slice();
    const newest = cachedHistory[0];
    const newestIssue = newest.issue;

    if (!lastSeenIssue) {
      lastSeenIssue = newestIssue;
      await generatePredictionsForNextPeriod(cachedHistory);
      console.log("Initial predictions generated.");
      return;
    }

    if (newestIssue !== lastSeenIssue) {
      console.log("New actual detected:", newestIssue, newest.number);
      await compareActualToPending(newest);
      lastSeenIssue = newestIssue;
      await generatePredictionsForNextPeriod(cachedHistory);
      console.log("Generated new predictions for next period after actual.");
    } else {
      if (!pendingPredictions) {
        await generatePredictionsForNextPeriod(cachedHistory);
      }
    }
  } catch (err) {
    console.error("Poll loop error:", err.message || err);
  }
}

// start poller
(async () => {
  await pollLoopOnce();
  setInterval(pollLoopOnce, POLL_MS);
})();

// ---------- HTTP endpoints ----------
app.get("/api/status", (req, res) => {
  const aiInfo = AI_STATS.map(s => ({
    id: s.id, name: s.name, wins: s.wins, losses: s.losses, accuracy: s.accuracy,
    lastPrediction: s.lastPrediction, lastResultCorrect: s.lastResultCorrect, history: s.history.slice(0,20)
  }));
  res.json({ success: true, lastSeenIssue, cachedHistory, pendingPredictions, aiInfo, serverTime: Date.now() });
});

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