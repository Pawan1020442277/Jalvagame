import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: "1mb" }));

// All 10 Gemini API Keys
const KEYS = [
  process.env.GOOGLE_KEY_1, process.env.GOOGLE_KEY_2, process.env.GOOGLE_KEY_3,
  process.env.GOOGLE_KEY_4, process.env.GOOGLE_KEY_5, process.env.GOOGLE_KEY_6,
  process.env.GOOGLE_KEY_7, process.env.GOOGLE_KEY_8, process.env.GOOGLE_KEY_9,
  process.env.GOOGLE_KEY_10
];

// Local memory to track hits
const AI_STATS = KEYS.map((k, i) => ({
  id: i + 1,
  name: `AI-${i + 1}`,
  wins: 0,
  losses: 0,
  history: [],
  lastPrediction: null
}));

// Ask AI for a prediction
async function askModel(apiKey, lastResults) {
  if (!apiKey) return Math.floor(Math.random() * 10);
  try {
    const ai = new GoogleGenerativeAI(apiKey);
    const model = ai.getGenerativeModel({ model: "gemini-1.5-flash" });

    const prompt = `
You are predicting WinGo game numbers.
Last 10 results: ${JSON.stringify(lastResults)}
Predict the next single digit between 0–9 (integer only).
Reply in JSON like {"prediction": X}
`;

    const result = await model.generateContent(prompt);
    const text = result.response.text().trim();
    console.log("AI raw:", text);

    let num = null;
    try {
      const parsed = JSON.parse(text);
      num = parsed.prediction;
    } catch {
      const match = text.match(/\d/);
      if (match) num = Number(match[0]);
    }

    if (num === null || isNaN(num)) num = Math.floor(Math.random() * 10);
    return num;
  } catch (err) {
    console.log("AI Error:", err.message);
    return Math.floor(Math.random() * 10);
  }
}

// Predict for all 10 AIs
app.post("/api/predict-all", async (req, res) => {
  const lastResults = Array.isArray(req.body.lastResults)
    ? req.body.lastResults.slice(-10)
    : [];
  const out = await Promise.all(
    KEYS.map((key, i) =>
      askModel(key, lastResults).then(pred => ({ i, pred }))
    )
  );
  out.forEach(o => (AI_STATS[o.i].lastPrediction = o.pred));
  res.json({ success: true, predictions: AI_STATS });
});

// Compare predictions vs actual result
app.post("/api/report-actual", (req, res) => {
  const actual = Number(req.body.actual);
  if (Number.isNaN(actual)) return res.json({ success: false });
  AI_STATS.forEach(s => {
    const win = s.lastPrediction === actual;
    if (win) s.wins++;
    else s.losses++;
    s.history.unshift(win ? 1 : 0);
    if (s.history.length > 10) s.history.length = 10;
  });
  const ranked = AI_STATS.map(s => ({
    id: s.id,
    name: s.name,
    wins: s.wins,
    losses: s.losses,
    hitRate: s.wins / (s.wins + s.losses || 1),
    history: s.history,
  })).sort((a, b) => b.hitRate - a.hitRate);
  res.json({ success: true, ranked });
});

// Serve Frontend
app.use(express.static(path.join(__dirname, "public")));
app.listen(process.env.PORT || 10000, () =>
  console.log("✅ Server running on", process.env.PORT || 10000)
);