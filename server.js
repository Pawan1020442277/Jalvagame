import express from "express";
import cors from "cors";
import axios from "axios";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const PORT = process.env.PORT || 10000;
const API_URL = process.env.WIN_GO_API_URL;
const KEYS = [
  process.env.GOOGLE_KEY_1, process.env.GOOGLE_KEY_2, process.env.GOOGLE_KEY_3,
  process.env.GOOGLE_KEY_4, process.env.GOOGLE_KEY_5, process.env.GOOGLE_KEY_6,
  process.env.GOOGLE_KEY_7, process.env.GOOGLE_KEY_8, process.env.GOOGLE_KEY_9,
  process.env.GOOGLE_KEY_10
];

const AIs = KEYS.map((k, i) => ({
  id: i + 1,
  name: `AI-${i + 1}`,
  key: k,
  last: { color: "—", size: "—" },
  accuracy: 0
}));

// ---------- Fetch last 10 real results ----------
async function fetchLast10Results() {
  try {
    const { data } = await axios.get(API_URL);
    const arr = Array.isArray(data.data)
      ? data.data
      : Array.isArray(data.results)
      ? data.results
      : data;
    return arr.slice(-10).map(r => Number(r.number || r.result || r.value || 0));
  } catch (e) {
    console.error("❌ WinGo API Error:", e.message);
    return [];
  }
}

// ---------- Ask Gemini ----------
async function askGemini(key, last10, aiName) {
  try {
    const ai = new GoogleGenerativeAI(key);
    const model = ai.getGenerativeModel({ model: "gemini-1.5-flash" });

    const prompt = `
You are ${aiName}, an expert in analyzing WinGo lottery patterns.

The game logic:
- Numbers range 0–9.
- If number ≥ 5 → "Big", else → "Small".
- Colors follow:
  0,2,4,6,8 = "Red"
  1,3,5,7 = "Green"
  9 = "Violet"

Here are the last 10 WinGo results: ${JSON.stringify(last10)}

Your task:
1. Carefully analyze the sequence.
2. Find trends in color (Red/Green/Violet) and size (Big/Small).
3. Predict the next result logically — not random.

Output ONLY this JSON (no text outside):
{"color":"Red","size":"Big"}
`;

    const res = await model.generateContent(prompt);
    const text = res.response.text().trim();
    console.log(`${aiName} →`, text);
    const parsed = JSON.parse(text);
    return parsed;
  } catch (e) {
    console.log(`${aiName} fallback`, e.message);
    return {
      color: ["Red", "Green", "Violet"][Math.floor(Math.random() * 3)],
      size: Math.random() > 0.5 ? "Big" : "Small"
    };
  }
}

// ---------- Endpoint ----------
app.get("/api/predict-all", async (req, res) => {
  const history = await fetchLast10Results();
  const predictions = await Promise.all(
    AIs.map(a => askGemini(a.key, history, a.name))
  );
  predictions.forEach((p, i) => (AIs[i].last = p));
  res.json({ success: true, history, AIs });
});

app.listen(PORT, () => console.log(`🚀 Server running on ${PORT}`));