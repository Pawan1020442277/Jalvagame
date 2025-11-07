import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI } from "@google/genai";

dotenv.config();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: "1mb" }));

const KEYS = [
  process.env.GOOGLE_KEY_1, process.env.GOOGLE_KEY_2, process.env.GOOGLE_KEY_3,
  process.env.GOOGLE_KEY_4, process.env.GOOGLE_KEY_5, process.env.GOOGLE_KEY_6,
  process.env.GOOGLE_KEY_7, process.env.GOOGLE_KEY_8, process.env.GOOGLE_KEY_9,
  process.env.GOOGLE_KEY_10
];

const AI_STATS = KEYS.map((k,i)=>({
  id:i+1, keyPresent:!!k, name:`AI-${i+1}`, wins:0, losses:0, history:[], lastPrediction:null
}));

async function askModelForDigit(apiKey,lastResults){
  if(!apiKey) return Math.floor(Math.random()*10);
  try{
    const ai=new GoogleGenAI({apiKey});
    const prompt=`Given last results: ${JSON.stringify(lastResults)}, predict next digit (0-9). Respond {"prediction":<digit>}`;
    const res=await ai.models.generateContent({model:"gemini-2.5-flash",contents:prompt});
    const text=res?.text?.trim()||"";
    try{
      const parsed=JSON.parse(text);
      if(typeof parsed.prediction!=="undefined") return Number(parsed.prediction);
    }catch(e){}
    const m=text.match(/\d/);
    if(m) return Number(m[0]);
    return Math.floor(Math.random()*10);
  }catch(e){
    console.log("AI error",e.message);return Math.floor(Math.random()*10);
  }
}

app.post("/api/predict-all",async(req,res)=>{
  const lastResults=Array.isArray(req.body.lastResults)?req.body.lastResults.slice(-10):[];
  const out=await Promise.all(KEYS.map((k,i)=>askModelForDigit(k,lastResults).then(p=>({i,p}))));
  out.forEach(o=>AI_STATS[o.i].lastPrediction=o.p);
  res.json({success:true,predictions:AI_STATS.map(s=>({...s}))});
});

app.post("/api/report-actual",(req,res)=>{
  const actual=Number(req.body.actual);
  if(isNaN(actual)) return res.json({success:false});
  AI_STATS.forEach(s=>{
    const win=s.lastPrediction===actual; if(win)s.wins++;else s.losses++;
    s.history.push(win?1:0); if(s.history.length>10)s.history.shift();
  });
  const ranked=AI_STATS.map(s=>({
    id:s.id,name:s.name,wins:s.wins,losses:s.losses,
    hitRate:(s.wins/(s.wins+s.losses)||0),history:s.history
  })).sort((a,b)=>b.hitRate-a.hitRate);
  res.json({success:true,ranked});
});

app.use(express.static(path.join(__dirname,"public")));
app.listen(process.env.PORT||8787,()=>console.log("Server running"));
