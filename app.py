from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from pythainlp.tokenize import word_tokenize
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# ── Load WangchanBERTa ──────────────────────────────────────────────────────
print("Loading WangchanBERTa model...")
tokenizer = AutoTokenizer.from_pretrained("Pongsathorn/wangchanberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("Pongsathorn/wangchanberta-base-sentiment")
model.eval()
print("Model loaded!")

id2label = {0: "pos", 1: "neu", 2: "neg"}

sentiment_texts = {
    "pos": "😊 เชิงบวก",
    "neu": "😐 เป็นกลาง",
    "neg": "😞 เชิงลบ",
}

# ── Groq Client ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ ไม่พบ GROQ_API_KEY กรุณาสร้างไฟล์ .env และใส่ GROQ_API_KEY=your_key")
groq_client = Groq(api_key=GROQ_API_KEY)

def analyze_sentiment(text: str):
    """Analyze Thai text sentiment using WangchanBERTa."""
    segmented = word_tokenize(text, engine="longest")
    preprocessed = " ".join(segmented)

    inputs = tokenizer(
        preprocessed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    predicted_label = id2label[predicted_class]
    confidence = float(np.max(probs.numpy()))

    return predicted_label, confidence


def rewrite_with_groq(original_text: str) -> str:
    """Use Groq LLM to rewrite negative Thai text into constructive message."""
    prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านการสื่อสารเชิงบวก 
    
ข้อความต่อไปนี้มีแนวโน้มเชิงลบหรืออาจก่อให้เกิดความขัดแย้ง:
"{original_text}"

กรุณา rewrite ข้อความนี้ให้:
1. สื่อสารความต้องการหรือความรู้สึกได้ครบถ้วน
2. ใช้ภาษาที่สร้างสรรค์และไม่ก้าวร้าว
3. เปิดโอกาสให้เกิดการพูดคุยที่ดี
4. ยังคงความหมายหลักของข้อความเดิม

ตอบเฉพาะข้อความที่ rewrite แล้วเท่านั้น ไม่ต้องมีคำอธิบายเพิ่มเติม"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "กรุณาใส่ข้อความ"}), 400

    label, confidence = analyze_sentiment(text)

    result = {
        "sentiment": label,
        "sentiment_text": sentiment_texts[label],
        "confidence": round(confidence * 100, 1),
        "rewritten": None,
    }

    if label == "neg":
        try:
            result["rewritten"] = rewrite_with_groq(text)
        except Exception as e:
            result["rewritten"] = f"ไม่สามารถ rewrite ได้: {str(e)}"

    return jsonify(result)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
