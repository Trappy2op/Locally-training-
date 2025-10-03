"""
pdf_rag_mistral_local_train.py
Tkinter GUI:
 - Upload PDF(s)
 - RAG retrieval
 - Local LoRA fine-tuning
 - Chat with base or fine-tuned model
"""

import os, re, json, threading, traceback
from pathlib import Path
from typing import List

import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from tkinter import *
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Progressbar, Checkbutton

# ---- HF / Transformers for local fine-tuning ----
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

# ---------------- CONFIG ----------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = "faiss_index.bin"
TEXTS_PATH = "rag_texts.json"
LORA_PATH = "lora_adapter"

EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

# ---------------- PDF / RAG Utilities ----------------
HEADING_PATTERNS = [r"^(Chapter\s+\d+[:.\-\s].*)$", r"^(Section\s+\d+[:.\-\s].*)$", r"^(\d+\.\s+[A-Za-z].*)$"]
HEADING_RE = re.compile("|".join(HEADING_PATTERNS), flags=re.IGNORECASE | re.MULTILINE)

def extract_pdf_chunks(path: str, min_len: int = 120) -> List[dict]:
    doc = fitz.open(path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) >= min_len]
        merged = []
        temp = ""
        for p in paras:
            if len(p) < min_len:
                temp += (" " + p)
            else:
                combined = (temp + " " + p).strip() if temp else p
                merged.append(combined)
                temp = ""
        if temp: merged.append(temp.strip())
        for i, para in enumerate(merged):
            chunks.append({"text": para, "meta": f"{os.path.basename(path)} | page {page_num} | chunk {i+1}"})
    doc.close()
    return chunks

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    return faiss.IndexFlatIP(dim)

def add_to_index(chunks: List[dict], save: bool = True):
    if not chunks: return 0
    texts = [c["text"] for c in chunks]
    emb = EMBED_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    emb = emb / norms

    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        index = faiss.read_index(INDEX_PATH)
        all_data = json.load(open(TEXTS_PATH, "r", encoding="utf-8"))
    else:
        index = build_faiss_index(emb)
        all_data = []

    index.add(emb.astype(np.float32))
    for t, c in zip(texts, chunks):
        all_data.append({"text": t, "meta": c["meta"]})
    if save:
        faiss.write_index(index, INDEX_PATH)
        with open(TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
    return len(chunks)

def load_index_texts():
    if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
        return faiss.read_index(INDEX_PATH), json.load(open(TEXTS_PATH, "r", encoding="utf-8"))
    return None, []

def rag_retrieve(query: str, k: int = 3) -> List[dict]:
    idx, all_data = load_index_texts()
    if not idx or not all_data: return []
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    distances, indices = idx.search(q_emb.astype(np.float32), k)
    results = []
    for dist, ind in zip(distances[0], indices[0]):
        if 0 <= ind < len(all_data):
            item = all_data[ind]
            results.append({"text": item["text"], "meta": item["meta"], "score": float(dist)})
    return results

# ---------------- Local Fine-Tuning ----------------
def fine_tune_lora(chunks: List[dict], output_dir=LORA_PATH, epochs=1):
    """
    Fine-tune Mistral locally on PDF chunks using LoRA.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for local fine-tuning.")

    # prepare dataset
    dataset = [{"text": f"{c['text']}"} for c in chunks]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    texts = [d["text"] for d in dataset]
    encodings = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    
    # base model
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto")
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"],
                             lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    class PDFDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
        def __len__(self):
            return self.input_ids.shape[0]
        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.input_ids[idx]}

    pdf_dataset = PDFDataset(encodings)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        fp16=True,
        save_strategy="epoch",
        logging_steps=5,
        report_to=[]
    )
    trainer = Trainer(model=model, args=args, train_dataset=pdf_dataset)
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

def load_finetuned_adapter(adapter_path=LORA_PATH):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

# ---------------- Tkinter GUI ----------------
class RAGApp:
    def __init__(self, root):
        self.root = root
        root.title("PDF-RAG + Mistral LoRA (Local Training)")
        root.geometry("1000x760")

        Label(root, text="Query:", font=("Arial", 12)).pack(pady=6)
        self.question_entry = Entry(root, width=120)
        self.question_entry.pack(pady=4)

        self.chat_display = scrolledtext.ScrolledText(root, width=140, height=30, font=("Courier", 10))
        self.chat_display.pack(pady=8)

        self.status_var = StringVar(value=f"Base Model: {BASE_MODEL}")
        Label(root, textvariable=self.status_var, fg="darkgreen").pack()
        self.progress = Progressbar(root, mode='indeterminate', length=400)

        self.use_rag_var = IntVar(value=1)
        Checkbutton(root, text="Use local RAG retrieval", variable=self.use_rag_var).pack()

        frame = Frame(root)
        frame.pack(pady=6)
        Button(frame, text="Ask", command=self.ask, width=18).grid(row=0, column=0, padx=6)
        Button(frame, text="Load PDF", command=self.load_pdf, width=18).grid(row=0, column=1, padx=6)
        Button(frame, text="Train on PDF (LoRA)", command=self.train_pdf, width=18).grid(row=0, column=2, padx=6)
        Button(frame, text="Export Chat", command=self.export_chat, width=18).grid(row=0, column=3, padx=6)

        self.chat_log = []
        self.pdf_chunks = []
        self.lora_model = None
        self.lora_tokenizer = None

        self.question_entry.bind("<Return>", lambda e: self.ask())

    def append_chat(self, speaker, text):
        self.chat_display.insert(END, f"{speaker}: {text}\n\n")
        self.chat_display.yview(END)

    def ask(self):
        q = self.question_entry.get().strip()
        if not q: return
        self.append_chat("You", q)
        self.question_entry.delete(0, END)

        def task():
            try:
                self.status_var.set("Retrieving context..." if self.use_rag_var.get() else "Ready.")
                context = "\n\n".join([c["text"] for c in rag_retrieve(q)]) if self.use_rag_var.get() else ""
                self.status_var.set("Generating answer...")
                if self.lora_model:
                    inputs = self.lora_tokenizer(q + "\n" + context, return_tensors="pt").to("cuda")
                    outputs = self.lora_model.generate(**inputs, max_new_tokens=350)
                    ans = self.lora_tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    ans = "Local Mistral inference not implemented here; use HF API if needed."
                self.append_chat("SafetyHead", ans)
                self.chat_log.append((q, ans))
                self.status_var.set("Ready.")
            except Exception as e:
                self.status_var.set(f"Error: {e}")
                self.append_chat("Error", traceback.format_exc())
        threading.Thread(target=task).start()

    def load_pdf(self):
        file = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file: return
        def task():
            try:
                self.status_var.set(f"Extracting {os.path.basename(file)}...")
                self.progress.pack(pady=4)
                self.progress.start()
                chunks = extract_pdf_chunks(file)
                added = add_to_index(chunks)
                self.pdf_chunks.extend(chunks)
                self.status_var.set(f"PDF added: {len(chunks)} chunks (total {len(self.pdf_chunks)} chunks).")
                messagebox.showinfo("PDF Loaded", f"{os.path.basename(file)} added ({len(chunks)} chunks).")
            finally:
                self.progress.stop()
                self.progress.pack_forget()
        threading.Thread(target=task).start()

    def train_pdf(self):
        if not self.pdf_chunks:
            messagebox.showwarning("No PDF", "Please load PDF(s) first.")
            return
        def task():
            try:
                self.status_var.set("Fine-tuning LoRA on PDF chunks...")
                self.progress.pack(pady=4)
                self.progress.start()
                output_dir = fine_tune_lora(self.pdf_chunks)
                self.lora_model, self.lora_tokenizer = load_finetuned_adapter(output_dir)
                self.status_var.set(f"LoRA training complete. Adapter loaded from {output_dir}.")
                messagebox.showinfo("Training Complete", "Local LoRA fine-tuning finished.")
            finally:
                self.progress.stop()
                self.progress.pack_forget()
        threading.Thread(target=task).start()

    def export_chat(self):
        if not self.chat_log:
            messagebox.showinfo("Empty", "No chat to export.")
            return
        file = filedialog.asksaveasfilename(defaultextension=".txt")
        if file:
            with open(file, "w", encoding="utf-8") as f:
                for q,a in self.chat_log:
                    f.write(f"Q: {q}\nA: {a}\n\n")
            messagebox.showinfo("Saved", "Chat exported.")

if __name__ == "__main__":
    root = Tk()
    app = RAGApp(root)
    root.mainloop()
