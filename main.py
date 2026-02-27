from __future__ import annotations
import json
import os
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Literal, Any
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))

VOCAB_PATH = MODELS_DIR / "vocab.json"
TFIDF_PARAMS_PATH = MODELS_DIR / "tfidf_params.json" 
BEST_RUN_NAME = os.getenv("BEST_RUN_NAME", "").strip()
_word_re = re.compile(r"\b\w+\b", re.UNICODE)
_br_re = re.compile(r"<br\s*/?>", re.IGNORECASE)

def preprocess_text(text: str) -> List[str]:
    text = str(text).lower()
    text = _br_re.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    try:
        from nltk.tokenize import word_tokenize  
        return word_tokenize(text)
    except Exception:
        return _word_re.findall(text)

class LSTM_Emb(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = embedding_layer
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)            
        _, (h_n, _) = self.lstm(emb)      

        if self.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)  
        else:
            h = h_n[-1]                              

        h = self.dropout(h)
        return self.fc(h).squeeze(1)                


class LSTM_TFIDFSeq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        pad_idx: int = 0,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)           
        emb = emb * w.unsqueeze(-1)       
        _, (h_n, _) = self.lstm(emb)

        if self.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h = h_n[-1]

        h = self.dropout(h)
        return self.fc(h).squeeze(1)    


class Assets:
    def __init__(self):
        if not VOCAB_PATH.exists():
            raise RuntimeError(f"Missing {VOCAB_PATH}. Put your notebook's models/ folder next to main.py.")

        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.itos: List[str] = vocab["itos"]
        self.stoi: Dict[str, int] = {k: int(v) for k, v in vocab["stoi"].items()}
        self.pad_idx: int = int(vocab.get("pad_idx", 0))
        self.unk_idx: int = int(vocab.get("unk_idx", 1))
        self.max_len: int = int(vocab.get("max_len", 256))
        self.tfidf_vocab: Optional[Dict[str, int]] = None
        self.tfidf_idf: Optional[List[float]] = None

        if TFIDF_PARAMS_PATH.exists():
            with open(TFIDF_PARAMS_PATH, "r", encoding="utf-8") as f:
                tfidf = json.load(f)
            self.tfidf_vocab = {k: int(v) for k, v in tfidf["vocab"].items()}
            self.tfidf_idf = [float(x) for x in tfidf["idf"]]


def list_model_cfgs() -> Dict[str, Dict[str, Any]]:
    cfgs: Dict[str, Dict[str, Any]] = {}
    if not MODELS_DIR.exists():
        return cfgs

    for p in MODELS_DIR.glob("*.json"):
        if p.name in ("vocab.json", "tfidf_params.json"):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "run_name" in cfg and "embedding" in cfg:
                cfgs[p.stem] = cfg
        except Exception:
            continue
    return cfgs

class Predictor:
    def __init__(self, device: Optional[str] = None):
        self.assets = Assets()
        all_cfgs = list_model_cfgs()
        if not all_cfgs:
            raise RuntimeError(f"No model configs found in {MODELS_DIR}. Make sure *.pt and *.json exist.")
        if BEST_RUN_NAME:
            if BEST_RUN_NAME not in all_cfgs:
                raise RuntimeError(f"BEST_RUN_NAME='{BEST_RUN_NAME}' not found in models/.")
            self.cfgs = {BEST_RUN_NAME: all_cfgs[BEST_RUN_NAME]}
            self.run_name = BEST_RUN_NAME
        else:
            picked = None
            for rn, cfg in all_cfgs.items():
                if bool(cfg.get("bidirectional", False)):
                    picked = rn
                    break
            self.run_name = picked or next(iter(all_cfgs.keys()))
            self.cfgs = {self.run_name: all_cfgs[self.run_name]}

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: Optional[nn.Module] = None

    def _build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        emb = cfg["embedding"]
        hidden_dim = int(cfg.get("hidden_dim", 128))
        num_layers = int(cfg.get("num_layers", 1))
        dropout = float(cfg.get("dropout", 0.3))
        bidirectional = bool(cfg.get("bidirectional", False))
        embed_dim = int(cfg.get("embed_dim", 100))

        vocab_size = len(self.assets.itos)

        if emb == "tfidf":
            if self.assets.tfidf_vocab is None or self.assets.tfidf_idf is None:
                raise RuntimeError(
                    "This model uses embedding='tfidf' but models/tfidf_params.json is missing.\n"
                    "Run the save cell for tfidf_params.json in the notebook and copy it into models/."
                )
            return LSTM_TFIDFSeq(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                pad_idx=self.assets.pad_idx,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )

        embedding_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=self.assets.pad_idx)
        return LSTM_Emb(
            embedding_layer=embedding_layer,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def load(self) -> nn.Module:
        if self._model is not None:
            return self._model

        cfg = self.cfgs[self.run_name]
        model = self._build_model(cfg)

        pt_path = MODELS_DIR / f"{self.run_name}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing weights file: {pt_path}")

        state = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        self._model = model
        return model

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        seqs = []
        for t in texts:
            tokens = preprocess_text(t)
            seq = [self.assets.stoi.get(tok, self.assets.unk_idx) for tok in tokens]
            if len(seq) > self.assets.max_len:
                seq = seq[: self.assets.max_len]
            else:
                seq = seq + [self.assets.pad_idx] * (self.assets.max_len - len(seq))
            seqs.append(seq)
        return torch.tensor(seqs, dtype=torch.long, device=self.device)

    def _tfidf_weights_batch(self, tokenized: List[List[str]]) -> torch.Tensor:
        assert self.assets.tfidf_vocab is not None and self.assets.tfidf_idf is not None
        ws = []
        for tokens in tokenized:
            counts = Counter(tokens)
            L = max(len(tokens), 1)
            weights: List[float] = []
            for tok in tokens[: self.assets.max_len]:
                j = self.assets.tfidf_vocab.get(tok, None)
                if j is None:
                    weights.append(0.0)
                else:
                    tf = counts[tok] / L
                    weights.append(tf * float(self.assets.tfidf_idf[j]))
            if len(weights) < self.assets.max_len:
                weights += [0.0] * (self.assets.max_len - len(weights))
            ws.append(weights)
        return torch.tensor(ws, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict_one(self, text: str) -> Dict[str, Any]:
        model = self.load()
        cfg = self.cfgs[self.run_name]
        emb = cfg["embedding"]

        tokens = preprocess_text(text)
        x = self._encode_batch([text])

        if emb == "tfidf":
            w = self._tfidf_weights_batch([tokens])  
            logits = model(x, w)  
        else:
            logits = model(x)    

        prob_pos = torch.sigmoid(logits).item()
        prob_neg = 1.0 - prob_pos
        label = 1 if prob_pos >= 0.5 else 0

        return {
            "run_name": self.run_name,
            "prob_pos": float(prob_pos),
            "prob_neg": float(prob_neg),
            "label": int(label),
            "label_text": "pos" if label == 1 else "neg",
        }

    @torch.no_grad()
    def predict_many(self, texts: List[str]) -> Dict[str, Any]:
        model = self.load()
        cfg = self.cfgs[self.run_name]
        emb = cfg["embedding"]

        tokenized = [preprocess_text(t) for t in texts]
        x = self._encode_batch(texts)  

        if emb == "tfidf":
            w = self._tfidf_weights_batch(tokenized)  
            logits = model(x, w) 
        else:
            logits = model(x)     

        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        results = []
        for p in probs:
            label = 1 if p >= 0.5 else 0
            results.append({
                "prob_pos": float(p),
                "prob_neg": float(1.0 - float(p)),
                "label": int(label),
                "label_text": "pos" if label == 1 else "neg",
            })

        return {"run_name": self.run_name, "results": results}

    def info(self) -> Dict[str, Any]:
        cfg = self.cfgs[self.run_name]
        perf = {
            "test_accuracy": cfg.get("test_accuracy"),
            "precision": cfg.get("precision"),
            "recall": cfg.get("recall"),
            "f1": cfg.get("f1"),
            "confusion_matrix": cfg.get("confusion_matrix"),
        }
        meta = {
            "embedding": cfg.get("embedding"),
            "bidirectional": bool(cfg.get("bidirectional", False)),
            "hidden_dim": cfg.get("hidden_dim"),
            "dropout": cfg.get("dropout"),
            "num_layers": cfg.get("num_layers"),
            "embed_dim": cfg.get("embed_dim"),
            "max_len": self.assets.max_len,
            "vocab_size": len(self.assets.itos),
            "device": str(self.device),
        }
        return {"run_name": self.run_name, "metadata": meta, "performance": perf}

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

class PredictResponse(BaseModel):
    run_name: str
    prob_pos: float
    prob_neg: float
    label: int
    label_text: str

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)

class BatchResponse(BaseModel):
    run_name: str
    results: List[Dict[str, Any]]

app = FastAPI(title="IMDB Sentiment")

predictor: Optional[Predictor] = None

@app.on_event("startup")
def _startup():
    global predictor
    predictor = Predictor()

@app.get("/health")
def health():
    assert predictor is not None
    try:
        predictor.load()
        loaded = True
        err = None
    except Exception as e:
        loaded = False
        err = str(e)
    return {
        "ok": loaded,
        "models_dir": str(MODELS_DIR),
        "run_name": predictor.run_name,
        "error": err,
    }

@app.get("/list_models")
def list_models():
    assert predictor is not None
    info = predictor.info()
    return {"count": 1, "models": [info["metadata"] | {"run_name": info["run_name"]}]}

@app.get("/models")
def models_alias():
    return list_models()

@app.get("/model_info")
def model_info():
    assert predictor is not None
    return predictor.info()

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    assert predictor is not None
    try:
        return predictor.predict_one(req.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    assert predictor is not None
    try:
        return predictor.predict_many(req.texts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))