from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_MODEL = "ProsusAI/finbert"


@dataclass
class SentimentResult:
    label: str               # "positive" | "neutral" | "negative"
    score: float             # confiance du label
    probs: Dict[str, float]  # proba par label


class FinBertSentiment:
    """
    Wrapper simple autour de FinBERT (ProsusAI/finbert).
    - Input: texte(s)
    - Output: label + score + probas
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # mapping id -> label (souvent: 0 negative, 1 neutral, 2 positive)
        self.id2label = self.model.config.id2label
        # normalise en minuscules pour uniformiser
        self.id2label = {k: v.lower() for k, v in self.id2label.items()}
        self.labels = [self.id2label[i] for i in sorted(self.id2label.keys())]

    @torch.inference_mode()
    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        max_length: int = 256,
    ) -> List[SentimentResult]:
        if isinstance(texts, str):
            texts = [texts]

        results: List[SentimentResult] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()

            for row in probs:
                row_list = row.tolist()
                probs_dict = {self.labels[j]: float(row_list[j]) for j in range(len(self.labels))}
                best_idx = int(torch.argmax(row).item())
                label = self.labels[best_idx]
                score = float(row[best_idx].item())
                results.append(SentimentResult(label=label, score=score, probs=probs_dict))

        return results

    def predict_df(self, texts: List[str], **kwargs: Any) -> pd.DataFrame:
        preds = self.predict(texts, **kwargs)
        rows = []
        for t, p in zip(texts, preds):
            rows.append({
                "text": t,
                "label": p.label,
                "score": p.score,
                **{f"p_{k}": v for k, v in p.probs.items()},
            })
        return pd.DataFrame(rows)