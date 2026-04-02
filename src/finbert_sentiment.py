"""
FinBERT sentiment analyser for GeoQuant AI -- Bloc 2 NLP.

Original author : Arthur (branch news_Arthur).
Integrated into NewsElena branch by Elena.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_MODEL = "ProsusAI/finbert"


@dataclass
class SentimentResult:
    """Container for one FinBERT prediction."""

    label: str               # "positive" | "neutral" | "negative"
    score: float             # confidence of the winning label
    probs: Dict[str, float]  # probability per label


class FinBertSentiment:
    """
    Thin wrapper around FinBERT (ProsusAI/finbert).

    Input  : one or more text strings.
    Output : label + confidence score + full probability distribution.

    The model is downloaded from Hugging Face on first use (~500 MB).
    Subsequent calls reuse the cached weights.

    Args:
        model_name: Hugging Face model identifier (default: ProsusAI/finbert).
        device:     "cuda" / "cpu" (auto-detected when None).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"  [FinBERT] Loading model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Normalise label ids to lowercase  (0=negative, 1=neutral, 2=positive)
        self.id2label: Dict[int, str] = {
            k: v.lower() for k, v in self.model.config.id2label.items()
        }
        self.labels: List[str] = [
            self.id2label[i] for i in sorted(self.id2label.keys())
        ]
        print(f"  [FinBERT] Ready. Labels: {self.labels}")

    @torch.inference_mode()
    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        max_length: int = 256,
    ) -> List[SentimentResult]:
        """
        Run FinBERT on one or more texts.

        Args:
            texts:      Single string or list of strings.
            batch_size: Number of texts processed per forward pass.
            max_length: Maximum token length (longer texts are truncated).

        Returns:
            List of SentimentResult (same length as input).
        """
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
            probs  = torch.softmax(logits, dim=-1).detach().cpu()

            for row in probs:
                row_list  = row.tolist()
                probs_dict = {
                    self.labels[j]: float(row_list[j])
                    for j in range(len(self.labels))
                }
                best_idx = int(torch.argmax(row).item())
                label    = self.labels[best_idx]
                score    = float(row[best_idx].item())
                results.append(
                    SentimentResult(label=label, score=score, probs=probs_dict)
                )

        return results

    def predict_df(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Run FinBERT and return results as a DataFrame.

        Columns: text | label | score | p_positive | p_neutral | p_negative

        Args:
            texts:   List of strings to analyse.
            **kwargs: Forwarded to :meth:`predict`.

        Returns:
            DataFrame with one row per input text.
        """
        preds = self.predict(texts, **kwargs)
        rows = []
        for t, p in zip(texts, preds):
            rows.append(
                {
                    "text": t,
                    "label": p.label,
                    "score": p.score,
                    **{f"p_{k}": v for k, v in p.probs.items()},
                }
            )
        return pd.DataFrame(rows)
