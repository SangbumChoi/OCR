"""A newly trained byte-level tokenizer for the near-random document VLM student."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
}


def iter_udd_text(dataset: Any) -> Iterator[str]:
    """Yield all textual supervision without decoding the UDD image column."""

    from .data import _metadata_view, _parse_elements

    metadata = _metadata_view(dataset)
    yield "User:"
    yield "Assistant:"
    yield "Return only the normalized bounding box as [x1, y1, x2, y2]."
    for index in range(len(metadata)):
        row = metadata[index]
        for instruction in row.get("instructions") or []:
            text = str(instruction).strip()
            if text:
                yield text
        for variants in row.get("answers") or []:
            for answer in variants or []:
                text = str(answer).strip()
                if text:
                    yield text
        for answer in row.get("teacher_answers") or []:
            text = str(answer).strip()
            if text:
                yield text
        for key in ("full_text", "table_html"):
            text = str(row.get(key) or "").strip()
            if text:
                yield text
        for element in _parse_elements(row):
            for key in ("key", "value"):
                text = str(element.get(key) or "").strip()
                if text:
                    yield text


class DocumentTokenizer:
    """Small adapter exposing the tokenizer contract consumed by :class:`StudentCollator`."""

    def __init__(self, backend: Any, metadata: dict[str, Any] | None = None):
        self.backend = backend
        self.metadata = metadata or {}
        self.pad_token_id = self._required_id("pad_token")
        self.bos_token_id = self._required_id("bos_token")
        self.eos_token_id = self._required_id("eos_token")
        self.unk_token_id = self._required_id("unk_token")

    def _required_id(self, name: str) -> int:
        token = self.metadata.get("special_tokens", SPECIAL_TOKENS).get(
            name,
            SPECIAL_TOKENS[name],
        )
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required special token {token!r}")
        return int(token_id)

    @property
    def vocab_size(self) -> int:
        return int(self.backend.get_vocab_size())

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = list(self.backend.encode(str(text), add_special_tokens=False).ids)
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.backend.decode(
            [int(token) for token in ids],
            skip_special_tokens=skip_special_tokens,
        )

    @property
    def fingerprint(self) -> str:
        """Identify the complete token-to-ID contract used by online distillation."""

        payload = self.backend.to_str().encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(output / "tokenizer.json"))
        metadata = {
            **self.metadata,
            "format": "docvlm-byte-level-bpe-v1",
            "fingerprint": self.fingerprint,
            "vocab_size": self.vocab_size,
            "special_tokens": self.metadata.get("special_tokens", SPECIAL_TOKENS),
            "normalization": "NFC",
        }
        (output / "tokenizer_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DocumentTokenizer":
        from tokenizers import Tokenizer

        root = Path(path)
        tokenizer_path = root / "tokenizer.json" if root.is_dir() else root
        config_path = tokenizer_path.with_name("tokenizer_config.json")
        metadata = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        return cls(Tokenizer.from_file(str(tokenizer_path)), metadata)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 64_000,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> "DocumentTokenizer":
        """Train NFC-preserving byte-level BPE, retaining a fallback path for every UTF-8 byte."""

        if vocab_size < 260:
            raise ValueError("byte-level tokenizer vocab_size must be at least 260")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

        backend = Tokenizer(
            models.BPE(
                unk_token=SPECIAL_TOKENS["unk_token"],
                byte_fallback=True,
            )
        )
        backend.normalizer = normalizers.NFC()
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=list(SPECIAL_TOKENS.values()),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        backend.train_from_iterator(texts, trainer=trainer)
        return cls(
            backend,
            {
                "requested_vocab_size": vocab_size,
                "min_frequency": min_frequency,
                "special_tokens": SPECIAL_TOKENS,
            },
        )
