"""Traduction de texte (pour la traduction vocale : parler français, sortir en anglais avec sa voix)."""

from __future__ import annotations

from .base import Engine, EngineError, split_text

# Codes de langue NLLB-200 (FLORES-200)
NLLB_CODES = {
    "fr": "fra_Latn", "en": "eng_Latn", "es": "spa_Latn", "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "nl": "nld_Latn", "pl": "pol_Latn", "ru": "rus_Cyrl", "ja": "jpn_Jpan", "zh": "zho_Hans", "zh-cn": "zho_Hans",
    "ko": "kor_Hang", "ar": "arb_Arab", "tr": "tur_Latn", "cs": "ces_Latn", "hu": "hun_Latn", "hi": "hin_Deva",
    "sv": "swe_Latn", "da": "dan_Latn", "fi": "fin_Latn", "el": "ell_Grek", "he": "heb_Hebr", "no": "nob_Latn",
    "ms": "zsm_Latn", "sw": "swh_Latn",
}


class _Seq2Seq(Engine):
    max_chars = 400

    def _generate(self, text: str, **gen_kwargs) -> str:
        import torch

        out = []
        for piece in split_text(text, self.max_chars):
            batch = self.tok([piece], return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                ids = self.model.generate(**batch, max_new_tokens=512, num_beams=4, **gen_kwargs)
            out.append(self.tok.decode(ids[0], skip_special_tokens=True).strip())
        return " ".join(o for o in out if o)


class MarianEngine(_Seq2Seq):
    """Opus-MT (Helsinki-NLP) : un modèle léger par paire de langues."""

    def load(self) -> None:
        from transformers import MarianMTModel, MarianTokenizer

        self.tok = MarianTokenizer.from_pretrained(str(self.model_dir))
        self.model = MarianMTModel.from_pretrained(str(self.model_dir)).to(self.device).eval()

    def translate(self, text: str, src: str, tgt: str) -> str:
        pair = tuple(self.spec.languages[:2])
        if (src.split("-")[0], tgt.split("-")[0]) != pair:
            raise EngineError(f"{self.spec.name} ne traduit que {pair[0]} → {pair[1]}.")
        return self._generate(text)


class NLLBEngine(_Seq2Seq):
    """NLLB-200 (Meta) : un seul modèle pour 200 langues."""

    def load(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir)).to(self.device).eval()

    def translate(self, text: str, src: str, tgt: str) -> str:
        try:
            s, t = NLLB_CODES[src], NLLB_CODES[tgt]
        except KeyError as exc:
            raise EngineError(f"Langue non prise en charge par NLLB : {exc}") from exc
        self.tok.src_lang = s
        return self._generate(text, forced_bos_token_id=self.tok.convert_tokens_to_ids(t))
