# Hinglish Medical Speech Recognition — Fine-tuned Whisper ASR

Fine-tuning OpenAI's Whisper model to transcribe Hindi-English code-mixed
("Hinglish") clinical speech — the way Indian doctors and patients actually talk.

---

## The Problem

In Indian hospitals and clinics, conversations naturally mix Hindi and English
in the same sentence:

> "उसकी BP बहुत high है, Metformin 500mg दे रहा हूँ"

General-purpose ASR systems (vanilla Whisper, Google Speech-to-Text, etc.) are
trained mostly on clean, monolingual speech. They fail badly on:
- Code-switching mid-sentence (Hindi grammar + English medical terms)
- Domain-specific vocabulary (drug names, dosages, lab values)
- Noisy real-world clinical environments

There is no publicly available Hinglish medical speech dataset, and no existing
medical ASR product (Dragon Medical, Amazon Transcribe Medical, etc.) supports
Hindi or code-mixed speech at all.

## What This Project Does

A complete pipeline that takes vanilla Whisper from unusable on this domain to
meaningfully better, using only free-tier Google Colab:

1. **Corpus creation** — 460+ Hinglish clinical sentences across 10 medical
   sub-specialties (cardiology, endocrinology, psychiatry, nephrology, etc.),
   generated with LLM assistance and covering medicine names, dosages, symptoms,
   diagnoses, and lab values.
2. **Speech synthesis** — every sentence converted to audio using gTTS
   (Google Text-to-Speech), Hindi language mode.
3. **Audio augmentation** — Gaussian noise, time-stretching, and pitch-shifting
   applied to simulate real clinical acoustic conditions (background noise,
   speaking-rate variation, speaker variability).
4. **Baseline evaluation** — vanilla Whisper-small transcribes all audio,
   Word Error Rate (WER) computed as the starting point.
5. **Fine-tuning** — Whisper-small fine-tuned on the corpus using HuggingFace's
   `Seq2SeqTrainer`.
6. **Evaluation** — fine-tuned model re-evaluated on the same test set, WER
   compared against baseline, with qualitative error analysis on medicine
   names and Hindi vocabulary.

## Results

| Condition         | Vanilla Whisper (baseline) | Fine-tuned Whisper | Improvement |
|-------------------|----------------------------|---------------------|-------------|
| Overall WER       | 64.2%                      | 8.9%                 | +55.3%      |
| Clean audio WER   | 63.7%                      | 12.1%                | +51.6%      |
| Noisy audio WER   | 64.7%                      | 6.0%                 | +58.7%      |

The vanilla model frequently mangled medicine names ("Metformin" →
"met form in") and substituted Hindi medical words with phonetically similar
English ones. Fine-tuning measurably reduced both error types.

## Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` (`Seq2SeqTrainer`,
`WhisperProcessor`) · `OpenAI Whisper` · `gTTS` · `audiomentations` ·
`jiwer` · `pandas` · `matplotlib`

## Methodology Reference

This project's pipeline structure (synthetic corpus → TTS audio → augmentation
→ Whisper fine-tuning → WER evaluation) is based on the methodology described
in **United-MedASR** (Banerjee et al., 2024, arXiv:2412.00055), adapted here
to the Hinglish code-mixed domain at a much smaller, reproducible scale
suitable for free-tier hardware.

---

## Honest Limitations

This is a proof-of-concept, not a production system. Being upfront about the
gaps — and being able to explain them — is part of doing this properly:

- **Small dataset.** ~460 sentences / ~0.5 hours of audio, versus thousands of
  hours in production medical ASR systems. Absolute WER numbers aren't
  comparable to commercial tools — only the *relative* improvement from
  fine-tuning is meaningful here.
- **Single synthetic voice.** All training audio comes from one gTTS voice.
  The model has never heard a real human doctor speak, so generalization to
  real clinical audio is untested.
- **Train/test split is at the file level, not the sentence level.** Clean and
  noisy versions of the same sentence can end up split across train and test,
  which may let the model partially memorize sentences rather than purely
  generalize. The *direction* of improvement (fine-tuning helps) is robust;
  the *exact* WER numbers should be read with this caveat in mind.
- **No tokenizer expansion.** Default Whisper tokenizer is used as-is;
  medical terms aren't given dedicated tokens, which likely fragments words
  like "Metformin" inefficiently.
- **No semantic correction layer.** United-MedASR adds a BART-based
  post-correction stage to fix medical terminology after transcription —
  not implemented here. Listed as natural future work.
- **Synthetic-only evaluation.** Test set is the same kind of synthetic audio
  as training (same TTS engine, same noise profile) — not real-world speech.

## Future Work

- Fix the train/test split to be sentence-level, eliminating leakage risk
- Add a BART-based semantic correction stage post-ASR
- Expand the tokenizer with medical-Hinglish vocabulary
- Collect or synthesize speech from multiple voices/speakers
- Validate against real (anonymized, consented) doctor-patient audio
- Scale the corpus using a larger, more systematically generated sentence set

---

## Repository Structure

```
├── notebook.ipynb          # Full pipeline, runs end-to-end on Colab T4 GPU
│                            # (outputs — WER tables, charts — saved inline in the notebook)
└── README.md
```

## How to Run

1. Open `notebook.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells in order — corpus generation through final evaluation
4. Expect ~30-45 minutes total (most of it is the fine-tuning step)

---

*This project was built as coursework for Speech Processing and Synthesis
(UCS749), and as an independent exploration of domain-adapted ASR for
underserved code-mixed languages.*
