"""Processus d'entraînement XTTS (lancé par voiceclone.training, ne pas utiliser directement).

Reprend la recette officielle de fine-tuning du GPT de XTTS v2 (coqui-tts), avec les fichiers du
modèle déjà téléchargés par VoiceClone. Écrit <output>/result.json avec le chemin du modèle obtenu
(poids allégés : sans l'état de l'optimiseur).

    python -m voiceclone.xtts_train '{"dataset": "...", "output": "...", "base_dir": "...",
                                      "language": "fr", "epochs": 10, "batch_size": 2, "grad_accum": 4}'
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path


def main(params: dict) -> None:
    import torch

    from voiceclone.engines.xtts import patch_coqui

    patch_coqui()  # avant les imports de TTS : compatibilité transformers 5 et lecture audio sans torchcodec
    from trainer import Trainer, TrainerArgs

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    base = Path(params["base_dir"])
    out = Path(params["output"])
    out.mkdir(parents=True, exist_ok=True)
    dataset = Path(params["dataset"])
    dataset_cfg = BaseDatasetConfig(formatter="coqui", dataset_name="voiceclone", path=str(dataset),
                                    meta_file_train=str(dataset / "metadata_train.csv"),
                                    meta_file_val=str(dataset / "metadata_eval.csv"), language=params["language"])
    model_args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150, debug_loading_failures=False,
        max_wav_length=255995, max_text_length=200, mel_norm_file=str(base / "mel_stats.pth"),
        dvae_checkpoint=str(base / "dvae.pth"), xtts_checkpoint=str(base / "model.pth"),
        tokenizer_file=str(base / "vocab.json"), gpt_num_audio_tokens=1026, gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025, gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True)
    cfg = GPTTrainerConfig(
        epochs=int(params["epochs"]), output_path=str(out), model_args=model_args, run_name="voiceclone_xtts_ft",
        project_name="voiceclone", dashboard_logger="tensorboard", logger_uri=None,
        audio=XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000),
        batch_size=int(params["batch_size"]), batch_group_size=48, eval_batch_size=int(params["batch_size"]),
        num_loader_workers=2, eval_split_max_size=256, print_step=50, plot_step=100, log_model_step=100,
        save_step=100000, save_n_checkpoints=1, save_checkpoints=True, print_eval=False, optimizer="AdamW",
        optimizer_wd_only_on_weights=True, optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06, lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[])
    model = GPTTrainer.init_from_config(cfg)
    train_samples, eval_samples = load_tts_samples([dataset_cfg], eval_split=True,
                                                   eval_split_max_size=cfg.eval_split_max_size,
                                                   eval_split_size=cfg.eval_split_size)
    trainer = Trainer(TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False,
                                  grad_accum_steps=int(params["grad_accum"])),
                      cfg, output_path=str(out), model=model, train_samples=train_samples, eval_samples=eval_samples)
    trainer.fit()
    run_dir = Path(trainer.output_path)
    del model, trainer
    gc.collect()

    candidates = sorted(run_dir.glob("best_model*.pth")) or sorted(run_dir.glob("checkpoint_*.pth"),
                                                                   key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"Aucun point de contrôle produit dans {run_dir}")
    ckpt = torch.load(candidates[-1], map_location="cpu", weights_only=False)
    ckpt.pop("optimizer", None)  # ~ 3 Go de moins ; inutile pour l'inférence
    ckpt["model"] = {k: v for k, v in ckpt["model"].items() if "dvae" not in k}
    final = out / "model.pth"
    torch.save(ckpt, final)
    (out / "result.json").write_text(json.dumps({"model": str(final), "checkpoint": str(candidates[-1])}),
                                     encoding="utf-8")
    print("Entraînement terminé :", final, flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
