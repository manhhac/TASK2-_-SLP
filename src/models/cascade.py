import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from .mt_model import build_mt_model, translate_vi_en


class CascadeASRMT:
    """Cascade: Whisper ASR (VI) → MarianMT (VI→EN)."""

    def __init__(
        self,
        asr_dir,
        mt_name="Helsinki-NLP/opus-mt-vi-en",
        device="cuda"
    ):
        """
        Khởi tạo cascade model.

        Args:
            asr_dir: Đường dẫn đến fine-tuned Whisper model
            mt_name: Tên MarianMT model
            device: Device (cuda/cpu)
        """
        self.device = device
        self.asr_proc = WhisperProcessor.from_pretrained(
            asr_dir, language="vi", task="transcribe"
        )
        self.asr = WhisperForConditionalGeneration.from_pretrained(asr_dir).to(device).eval()
        self.mt, self.mt_tok = build_mt_model(mt_name)
        self.mt = self.mt.to(device).eval()

    @torch.no_grad()
    def transcribe(self, wavs, batch_size=8, num_beams=1):
        """
        Transcribe audio thành text tiếng Việt.

        Args:
            wavs: List[np.ndarray float32 16k]
            batch_size: Batch size
            num_beams: Beam size (1 = greedy)

        Returns:
            List[str] text tiếng Việt
        """
        outs = []
        for i in range(0, len(wavs), batch_size):
            b = wavs[i:i+batch_size]
            feats = self.asr_proc.feature_extractor(
                b, sampling_rate=16000, return_tensors="pt", padding=True
            ).input_features.to(self.device)
            ids = self.asr.generate(
                feats,
                num_beams=num_beams,
                language="vi",
                task="transcribe"
            )
            outs.extend(self.asr_proc.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return outs

    def translate(self, texts, num_beams=4):
        """
        Dịch text tiếng Việt sang tiếng Anh.

        Args:
            texts: List[str] text tiếng Việt
            num_beams: Beam size

        Returns:
            List[str] text tiếng Anh
        """
        return translate_vi_en(
            texts,
            self.mt,
            self.mt_tok,
            device=self.device,
            num_beams=num_beams
        )

    def forward(self, wavs, asr_beams=1, mt_beams=4):
        """
        Pipeline hoàn chỉnh: audio → VI text → EN text.

        Args:
            wavs: List[np.ndarray] audio
            asr_beams: Beam size cho ASR
            mt_beams: Beam size cho MT

        Returns:
            Tuple[List[str], List[str]]: (hyp_vi, hyp_en)
        """
        hyp_vi = self.transcribe(wavs, num_beams=asr_beams)
        hyp_en = self.translate(hyp_vi, num_beams=mt_beams)
        return hyp_vi, hyp_en
