import logging
import os
import time
from typing import Optional, Tuple
import numpy as np
import onnxruntime
import torch
from funasr_local.torch_utils.set_all_random_seed import set_all_random_seed
from extract_paraformer_feature import Speech2Text
from scipy import signal


class Audio2Mouth(object):
    def __init__(self, use_gpu: bool = False, weight_dir: str = "./weights"):
        self.weight_dir = weight_dir
        self.device = "cuda" if use_gpu else "cpu"

        model_path = os.path.join(self.weight_dir, "model_1.onnx")
        provider = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
        self.audio2mouth_model = onnxruntime.InferenceSession(model_path, providers=[provider])
        self.w = np.array([1.0]).astype(np.float32)
        self.sp = np.array([2]).astype(np.int64)
        self.p_list = [str(ii) for ii in range(32)]

        self.init_paraformer()

    def init_paraformer(self):
        """
        init paramformer with funasr to extract audio feature
        """
        model_path = os.path.join(
            self.weight_dir,
            "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )

        # for speech sound
        asr_train_config = os.path.join(model_path, "config.yaml")
        asr_model_file = os.path.join(model_path, "model.pb")
        cmvn_file = os.path.join(model_path, "am.mvn")
        lm_file = os.path.join(model_path, "lm/lm.pb")
        lm_train_config = os.path.join(model_path, "lm/lm.yaml")
        beam_size = 1
        penalty = 0.0
        maxlenratio = 0.0
        minlenratio = 0.0
        ctc_weight = 0.0
        lm_weight = 0.0
        frontend_conf = {
            "fs": 16000,
            "win_length": 400,
            "hop_length": 160,
            "window": "hamming",
            "n_mels": 80,
            "lfr_m": 7,
            "lfr_n": 6,
        }
        token_type: Optional[str] = None
        bpemodel: Optional[str] = None
        dtype: str = "float32"
        ngram_weight: float = 0.9
        nbest: int = 1

        set_all_random_seed(0)

        speech2text_kwargs = dict(
            asr_train_config=asr_train_config,
            asr_model_file=asr_model_file,
            cmvn_file=cmvn_file,
            lm_train_config=lm_train_config,
            lm_file=lm_file,
            token_type=token_type,
            bpemodel=bpemodel,
            device=self.device,
            maxlenratio=maxlenratio,
            minlenratio=minlenratio,
            dtype=dtype,
            beam_size=beam_size,
            ctc_weight=ctc_weight,
            lm_weight=lm_weight,
            ngram_weight=ngram_weight,
            penalty=penalty,
            nbest=nbest,
            frontend_conf=frontend_conf,
        )
        self.speech2text = Speech2Text(**speech2text_kwargs)

    def extract_para_feature(self, audio, frame_cnt):
        s = time.time()

        batch_ = {
            "speech": torch.tensor(np.array([audio], dtype=np.float32)),
            "frame_cnt": frame_cnt,
            "speech_lengths": torch.tensor(np.array([len(audio)])),
        }
        results = self.speech2text(**batch_)
        logging.info("extract paraformer feature in {}ms".format(round((time.time() - s), 3)))
        return results

    def butter_lowpass_filtfilt(self, data, order=4, cutoff=7, fs=30):
        wn = 2 * cutoff / fs
        b, a = signal.butter(order, wn, "lowpass", analog=False)
        output = signal.filtfilt(b, a, data, padtype=None, axis=0)
        return output

    def mouth_smooth(self, param_res):
        for key in param_res[0]:
            val_list = []
            for ii in range(len(param_res)):
                val_list.append(param_res[ii][key])
            val_list = self.butter_lowpass_filtfilt(np.asarray(val_list), order=4, cutoff=10, fs=30)
            for ii in range(len(param_res)):
                param_res[ii][key] = val_list[ii]
        return param_res

    def geneHeadInfo(self, sampleRate, bits, sampleNum):
        import struct

        rHeadInfo = b"\x52\x49\x46\x46"
        fileLength = struct.pack("i", sampleNum + 36)
        rHeadInfo += fileLength
        rHeadInfo += b"\x57\x41\x56\x45\x66\x6d\x74\x20\x10\x00\x00\x00\x01\x00\x01\x00"
        rHeadInfo += struct.pack("i", sampleRate)
        rHeadInfo += struct.pack("i", int(sampleRate * bits / 8))
        rHeadInfo += b"\x02\x00"
        rHeadInfo += struct.pack("H", bits)
        rHeadInfo += b"\x64\x61\x74\x61"
        rHeadInfo += struct.pack("i", sampleNum)
        return rHeadInfo

    @torch.inference_mode()
    def inference(self, subtitles=None, input_audio=None):
        f1 = time.time()

        frame_cnt = int(len(input_audio) / 16000 * 30)
        au_data = self.extract_para_feature(input_audio, frame_cnt)
        ph_data = np.zeros((au_data.shape[0], 2))

        audio_length = ph_data.shape[0] / 30
        logging.info("extract all feature in {}s".format(round(time.time() - f1, 3)))
        logging.info("audio length: {}s".format(round(audio_length, 3)))

        param_res = []
        interval = 1.0
        frag = int(interval * 30 / 5 + 0.5)

        start_time = 0.0
        end_time = start_time + interval
        is_end = False
        while True:
            start = int(start_time * 16000)
            end = start + 16000
            if end_time >= audio_length:
                is_end = True
                end = int(audio_length * 16000)
                start = end - 16000
                start_time = audio_length - interval
                end_time = audio_length
            start_frame = int(start_time * int(30))
            end_frame = start_frame + int(30 * interval)

            input_au = au_data[start_frame:end_frame]
            input_ph = ph_data[start_frame:end_frame]
            input_au = input_au[np.newaxis, :].astype(np.float32)
            input_ph = input_ph[np.newaxis, :].astype(np.float32)

            output, feat = self.audio2mouth_model.run(
                ["output", "feat"],
                {"input_au": input_au, "input_ph": input_ph, "input_sp": self.sp, "w": self.w},
            )

            if start_time == 0.0:
                if is_end is False:
                    for tt in range(int(30 * interval) - frag):
                        param_frame = {}
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0, tt, ii])
                            param_frame[key] = round(value, 3)
                        param_res.append(param_frame)
                else:
                    for tt in range(int(30 * interval)):
                        param_frame = {}
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0, tt, ii])
                            param_frame[key] = round(value, 3)
                        param_res.append(param_frame)
            elif is_end is False:
                for tt in range(frag, int(30 * interval) - frag):
                    frame_id = start_frame + tt
                    if frame_id < len(param_res):
                        scale = min((len(param_res) - frame_id) / frag, 1.0)
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0, tt, ii])
                            value = (1 - scale) * value + scale * param_res[frame_id][key]
                            param_res[frame_id][key] = round(value, 3)
                    else:
                        param_frame = {}
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0, tt, ii])
                            param_frame[key] = round(value, 3)
                        param_res.append(param_frame)
            else:
                for tt in range(frag, int(30 * interval)):
                    frame_id = start_frame + tt
                    if frame_id < len(param_res):
                        scale = min((len(param_res) - frame_id) / frag, 1.0)
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0, tt, ii])
                            value = (1 - scale) * value + scale * param_res[frame_id][key]
                            param_res[frame_id][key] = round(value, 3)
                    else:
                        param_frame = {}
                        for ii, key in enumerate(self.p_list):
                            value = float(output[0, tt, ii])
                            param_frame[key] = round(value, 3)
                        param_res.append(param_frame)

            start_time = end_time - (frag / 10)
            end_time = start_time + interval
            if is_end is True:
                break

        param_res = self.mouth_smooth(param_res)

        logging.info(
            "generate {} frames in {}s with avg inference time {}ms/frame".format(
                len(param_res),
                round(time.time() - f1, 3),
                round((time.time() - f1) / len(param_res) * 1000, 3),
            )
        )

        return param_res, None, None
