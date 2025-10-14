import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, Optional

#############################
# 1) 모델 정의
#############################
class ExerciseTransformer(nn.Module):
    """
    Transformer 기반 분류 모델.
    - input_projection: (input_dim -> d_model)
    - transformer_encoder: num_layers x EncoderLayer
    - fc: (d_model -> num_classes)
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        num_classes: int,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.batch_first = batch_first
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape depends on batch_first
        # if batch_first: (batch, seq, input_dim)
        # else:          (seq, batch, input_dim)
        x = self.input_projection(src)
        x = self.transformer_encoder(x)
        if self.batch_first:
            x = x.mean(dim=1)  # (batch, d_model)
        else:
            x = x.mean(dim=0)  # (batch, d_model)
        return self.fc(x)


#############################
# 2) 체크포인트 유추 함수
#############################

def _get(sd: dict, *keys: str) -> Optional[torch.Tensor]:
    for k in keys:
        v = sd.get(k)
        if v is not None:
            return v
    return None


def infer_ckpt_hparams(sd: dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """ckpt에서 d_model, dim_ff, num_layers 유추.
    returns: (d_model, dim_feedforward, num_layers)
    """
    # d_model: self_attn.out_proj.weight: [d_model, d_model]
    w_out = _get(
        sd,
        "transformer.layers.0.self_attn.out_proj.weight",
        "transformer_encoder.layers.0.self_attn.out_proj.weight",
    )
    d_model = w_out.shape[0] if w_out is not None else None

    # dim_ff: linear1.weight: [dim_ff, d_model]
    w_ff = _get(
        sd,
        "transformer.layers.0.linear1.weight",
        "transformer_encoder.layers.0.linear1.weight",
    )
    dim_ff = w_ff.shape[0] if w_ff is not None else None

    # num_layers: 키에서 최대 인덱스 + 1
    layer_idxs = set()
    for k in sd.keys():
        if "layers." in k:
            try:
                # ...layers.{i}.
                idx = int(k.split("layers.")[1].split(".")[0])
                layer_idxs.add(idx)
            except Exception:
                pass
    num_layers = (max(layer_idxs) + 1) if layer_idxs else None
    return d_model, dim_ff, num_layers


#############################
# 3) 키 리매핑 + 로더
#############################

def remap_keys(sd: dict) -> OrderedDict:
    """과거 체크포인트 키 -> 현재 모델 키 리매핑."""
    new_sd = OrderedDict()
    for k, v in sd.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        # transformer -> transformer_encoder
        if name.startswith("transformer.layers."):
            name = name.replace("transformer.layers.", "transformer_encoder.layers.")
        # 입력 프로젝션 명칭 통일
        if name.startswith("input_proj."):
            name = name.replace("input_proj", "input_projection")
        if name.startswith("input_linear."):
            name = name.replace("input_linear", "input_projection")
        new_sd[name] = v
    return new_sd


def load_compatible(model: nn.Module, ckpt_path: str, map_location: str = "cpu") -> None:
    import math
    pkg = torch.load(ckpt_path, map_location=map_location)
    sd = pkg.get("state_dict", pkg)
    sd = remap_keys(sd)

    # 유사 shape 키를 이용해 input_projection 채우기 시도
    msd = model.state_dict()
    missing_before = [k for k in msd.keys() if k not in sd]
    extras = [k for k in sd.keys() if k not in msd]

    for k in list(missing_before):
        if k.startswith("input_projection.") and ("weight" in k or "bias" in k):
            want_shape = msd[k].shape
            cands = [e for e in extras if ("input" in e or "proj" in e or "projection" in e or "embed" in e) and sd[e].shape == want_shape]
            if cands:
                sd[k] = sd[cands[0]]

    result = model.load_state_dict(sd, strict=False)

    # input_projection이 여전히 비면 합리적 초기화
    still_missing = set(result.missing_keys)
    if "input_projection.weight" in still_missing:
        with torch.no_grad():
            nn.init.kaiming_uniform_(model.input_projection.weight, a=math.sqrt(5))
            if model.input_projection.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.input_projection.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(model.input_projection.bias, -bound, bound)
        still_missing.discard("input_projection.weight")
        still_missing.discard("input_projection.bias")

    print("[load] missing:", sorted(list(still_missing)))
    print("[load] unexpected:", result.unexpected_keys)


    # 유사 shape 키를 이용해 input_projection 채우기 시도
    msd = model.state_dict()
    missing = [k for k in msd.keys() if k not in sd]
    extras = [k for k in sd.keys() if k not in msd]

    for k in list(missing):
        if k.startswith("input_projection.") and ("weight" in k or "bias" in k):
            want_shape = msd[k].shape
            cands = [e for e in extras if ("input" in e or "proj" in e) and sd[e].shape == want_shape]
            if cands:
                sd[k] = sd[cands[0]]

    result = model.load_state_dict(sd, strict=False)
    print("[load] missing:", result.missing_keys)
    print("[load] unexpected:", result.unexpected_keys)


#############################
# 4) 하이퍼파라미터 설정
#############################
WEIGHT_FILE_PATH = "best_transformer.pth"
NUM_CLASSES = 4
INPUT_DIM = 132  # 알 수 없으면 기존 데이터 피쳐 수로 유지
NHEAD = 4        # ckpt로부터 정확 추정 어려움. 학습 시 설정과 일치해야 함
BATCH_FIRST = True

# ckpt 먼저 열어 추정치 확보
_ckpt_pkg = torch.load(WEIGHT_FILE_PATH, map_location="cpu")
_ckpt_sd = _ckpt_pkg.get("state_dict", _ckpt_pkg)
_ckpt_sd = remap_keys(_ckpt_sd)

_D_MODEL_CKPT, _DIMFF_CKPT, _NUML_CKPT = infer_ckpt_hparams(_ckpt_sd)

D_MODEL = _D_MODEL_CKPT if _D_MODEL_CKPT is not None else 132
DIM_FEEDFORWARD = _DIMFF_CKPT if _DIMFF_CKPT is not None else 2048
NUM_LAYERS = _NUML_CKPT if _NUML_CKPT is not None else 2

print(f"[infer] d_model={D_MODEL}, dim_ff={DIM_FEEDFORWARD}, num_layers={NUM_LAYERS}")

#############################
# 5) 모델 생성 및 가중치 로드
#############################
model = ExerciseTransformer(
    input_dim=INPUT_DIM,
    d_model=D_MODEL,
    nhead=NHEAD,
    dim_feedforward=DIM_FEEDFORWARD,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    batch_first=BATCH_FIRST,
)

load_compatible(model, WEIGHT_FILE_PATH, map_location="cpu")
model.eval()

#############################
# 6) 간단 추론 테스트
#############################
if __name__ == "main" or __name__ == "__main__":
    SEQ_LEN = 100
    BATCH = 1
    if BATCH_FIRST:
        dummy = torch.rand(BATCH, SEQ_LEN, INPUT_DIM)
    else:
        dummy = torch.rand(SEQ_LEN, BATCH, INPUT_DIM)
    with torch.no_grad():
        logits = model(dummy)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
    print("[shape] input:", tuple(dummy.shape))
    print("[shape] logits:", tuple(logits.shape))
    print("[pred] class:", int(pred.item()))
