from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL


@register_model("mimo_vl")
class MiMo_VL(Qwen2_5_VL):
    """MiMo-VL-7B model wrapper.

    MiMo-VL is fully compatible with the Qwen2_5_VL architecture
    (Qwen2_5_VLForConditionalGeneration) so we reuse that implementation.
    Default pretrained model is set to MiMo-VL-7B-SFT-2508.
    """

    def __init__(
        self,
        pretrained: str = "XiaomiMiMo/MiMo-VL-7B-SFT-2508",
        **kwargs,
    ):
        super().__init__(pretrained=pretrained, **kwargs)
