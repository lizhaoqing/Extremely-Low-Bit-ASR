from distutils.util import strtobool


def add_arguments_bitsharing_common(group):
    group.add_argument(
        "--enc-weight-bit",
        type=int,
        default=8,
        help="Encoder weight bit-width for quantization",
    )
    group.add_argument(
        "--weight-alpha",
        type=float,
        default=1,
        help="Weight scaling factor alpha",
    )
    group.add_argument(
        "--quant-mode",
        type=str,
        default="symmetric",
        choices=["symmetric", "asymmetric"],
        help="Quantization mode",
    )
    group.add_argument(
        "--per-channel",
        default=False,
        type=strtobool,
        help="Per-channel quantization",
    )
    group.add_argument(
        "--use-scaling",
        default=False,
        type=strtobool,
        help="Use learnable scaling factor alpha",
    )
    group.add_argument(
        "--enc-shared-layer-num",
        type=int,
        default=1,
        help="Number of shared encoder layer iterations",
    )
    group.add_argument(
        "--dec-shared-layer-num",
        type=int,
        default=1,
        help="Number of shared decoder layer iterations",
    )
    group.add_argument(
        "--dec-weight-bit",
        type=int,
        default=8,
        help="Decoder weight bit-width for quantization",
    )
    group.add_argument(
        "--dec-extra-bit",
        type=int,
        default=4,
        help="Decoder extra bit-width",
    )
    group.add_argument(
        "--quant-decoder",
        default=False,
        type=strtobool,
        help="Whether to quantize decoder",
    )
    group.add_argument(
        "--quant-cnn",
        default=False,
        type=strtobool,
        help="Whether to quantize CNN module",
    )
    group.add_argument(
        "--mix-rate",
        type=float,
        default=0.5,
        help="Stochastic precision mix rate schedule (1.8=log-linear)",
    )
    group.add_argument(
        "--lambda-1",
        type=float,
        default=1.,
        help="Weight for hard-label CE loss",
    )
    group.add_argument(
        "--lambda-2",
        type=float,
        default=1.,
        help="Weight for soft-label KD loss",
    )

    return group
