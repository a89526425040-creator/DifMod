from difmod.config import ModelConfig
from difmod.model import DifMod


def main() -> None:
    model = DifMod(ModelConfig())
    groups = {
        "text_encoder": model.text_encoder,
        "autoencoder": model.autoencoder,
        "flow_transformer": model.flow,
        "total": model,
    }
    for name, module in groups.items():
        count = sum(parameter.numel() for parameter in module.parameters())
        fp32_mb = count * 4 / (1024 ** 2)
        int8_mb = count / (1024 ** 2)
        print(f"{name}: {count:,} parameters | FP32 weights {fp32_mb:.2f} MiB | INT8 ideal {int8_mb:.2f} MiB")


if __name__ == "__main__":
    main()
