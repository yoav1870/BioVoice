from __future__ import annotations

from pprint import pprint

import torch


def _safe_attrs(obj, keys: list[str]) -> dict[str, object]:
    return {key: getattr(obj, key) for key in keys if hasattr(obj, key)}


def describe_spec(spec) -> None:
    print("=== SPEC MODULE ===")
    print(spec)
    print()
    print("spec class:", type(spec))

    torchfbank = getattr(spec, "torchfbank", None)
    if torchfbank is None:
        print("\nNo torchfbank attribute found.")
        return

    print("\n=== TORCHF BANK ===")
    print(torchfbank)

    for i, layer in enumerate(torchfbank):
        print(f"\n=== LAYER {i} ===")
        print("type:", type(layer))
        print(layer)

        layer_vals = _safe_attrs(
            layer,
            [
                "sample_rate",
                "n_fft",
                "win_length",
                "hop_length",
                "f_min",
                "f_max",
                "n_mels",
                "center",
                "power",
                "normalized",
                "norm",
                "mel_scale",
            ],
        )
        if layer_vals:
            print("layer attrs:")
            pprint(layer_vals)

        spectrogram = getattr(layer, "spectrogram", None)
        if spectrogram is not None:
            print("spectrogram attrs:")
            pprint(
                _safe_attrs(
                    spectrogram,
                    [
                        "n_fft",
                        "win_length",
                        "hop_length",
                        "power",
                        "center",
                        "normalized",
                    ],
                )
            )

        mel_scale = getattr(layer, "mel_scale", None)
        if mel_scale is not None:
            print("mel_scale attrs:")
            pprint(
                _safe_attrs(
                    mel_scale,
                    ["sample_rate", "f_min", "f_max", "n_mels", "norm", "mel_scale"],
                )
            )


def load_official_redimnet():
    device = torch.device("cpu")
    model = (
        torch.hub.load(
            "IDRnD/ReDimNet",
            "ReDimNet",
            model_name="b6",
            train_type="ptn",
            dataset="vox2",
        )
        .to(device)
        .eval()
    )
    return model


def main() -> None:
    print("Loading official ReDimNet from torch hub...")
    model = load_official_redimnet()
    print("Loaded model class:", type(model))

    spec = getattr(model, "spec", None)
    if spec is None:
        raise RuntimeError("Loaded model does not expose a `.spec` module.")

    describe_spec(spec)

    print("\n=== DUMMY FORWARD THROUGH SPEC ===")
    with torch.no_grad():
        wav = torch.zeros(1, 16000)
        mel = spec(wav)
    print("dummy mel shape:", tuple(mel.shape))


if __name__ == "__main__":
    main()
