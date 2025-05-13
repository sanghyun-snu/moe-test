import gguf
import numpy as np
from pathlib import Path

def load_gguf_quantized_model(gguf_path):
    """
    GGUF 파일을 로드하고(양자화 여부에 상관없이) 모두 float32 ndarray로 반환.
    지원되지 않는 quantization type이 나오면 경고를 출력하고 빈 dict 반환.
    """
    try:
        reader = gguf.GGUFReader(gguf_path)
    except ValueError as e:
        # 예: np.uint32(31) is not a valid GGMLQuantizationType
        print(f"[Warning] Unsupported quantization type in {gguf_path}:\n  {e}")
        print("  → gguf 패키지를 최신 버전으로 업그레이드해 보세요: pip install --upgrade gguf")
        return {}

    tensors = {}
    for t in reader.tensors:
        # dequantize() 메소드가 있으면 양자화 텐서로 간주하고 디양자화
        if callable(getattr(t, "dequantize", None)):
            try:
                arr = t.dequantize()
            except Exception:
                arr = np.array(t.data, dtype=np.float32)
        else:
            # 비양자화(FP16, FP32)일 때
            arr = np.array(t.data, dtype=np.float32)

        # 원래 shape으로 복원
        try:
            arr = arr.reshape(t.shape)
        except Exception:
            pass

        tensors[t.name] = arr
    return tensors

def analyze_sparsity(tensor, threshold=1e-6):
    if threshold == 0:
        near_zero = tensor == 0
    else:
        near_zero = np.abs(tensor) < threshold
    total_elements = tensor.size
    zero_elements = np.sum(near_zero)
    sparsity = zero_elements / total_elements
    return sparsity, total_elements, zero_elements

def main(gguf_path, threshold=1e-6):
    print(f"Loading GGUF model from: {gguf_path}")
    print(f"Using threshold: {threshold}")

    tensors = load_gguf_quantized_model(gguf_path)
    if not tensors:
        print("Model loading skipped due to unsupported quantization.")
        return

    print("\nAnalyzing sparsity for each tensor...\n")
    total_sparsity = 0
    total_tensors = 0

    for name, tensor in sorted(tensors.items()):
        sparsity, total_elements, zero_elements = analyze_sparsity(tensor, threshold)
        print(f"Tensor: {name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Total elements: {total_elements:,}")
        print(f"  Zero elements: {zero_elements:,}")
        print(f"  Sparsity: {sparsity:.4%}\n")
        total_sparsity += sparsity
        total_tensors += 1

    avg_sparsity = total_sparsity / total_tensors if total_tensors else 0
    print("Summary:")
    print(f"  Total tensors analyzed: {total_tensors}")
    print(f"  Average sparsity: {avg_sparsity:.4%}")

if __name__ == "__main__":
    gguf_file = "DS4X8R1L3.1-Dp-Thnkr-UnC-24B-D_AU-IQ4_XS.gguf"
    threshold = 0
    if not Path(gguf_file).exists():
        print(f"GGUF file not found: {gguf_file}")
    else:
        main(gguf_file, threshold)
