import gc
import time
import statistics
import tiktoken
from transformers import GPT2TokenizerFast

def measure(fn, docs, repeats=5): #Warm up
    fn(docs)
    times = []
    for _ in range(repeats):
        gc.disable()
        start = time.perf_counter_ns()
        fn(docs)
        end = time.perf_counter_ns()
        gc.enable()
        times.append(end - start)
    return statistics.mean(times), statistics.stdev(times)

if __name__ == "__main__":
    docs = [...]
    num_bytes = sum(len(d.encode()) for d in docs)

    # tiktoken
    enc = tiktoken.get_encoding("gpt2")
    t_mean, t_stdev = measure(lambda d: enc.encode_ordinary_batch(d, num_threads=8), docs)

    # HF
    hf = GPT2TokenizerFast.from_pretrained("gpt2")
    hf.model_max_length = int(1e30)
    hf_mean, hf_stdev = measure(lambda d: hf(d, return_tensors=None), docs)

    print(f"tiktoken: {num_bytes / (t_mean)*1e9:.2f} ± {t_stdev/ t_mean*100:.1f}% байт/с")
    print(f"HuggingFace: {num_bytes / (hf_mean)*1e9:.2f} ± {hf_stdev/ hf_mean*100:.1f}% байт/с")
