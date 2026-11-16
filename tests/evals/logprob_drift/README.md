# Logprob Drift Evaluation

Some changes to vLLM alter what a model computes without moving any accuracy score: a value
rounded to bf16 too early, a stale KV block reused. This eval catches them from the model's
logprobs, usually before any answer changes.

No build reproduces itself exactly, so outputs can't simply be diffed. Instead the eval asks
whether a candidate build differs from a reference build by more than reruns differ from
each other.

## Usage

Record the reference build, typically `main` at the PR's merge base, then the candidate,
comparing it with the reference. Record both on the same machine, one after the other.

```bash
# On the reference build
pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py --drift-out=runs/main

# On the candidate build
pytest -s -v tests/evals/logprob_drift/test_logprob_drift.py \
    --drift-out=runs/pr --drift-reference=runs/main
```

Each run starts its own server with empty compile and autotune caches, and `--drift-out`
must be a new directory. The comparison is also written to `result.json` there. A run takes
a minute or two plus server startup with the small model.

Or run the steps by hand, starting a new server with empty caches for each run. This also
lets you alternate the two builds' runs:

```bash
C=$(mktemp -d)
VLLM_CACHE_ROOT=$C TRITON_CACHE_DIR=$C/triton VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$C/flashinfer \
    vllm serve Qwen/Qwen3-0.6B --max-model-len 2048 --max-num-seqs 32
python -m tests.evals.logprob_drift.logprob_drift_eval record --out runs/main/1.npz
python -m tests.evals.logprob_drift.logprob_drift_eval compare \
    --reference runs/main/*.npz --candidate runs/pr/*.npz
```

`record --setup` takes JSON, such as the server's arguments, that `compare` then requires to
match.

For example, replaying the bug fixed in
[#53877](https://github.com/vllm-project/vllm/pull/53877), with the fix (56058fd5) as the
reference and its parent (87b9b5b8) as the candidate, on Qwen/Qwen3.5-0.8B with
`--enforce-eager` on an A40. The bug rounded beta to bf16 in the packed Gated DeltaNet decode
kernel, so only decode sees it:

```text
First runs: probe KL 0.000183, same top token 99.54 %, 30.4 % of outputs differ
Across builds over rerun noise: probes 0.99×, decode 3.59×, traffic 0.98×
Verdict: CHANGED, 3.59× is above the 1.25× needed (p = 0.0040 over 252 splits, alpha 0.05)
```

Two builds of identical engine code (c0056b19 and e9499996, with Qwen3-0.6B) read
`same within noise, 0.97× is not above the 1.14× needed (p = 0.8214 …)`.

## What a run measures

Each run sends three kinds of requests to a fresh server:

- **Probes**: 32 chunks of 1,024 WikiText-2 tokens, scored with prompt logprobs on their last
  512 positions, as llama.cpp's `--kl-divergence` does. Both builds see identical tokens, so
  a kernel that rounds differently stands out even when no answer changes.
- **Decode**: the first half of each chunk, continued greedily for 128 tokens, one request
  at a time on an idle server. Decode kernels run without batching, so reruns usually
  repeat exactly and a kernel that decodes differently stands out.
- **Traffic**: 250 GSM8K questions behind a shared 5-shot prefix, generated greedily, more
  than fit in one batch. This runs prefix caching, chunked prefill mixed with decode,
  CUDA-graph padding and KV block reuse, as serving does. Probes can't reach these: prompt
  logprobs skip the prefix cache and never decode.

Two generated outputs are compared up to and including their first differing token.

KL is taken over the top 20 tokens of each side, a lower bound on the true KL: a token one
side lists and the other doesn't gets at most the other side's smallest listed probability.

## How it decides

Reruns differ because batches form differently each time and a fresh server may compile and
autotune its kernels differently. So each part is scored as the mean KL across builds over
the mean KL between reruns: about 1× when nothing changed. The score is the largest of the
three.

- Probes and decode take their noise from the reruns of both builds, since their reruns
  are often bit-identical.
- Traffic takes its noise from the reference's reruns alone. A bug that corrupts the KV cache
  at random makes the candidate's runs disagree with each other, and pooling would count that
  as noise.

The verdict is a permutation test. If the builds are the same, the real split of the runs
into reference and candidate is just one of many equally likely splits. The candidate reads
changed when its score is above what more than `alpha` of all splits reach: the score
needed. Unchanged code then reads changed at most `alpha` of the time, with no thresholds to
tune. That holds only when nothing but the build differs between runs, hence a fresh server
with empty caches for every run, and both builds on one machine in one sitting.

The score needed also tells what a "same" verdict rules out: a change that would have
scored above it.

Both builds need the same number of runs, from 4 to 8 for `alpha = 0.05`. The reversed
split can score as high as the real one, so p is at least 2 over the number of splits: 4
runs give 70 splits, 3 only 20. `compare` refuses other counts.

## Configuration

```yaml
model_name: "Qwen/Qwen3-0.6B"
runs: 5                   # Runs per build, 4 to 8, each with its own server
server_args: "--max-model-len 2048 --max-num-seqs 32"
env:                      # Environment variables for the server (optional)
  VLLM_USE_FLASHINFER_SAMPLER: "0"
startup_max_wait_seconds: 1200  # Optional
```

Runs record their `server_args`, `env`, model, GPU type and data, and `compare` refuses runs
that differ in any of them. The GPU type is read where the client runs, so run the client
on the server's machine. A check covers only the code paths these settings run: an eager
kernel is checked only with `--enforce-eager`.

## Limits

- It finds changes, not bugs. An intended numerics change reads as changed; then the
  statistics above tell how large it is.
- A change that doesn't score above the score needed is missed: the verdict says how large
  that is. A change that only makes reruns disagree more is seen only in traffic.
- A bug that strikes only some runs must strike enough of them to stand out. Record more
  runs when such a bug is the worry.
- A build that tokenizes the data differently is refused, not compared.
