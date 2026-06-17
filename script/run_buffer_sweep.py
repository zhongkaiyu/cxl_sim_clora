#!/usr/bin/env python3
"""NDP controller buffer sweep for sensitivity panel (e), resolving the knee.

The buffer admits a sub-request iff buf_used + sub_req_inst_size <= buf_size,
so in-flight depth = buf_size / sub_req_inst_size (sub_req_inst_size=128, the
Table-4 value). The smallest functional buffer is 1 command (buf=128); below
that nothing is admitted (deadlock). We sweep small command-depths and show
two regimes for the CXL-bound long-KV workload:
  * default link latency  -> buffer barely matters (knee at ~2 commands, ~2%)
  * high link latency (6.4 us) -> buffer must be deep enough to hide latency
    via more in-flight requests; a 1-command buffer loses ~10%.

This is the honest "rise then saturate": the buffer is only a bottleneck when
the link is latency-bound. Updates the "ndp_buffer" entry of sensitivity_abs.json.
"""
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRIVER = os.path.join(HERE, "clora_driver.py")
BASE_CONF = os.path.join(ROOT, "config", "parameters.conf")
TRIALS = 3
INST = 128                                   # sub_req_inst_size (Table 4)
DEPTHS = [1, 2, 3, 4, 8, 16, 32]             # in-flight commands
TPUT_RE = re.compile(
    r"^CLoRA\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M)

BASE = ["--steps", "8", "--warmup-steps", "8", "--batch", "32",
        "--n-adapters", "1000", "--dist", "uniform",
        "--gpu-mem-gb", "26", "--base-model-gb", "14",
        "--model-d", "4096", "--n-layers", "32",
        "--gpu-compute-tflops", "312", "--gpu-mem-bw-gb", "1935",
        "--top-k-hot", "50", "--seed", "42",
        "--kv-min", "2048", "--kv-max", "4096"]    # long-KV (CXL-bound)
LATS = {"Long-KV, L=200ns": 0, "Long-KV, L=6.4us": 6400}   # L_CXL_switch ns


def make_conf(buf, lat, dest):
    text = open(BASE_CONF).read()
    for k, v in {"sub_req_inst_size": INST, "cxlctrl_buf_size": buf,
                 "L_CXL_switch": lat}.items():
        pat = re.compile(rf"^{re.escape(k)}\s*=\s*[^;#\n]*;", re.M)
        text = pat.sub(f"{k} = {v};", text) if pat.search(text) else text + f"\n{k} = {v};\n"
    path = os.path.join(dest, f"p_{lat}_{buf}.conf")
    open(path, "w").write(text)
    return path


def run(conf):
    cmd = [sys.executable, DRIVER] + BASE + ["--c-parameter-file", conf]
    vals = []
    for _ in range(TRIALS):
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=300)
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-800:]); raise RuntimeError("driver failed")
        vals.append(float(TPUT_RE.search(p.stdout).group(1).replace(",", "")))
    return statistics.median(vals)


def main():
    d = tempfile.mkdtemp(prefix="bs_", dir="/tmp")
    series = {name: [] for name in LATS}
    for name, lat in LATS.items():
        print(f"=== {name} ===")
        for depth in DEPTHS:
            t = run(make_conf(depth * INST, lat, d))
            series[name].append(t)
            print(f"  depth={depth:>2} cmd  {t:>8,.0f} tok/s", flush=True)

    rows = []
    for i, depth in enumerate(DEPTHS):
        row = {"x": depth}
        for name in LATS:
            row[name] = round(series[name][i], 1)
        rows.append(row)

    path = os.path.join(HERE, "sensitivity_abs.json")
    j = json.load(open(path))
    j["ndp_buffer"] = {
        "title": "(e) NDP controller buffer",
        "xlabel": "in-flight commands (buf / sub_req_inst)",
        "default_x": 8, "log": True, "rows": rows,
        "series": list(LATS.keys()),
        "note": "long-KV; buffer binds only when the link is latency-bound "
                "(deeper buffer hides CXL latency via more in-flight requests)."}
    json.dump(j, open(path, "w"), indent=2)
    print("\nupdated ndp_buffer in script/sensitivity_abs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
