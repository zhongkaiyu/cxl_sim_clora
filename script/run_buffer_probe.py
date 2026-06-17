#!/usr/bin/env python3
"""Probe: when does the NDP controller buffer actually bind?

The buffer admits a sub-request iff buf_used_size + sub_req_inst_size <= buf_size
(flash.c). So #in-flight commands = buf_size / sub_req_inst_size. To sweep BELOW
1 command we must shrink sub_req_inst_size. We test long-KV (Uniform-long, the
CXL-bound workload) across small buffers, at low latency AND high latency, to
see whether deeper buffering is needed to hide CXL latency via more in-flight
requests (Little's law) or whether the workload is purely bandwidth-bound
(buffer irrelevant).
"""
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
TPUT_RE = re.compile(
    r"^CLoRA\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M)

BASE = ["--steps", "6", "--warmup-steps", "6", "--batch", "32",
        "--n-adapters", "1000", "--dist", "uniform",
        "--gpu-mem-gb", "26", "--base-model-gb", "14",
        "--model-d", "4096", "--n-layers", "32",
        "--gpu-compute-tflops", "312", "--gpu-mem-bw-gb", "1935",
        "--top-k-hot", "50", "--seed", "42",
        "--kv-min", "2048", "--kv-max", "4096"]   # long-KV


def make_conf(overrides, dest, tag):
    text = open(BASE_CONF).read()
    for k, v in overrides.items():
        pat = re.compile(rf"^{re.escape(k)}\s*=\s*[^;#\n]*;", re.M)
        repl = f"{k} = {v};"
        text = pat.sub(repl, text) if pat.search(text) else text + f"\n{repl}\n"
    path = os.path.join(dest, f"p_{tag}.conf")
    open(path, "w").write(text)
    return path


def run(conf):
    cmd = [sys.executable, DRIVER] + BASE + ["--c-parameter-file", conf]
    vals = []
    for _ in range(2):
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=300)
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-800:]); return None
        m = TPUT_RE.search(p.stdout)
        if not m:
            return None
        vals.append(float(m.group(1).replace(",", "")))
    return statistics.median(vals)


def main():
    d = tempfile.mkdtemp(prefix="bp_", dir="/tmp")
    INST = 32                       # finer command size so we can sweep < 4 commands
    bufs = [32, 64, 128, 256, 512, 1024, 2048]   # = 1,2,4,8,16,32,64 commands
    for lat in (0, 3200, 12800):    # L_CXL_switch ns: none / high / very high
        print(f"\n=== L_CXL_switch = {lat} ns (sub_req_inst_size={INST}) ===")
        for b in bufs:
            conf = make_conf({"sub_req_inst_size": INST,
                              "cxlctrl_buf_size": b,
                              "L_CXL_switch": lat}, d, f"{lat}_{b}")
            t = run(conf)
            cmds = b // INST
            print(f"  buf={b:>5} ({cmds:>2} cmd)  "
                  f"long-KV = {t:>8,.0f} tok/s" if t else
                  f"  buf={b:>5} ({cmds:>2} cmd)  FAILED/deadlock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
