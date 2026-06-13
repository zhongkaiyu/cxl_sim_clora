#!/usr/bin/env python3
"""Reviewer-requested hardware sensitivity study for CLoRA.

Sweeps four hardware parameters one at a time around the paper's operating
point (Table 4: A100 + 4 CLoRA devices, Llama2-7B) and reports decode
throughput:

  1. CXL link latency      -- added one-way switch traversal latency.
                              C sim: L_CXL_switch (per CA transfer and per
                              data return, i.e. 2x per round trip, matching
                              the 2*L_CXL term in paper Eq 8).
                              Cost model: hw.cxl_latency = 200 + delta.
  2. CXL link bandwidth    -- C sim: cxl_bandwidth (B/ns == GB/s);
                              cost model: hw.cxl_link_bw (W_CXL).
  3. NDP core throughput   -- C sim: chip_computing_power (GOPS) with
                              ndp_compute_model=1 (ops-based PE timing;
                              the legacy model collapses to ~1 ns for
                              addr_num==1 sub-requests and would show a
                              flat line). Cost model: hw.cxl_compute (C_CXL).
  4. NDP buffer size       -- C sim: cxlctrl_buf_size. The controller admits
                              a sub-request only while
                              buf_used + sub_req_inst_size <= buf_size
                              (flash.c), so buf_size/128 is the number of
                              in-flight NDP requests per device.

Each sweep point runs the full driver (10 warmup + 10 measured steps,
batch 32, 1000 adapters, seed 42) on two workloads chosen to cover both
regimes: Uniform (HBM-bound) and Uniform-long (CXL-traffic-bound).
Median of N_TRIALS runs absorbs the C sim's srand(time(NULL)) jitter.

Usage:  python3 script/sensitivity_study.py [--trials 3] [--out SENSITIVITY.md]
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRIVER = os.path.join(HERE, "clora_driver.py")
BASE_CONF = os.path.join(ROOT, "config", "parameters.conf")

THROUGHPUT_RE = re.compile(
    r"^CLoRA\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)",
    re.MULTILINE)

# ---- fixed operating point (RESULTS.md, Llama2-7B on A100) -----------------

BASE_ARGS = [
    "--steps", "10", "--warmup-steps", "10", "--batch", "32",
    "--n-adapters", "1000", "--dist", "uniform",
    "--gpu-mem-gb", "26", "--base-model-gb", "14",
    "--model-d", "4096", "--n-layers", "32",
    "--gpu-compute-tflops", "312", "--gpu-mem-bw-gb", "1935",
    "--top-k-hot", "50", "--seed", "42",
]

WORKLOADS = {
    "Uniform":      ["--kv-min", "100",  "--kv-max", "1024"],
    "Uniform-long": ["--kv-min", "2048", "--kv-max", "4096"],
}

# ---- sweep definitions ------------------------------------------------------
# Each point: (label, conf_overrides: {key: value}, driver_flags: [..])
# The first point of each sweep is the paper/RESULTS.md default.

SWEEPS = {
    "cxl_latency": {
        "title": "CXL link latency (added one-way switch latency, ns)",
        "x_label": "added latency (ns)",
        "points": [
            (f"+{d}", {"L_CXL_switch": d},
             ["--cxl-latency-ns", str(200 + d)])
            for d in [0, 100, 200, 400, 800, 1600]
        ],
    },
    "cxl_bandwidth": {
        "title": "CXL link bandwidth per device (GB/s)",
        "x_label": "link bandwidth (GB/s)",
        "points": [
            (str(bw), {"cxl_bandwidth": bw},
             ["--cxl-link-bw-gb", str(bw)])
            for bw in [128, 32, 64, 256, 512]
        ],
    },
    "ndp_throughput": {
        "title": "NDP core throughput per device (FP16 TFLOPS, "
                 "ops-based PE model)",
        "x_label": "NDP TFLOPS / device",
        "points":
            # continuity reference: default conf (legacy PE model)
            [("2.0 (legacy PE model)", {}, ["--ndp-tflops", "2.0"])] +
            [(str(tf),
              {"chip_computing_power": int(tf * 1000), "ndp_compute_model": 1},
              ["--ndp-tflops", str(tf)])
             for tf in [2.0, 0.25, 0.5, 1.0, 4.0, 8.0]],
    },
    "ndp_buffer": {
        "title": "NDP controller buffer size (bytes; 128 B per in-flight "
                 "request)",
        "x_label": "buffer size (B)",
        "points": [
            (f"{b} ({b // 128} reqs)", {"cxlctrl_buf_size": b}, [])
            for b in [1024, 128, 256, 512, 2048, 4096]
        ],
    },
}


# ---- conf generation ---------------------------------------------------------

def make_conf(overrides: dict, dest_dir: str, tag: str) -> str:
    with open(BASE_CONF) as f:
        text = f.read()
    for key, value in overrides.items():
        pattern = re.compile(rf"^{re.escape(key)}\s*=\s*[^;#\n]*;",
                             re.MULTILINE)
        replacement = f"{key} = {value};"
        if pattern.search(text):
            text = pattern.sub(replacement, text)
        else:
            # insert ahead of the static-parameters section so load_parameters
            # sees it with the other hardware keys
            text += f"\n{replacement}\n"
    path = os.path.join(dest_dir, f"parameters_{tag}.conf")
    with open(path, "w") as f:
        f.write(text)
    return path


# ---- runner -----------------------------------------------------------------

def run_point(conf_path: str, driver_flags: list, workload_flags: list,
              trials: int) -> float:
    samples = []
    for _ in range(trials):
        cmd = ([sys.executable, DRIVER] + BASE_ARGS + workload_flags
               + driver_flags + ["--c-parameter-file", conf_path])
        proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, timeout=1800)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-2000:])
            raise RuntimeError(f"driver failed: {' '.join(cmd)}")
        m = THROUGHPUT_RE.search(proc.stdout)
        if not m:
            sys.stderr.write(proc.stdout[-2000:])
            raise RuntimeError("CLoRA throughput line not found")
        samples.append(float(m.group(1).replace(",", "")))
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3,
                    help="runs per sweep point; median reported (default 3)")
    ap.add_argument("--out", default=os.path.join(ROOT, "SENSITIVITY.md"))
    ap.add_argument("--json-out",
                    default=os.path.join(HERE, "sensitivity_results.json"))
    ap.add_argument("--only", choices=list(SWEEPS) + ["all"], default="all")
    args = ap.parse_args()

    sweeps = SWEEPS if args.only == "all" else {args.only: SWEEPS[args.only]}
    # The C binary stores the parameter path in a char[80]; keep paths short
    # (macOS default tempdir is ~50 chars before the filename even starts).
    conf_dir = tempfile.mkdtemp(prefix="cs_", dir="/tmp")
    results = {}
    t0 = time.time()

    n_points = sum(len(s["points"]) for s in sweeps.values()) * len(WORKLOADS)
    done = 0
    for sweep_name, sweep in sweeps.items():
        results[sweep_name] = {"title": sweep["title"], "rows": []}
        for idx, (label, conf_over, drv_flags) in enumerate(sweep["points"]):
            tag = f"{sweep_name[:6]}{idx}"
            conf_path = make_conf(conf_over, conf_dir, tag)
            row = {"label": label, "conf": conf_over,
                   "driver_flags": drv_flags, "throughput": {}}
            for wname, wflags in WORKLOADS.items():
                tput = run_point(conf_path, drv_flags, wflags, args.trials)
                row["throughput"][wname] = tput
                done += 1
                print(f"[{done:3d}/{n_points}] {sweep_name:15s} {label:24s} "
                      f"{wname:13s} {tput:>10,.1f} tok/s "
                      f"({time.time() - t0:5.0f}s elapsed)", flush=True)
            results[sweep_name]["rows"].append(row)

    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=2)

    write_markdown(results, args.out, args.trials)
    print(f"\nwrote {args.out} and {args.json_out} "
          f"({time.time() - t0:.0f}s total)")
    return 0


# ---- report -----------------------------------------------------------------

def write_markdown(results: dict, out_path: str, trials: int) -> None:
    lines = [
        "# CLoRA Hardware Sensitivity Study",
        "",
        "Reviewer-requested one-factor-at-a-time sweeps around the paper's",
        "operating point (Table 4: A100, 4 CLoRA devices, Llama2-7B,",
        "batch 32, 1000 adapters, seed 42, 10 warmup + 10 measured steps,",
        f"median of {trials} trials). The first row of each table is the",
        "default configuration; 'rel' columns are normalized to it.",
        "",
        "Knob plumbing: every point sets the C event simulator's parameter",
        "file *and* the matching cost-model value, so both the measured",
        "timing and the strategy-selection algorithm (Algorithm 1, Eqs 1-9)",
        "see the same hardware.",
        "",
    ]
    for sweep_name, sweep in results.items():
        lines += [f"## {sweep['title']}", ""]
        wnames = list(sweep["rows"][0]["throughput"].keys())
        header = "| point |"
        sep = "|---|"
        for w in wnames:
            header += f" {w} (tok/s) | rel |"
            sep += "---:|---:|"
        lines += [header, sep]
        base = sweep["rows"][0]["throughput"]
        for row in sweep["rows"]:
            cells = [row["label"]]
            for w in wnames:
                tput = row["throughput"][w]
                rel = tput / base[w] if base[w] else float("nan")
                cells += [f"{tput:,.0f}", f"{rel:.2f}x"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
