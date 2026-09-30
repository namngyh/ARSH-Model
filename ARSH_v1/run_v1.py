"""Inventory historical bars or run the five ARSH v1 regime diagnostics."""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

from v1_core import (
    EXPECTED_HASHES, STATES, analyze_periods, entropy_stats, filter_observations,
    load_sources, make_observations, model_for, overlap_jsd, records_frame,
    transition_duration,
)
from v1_shadow import run_shadow


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def render_report(name: str, report: dict) -> str:
    lines = [
        f"# ARSH v1 — 5 kiểm định lịch sử ({name})", "",
        f"- Model SHA-256: {report['model']['sha256']}",
        f"- Nến nguồn: {report['input']['bars']:,}; lợi suất 1 phút hợp lệ: {report['input']['accepted_returns']:,}.",
        f"- Khoảng quan sát: {report['input']['first']} đến {report['input']['last']}.",
        "- Các số trên dữ liệu trước khi model được khóa chỉ mô tả lịch sử; không chứng minh hiệu quả tương lai.",
        "- CSV đầu ra có event_bar_end nhưng available_at để trống vì CSV lịch sử không chứng minh thời điểm nến thực sự sẵn dùng.",
        "",
        "## 1. Entropy posterior", "",
    ]
    e = report["posterior_entropy"]
    lines += [
        f"Entropy chuẩn hóa trung bình: **{e['mean_normalized']:.4f}** (0 = tập trung vào một state, 1 = đều trên 7 state).",
        f"Tỷ lệ quan sát có entropy chuẩn hóa > 0,8: {e['fraction_normalized_above_0_8']:.2%}.",
        "Entropy thấp chỉ cho thấy model tự tin, không chứng minh state đúng.", "",
        "## 2. Chồng lấn và JSD của emission", "",
        "| Cặp | Overlap | JSD (nat) |", "|---|---:|---:|",
    ]
    for pair in report["emission_overlap_jsd"]["pairs"][:10]:
        lines.append(f"| {pair['states'][0]}–{pair['states'][1]} | {pair['overlap']:.4f} | {pair['jsd_nats']:.4f} |")
    lines += [
        "", "Bảng đủ 21 cặp và hai ma trận nằm trong diagnostics JSON.", "",
        "## 3–4. Chuyển trạng thái và thời lượng", "",
        "| State | P(ở lại) lý thuyết | Thời lượng kỳ vọng (quan sát) | Số chuỗi argmax | Trung bình chuỗi argmax |",
        "|---|---:|---:|---:|---:|",
    ]
    t = report["transition_and_duration"]
    for i, state in enumerate(STATES):
        run = t["observed_argmax_runs"][i]
        mean = "NA" if run["mean_observations"] is None else f"{run['mean_observations']:.2f}"
        lines.append(
            f"| {state} | {t['theoretical_transition'][i][i]:.4f} | "
            f"{t['theoretical_duration_observations'][i]:.2f} | {run['run_count']} | {mean} |"
        )
    lines += [
        "", "Thời lượng HMM là từ ma trận tham số; chuỗi argmax là nhãn hiển thị thực tế và có thể ngắn hơn nhiều.",
        "", "## 5. Đặc điểm kinh tế", "",
        "| State | Soft share | Return cùng phút (%) | Độ lệch chuẩn (%) | Return phút kế tiếp (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["economic_profiles"]["profiles"]:
        next_mean = row.get("next_observation_mean_return_pct")
        lines.append(
            f"| {row['state']} | {row.get('soft_share', 0):.2%} | "
            f"{row.get('contemporaneous_mean_return_pct', float('nan')):.5f} | "
            f"{row.get('contemporaneous_std_return_pct', float('nan')):.5f} | "
            f"{'NA' if next_mean is None else f'{next_mean:.5f}'} |"
        )
    lines += [
        "", "Phút kế tiếp chỉ được ghép trong cùng chuỗi liên tục. Đây là mô tả thống kê, không phải hiệu quả giao dịch.",
        "", "## Tách giai đoạn lịch sử", "",
        "| Giai đoạn | Số quan sát | Số ngày | Entropy chuẩn hóa TB |",
        "|---|---:|---:|---:|",
    ]
    for period, info in report["periods"].items():
        lines.append(
            f"| {period} | {info['observations']:,} | {info['days']} | "
            f"{info['posterior_entropy']['mean_normalized']:.4f} |"
        )
    if not any(key.startswith("from_2026_09_30") for key in report["periods"]):
        lines += ["", "**Chưa có quan sát từ 30/09/2026 trở đi trong đầu vào này.**"]
    if "online_shadow" in report:
        s = report["online_shadow"]
        lines += ["", "## Online learning chạy song song", "",
                  f"Trạng thái: {s['status']}; số cập nhật: {s['updates']}."]
        if s["updates"]:
            lines.append(f"Chênh lệch log density trước cập nhật (shadow − cố định): {s['mean_shadow_minus_frozen']:.6f}.")
        lines.append("Bộ cập nhật này là nguyên mẫu nghiên cứu; không thay artifact đang báo cáo.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["inventory", "analyze"], default="analyze")
    parser.add_argument("--data", type=Path, nargs="+", required=True,
                        help="One full-history v0.5 CSV and/or a directory of versioned day CSVs")
    parser.add_argument("--model", choices=["strict", "original", "both"], default="strict")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--shadow-start", help="YYYY-MM-DD; experimental updater begins only from this date")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw, source_audit = load_sources(args.data)
    inventory = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_files": source_audit, "bars": len(raw),
        "first": str(raw.index.min()), "last": str(raw.index.max()),
        "days": int(raw.index.normalize().nunique()),
        "python": platform.python_version(),
        "note": "Read-only inventory. No raw bars are copied into output.",
    }
    write_json(out / "DATA_INVENTORY.json", inventory)
    print(f"Inventory: {out / 'DATA_INVENTORY.json'}")
    if args.mode == "inventory":
        return
    obs, starts, return_audit = make_observations(raw)
    names = ["strict", "original"] if args.model == "both" else [args.model]
    for name in names:
        runtime, actual_hash, stable_to_internal = model_for(name)
        posterior, score, soft_trans = filter_observations(runtime, stable_to_internal, obs, starts)
        _, entropy = entropy_stats(posterior)
        records = records_frame(obs, starts, posterior, score, name, actual_hash, runtime.model_version)
        records_path = out / f"states_{name}.csv"
        records.to_csv(records_path, index=False, encoding="utf-8-sig")
        report = {
            "status": "historical_diagnostics_complete",
            "model": {"name": name, "sha256": actual_hash, "version": runtime.model_version,
                      "candidate_only": name == "strict", "expected_sha256": EXPECTED_HASHES[name]},
            "input": {**inventory, **return_audit, "first": str(obs.index.min()),
                      "last": str(obs.index.max())},
            "posterior_entropy": entropy,
            "emission_overlap_jsd": overlap_jsd(runtime, stable_to_internal),
            "transition_and_duration": transition_duration(runtime, stable_to_internal, posterior, starts),
            "filtered_soft_transition_counts": soft_trans.tolist(),
            "economic_profiles": None,
            "periods": analyze_periods(runtime, stable_to_internal, obs, starts, posterior, score),
            "output_contract": {
                "states_csv": records_path.name,
                "probability_columns": [f"p_{state}" for state in STATES],
                "probability_order": list(STATES),
                "timestamp_label": "start time of endpoint 1-minute bar, Asia/Ho_Chi_Minh",
                "event_bar_end": "timestamp_label + one minute",
                "available_at": "blank until provider first-availability is verified",
                "asof_join_ready": False,
            },
        }
        # Reuse the same descriptive profile definition for the whole sample.
        from v1_core import economic_profiles
        report["economic_profiles"] = economic_profiles(obs, posterior, starts)
        if args.shadow_start and name == "strict":
            shadow_records, shadow_report = run_shadow(runtime, stable_to_internal, obs, starts,
                                                       score, args.shadow_start)
            report["online_shadow"] = shadow_report
            if shadow_records is not None:
                shadow_records.to_csv(out / "states_shadow_experimental.csv", index=False,
                                      encoding="utf-8-sig")
        write_json(out / f"diagnostics_{name}.json", report)
        (out / f"REPORT_{name}.md").write_text(render_report(name, report), encoding="utf-8")
        print(f"{name}: {len(obs)} returns; {out / f'REPORT_{name}.md'}")


if __name__ == "__main__":
    main()
