"""Render the legacy co-design comparison from saved per-round results.

No evaluation, model call or remote mutation. Output units: WNS ns, TNS us.
"""
import json
import math
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/sta_codesign_table_20260910"
CASES = ("bp", "bp_multi", "ariane133", "swerv_wrapper")
METHODS = ("codex", "agentic3d")


def collect():
    data = {}
    for case in CASES:
        base = ROOT / ("output/sta_codesign_bp_multi_20260903-v2-exact-sta" if case == "bp_multi"
                       else "output/sta_codesign_new_cases_20260909/" + case)
        row = {}
        references = []
        for method in METHODS:
            source = base / method / "history.json"
            history = json.loads(source.read_text())
            valid = []
            for record in history:
                metrics = record.get("metrics", record.get("outer_result", {}).get("metrics", {}))
                if metrics.get("runs_successfully") != 1 or not metrics.get("combined_score", 0) > 0:
                    continue
                valid.append((record, metrics))
                references.append((metrics["final_global_wns_setup_ns"] / metrics["normalized_wns_cost"],
                                   metrics["final_global_tns_setup_ns"] / metrics["normalized_tns_cost"]))
            record, metrics = max(valid, key=lambda pair: pair[1]["combined_score"])
            values = [metrics["combined_score"], metrics["final_global_wns_setup_ns"],
                      metrics["final_global_tns_setup_ns"] / 1000]
            assert math.isclose(1 / values[0], metrics["normalized_sta_cost"], rel_tol=1e-10)
            row[method] = dict(values=values, best_round=record["round"],
                               valid=len(valid), recorded_attempts=len(history), budget=5,
                               cost=metrics["normalized_sta_cost"], source=str(source.relative_to(ROOT)))
        # Recover exact per-case reference from stored normalized/raw pairs;
        # verify agreement over every valid round and both methods, not just best.
        for wns, tns in references:
            assert math.isclose(wns, references[0][0], rel_tol=1e-10)
            assert math.isclose(tns, references[0][1], rel_tol=1e-10)
        row["reference"] = [1., references[0][0], references[0][1] / 1000]
        for method in METHODS:
            v, ref = row[method]["values"], row["reference"]
            row[method]["improvement_percent"] = [(v[0] / ref[0] - 1) * 100,
                (v[1] - ref[1]) / abs(ref[1]) * 100, (v[2] - ref[2]) / abs(ref[2]) * 100]
        data[case] = row
    return data


def tex_escape(text):
    return text.replace("_", r"\_")


def main():
    data = collect()
    avg = {m: [mean(data[c][m]["improvement_percent"][i] for c in CASES) for i in range(3)]
           for m in METHODS}
    OUT.mkdir(exist_ok=True)
    payload = dict(cases=data, arithmetic_mean_improvement_percent=avg,
        selection="Highest valid final STA fitness within each method/case; all three metrics from that same candidate.",
        reference_provenance="Recovered from raw/normalized metrics; agreement checked across all valid rounds in both methods.",
        scope="Legacy Agentic3D, not knowledge-v2. Pre-CTS placement-estimated RC, exactly two OPT passes; not signoff.",
        caveats=["bp_multi Agentic3D has four recorded valid attempts, not five; table and mean are provisional.",
                 "ariane133 has five recorded attempts per method, four valid each; Codex round1 STA failure, Agentic3D round5 corrupt patch.",
                 "Failed attempts are not replaced and do not contribute to best-candidate metrics.",
                 "Fitness mean improvement is not STA cost reduction. All results are single-run best-of-search, not repeated-trial statistics."])
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    # Match the supplied black/white grouped-header academic table style.
    width, height = 1800, 526
    scale = 2
    im = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(im)
    def font(size=26, bold=False, mono=False):
        return ImageFont.truetype("/System/Library/Fonts/" + ("Courier.ttc" if mono else "Times.ttc"),
                                  size * scale, index=1 if bold else 0)
    def text(x, y, value, size=26, bold=False, center=False, mono=False):
        selected = (ImageFont.truetype("/System/Library/Fonts/Supplemental/STIXGeneral.otf", size*scale)
                    if "↑" in value else font(size, bold, mono))
        draw.text((x * scale, y * scale), value, font=selected, fill="black",
                  anchor="mm" if center else "lm")
    def line(x1, y1, x2, y2, weight=1):
        draw.line((x1*scale, y1*scale, x2*scale, y2*scale), fill="black", width=max(1, int(weight*scale)))
    text(24, 25, "Table 1: Partition–placement co-optimization results after two timing-optimization passes.", 31)
    text(24, 61, "The last row reports the arithmetic mean of per-case improvements relative to Reference. Higher is better.", 29)
    left, first, col = 24, 290, 160
    boundaries = [first + 3*col*i for i in range(4)]
    end = boundaries[-1]
    line(left, 87, end, 87, 1.7)
    text(34, 126, "Chip case", 26)
    headers = ("Reference", "Codex (GPT-5.6)", "Agentic3D (legacy)")
    for group, title in enumerate(headers):
        text(boundaries[group]+1.5*col, 108, title, 27, center=True)
        for j, label in enumerate(("Fitness ↑", "WNS (ns) ↑", "TNS (μs) ↑")):
            text(boundaries[group]+(j+.5)*col, 144, label, 25, center=True)
    line(left, 166, end, 166, 1.2)
    rows = []
    for i, case in enumerate(CASES):
        y = 194 + i*43
        text(34, y, case, 24, mono=True)
        values = data[case]["reference"] + data[case]["codex"]["values"] + data[case]["agentic3d"]["values"]
        improved = [False]*3 + [v>0 for m in METHODS for v in data[case][m]["improvement_percent"]]
        rows.append(values)
        for j, value in enumerate(values):
            text(first+(j+.5)*col, y, f"{value:.3f}", 27, bold=improved[j], center=True)
    line(left, 350, end, 350, 1.2)
    text(34, 378, "Avg. impr. (%)", 26)
    av = [0., 0., 0.] + avg["codex"] + avg["agentic3d"]
    for j, value in enumerate(av):
        label = f"{value:+.3f}" if j>=3 else "0.000"
        text(first+(j+.5)*col, 378, label, 27, bold=j>=3 and value>0, center=True)
    line(left, 404, end, 404, 1.7)
    for x in boundaries[:-1]:
        line(x, 93, x, 160, .8); line(x, 173, x, 343, .8); line(x, 357, x, 397, .8)
    text(24, 432, "Bold: improvement over Reference. Metrics are from the same highest-fitness valid candidate; TNS is shown in μs.", 23)
    text(24, 463, "Budget: 5 final attempts/case/method. bp_multi Agentic3D: 4 valid attempts only (provisional); ariane133: 4/5 valid for both.", 23)
    text(24, 494, "Fitness = 1 / [0.6 × |WNS| / |WNS_ref| + 0.4 × |TNS| / |TNS_ref|]. New knowledge-v2 runs are excluded.", 23)
    im.save(OUT / "sta_codesign_results.png")
    lines = [r"% Requires \usepackage{booktabs,graphicx}", r"\begin{table*}[t]", r"\centering",
        r"\caption{Partition--placement co-optimization results after two timing-optimization passes. The last row reports the arithmetic mean of per-case improvements relative to Reference. Higher fitness, WNS and TNS are better. Bold entries improve over Reference.}",
        r"\label{tab:sta-codesign}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l|rrr|rrr|rrr}", r"\toprule",
        r"Chip case & \multicolumn{3}{c|}{Reference} & \multicolumn{3}{c|}{Codex (GPT-5.6)} & \multicolumn{3}{c}{Agentic3D (legacy)} \\",
        r" & Fitness $\uparrow$ & WNS (ns) $\uparrow$ & TNS ($\mu$s) $\uparrow$ & Fitness $\uparrow$ & WNS (ns) $\uparrow$ & TNS ($\mu$s) $\uparrow$ & Fitness $\uparrow$ & WNS (ns) $\uparrow$ & TNS ($\mu$s) $\uparrow$ \\", r"\midrule"]
    for case, values in zip(CASES, rows):
        bolds = [False]*3 + [v>0 for m in METHODS for v in data[case][m]["improvement_percent"]]
        cells = [r"\texttt{"+tex_escape(case)+"}"]
        for v, bold in zip(values, bolds):
            s=f"{v:.3f}"; cells.append(r"\textbf{"+s+"}" if bold else s)
        lines.append(" & ".join(cells)+r" \\")
    lines.extend([r"\midrule", "Avg. impr. (\\%) & " + " & ".join(
        f"{v:.3f}" if i<3 else r"\textbf{"+f"{v:+.3f}"+"}" for i,v in enumerate(av))+r" \\",
        r"\bottomrule", r"\end{tabular}}", r"\par\vspace{2pt}\footnotesize",
        r"Metrics use the same highest-fitness valid candidate. Budget: five final attempts per case/method. The bp\_multi Agentic3D result uses four valid attempts and is provisional; ariane133 has four valid results out of five attempts for both methods. New knowledge-v2 runs are excluded.", r"\end{table*}"])
    (OUT / "sta_codesign_results.tex").write_text("\n".join(lines)+"\n")
    md = ["# 切分—布局协同优化：旧框架结果", "", "| Case | Ref. Fitness | WNS (ns) | TNS (μs) | Codex Fitness | WNS (ns) | TNS (μs) | Agentic3D Fitness | WNS (ns) | TNS (μs) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, v in zip(CASES, rows): md.append("| "+c+" | "+" | ".join(f"{x:.3f}" for x in v)+" |")
    md.append("| Avg. impr. (%) | "+" | ".join(f"{v:+.3f}" for v in av)+" |")
    md.extend(["", "注：各方法取最高有效 Fitness 对应的同一候选；WNS/TNS 越接近零越好。末行先逐 case 求改善率再取算术平均。Fitness 改善率不是加权 STA 代价降幅。bp_multi Agentic3D 仅四次有效结果，本表及平均值是阶段性比较；ariane133 两种方法均为五次尝试、四次有效。旧框架单次搜索结果，不混入新 knowledge-v2，也不代表多次重复实验均值。", "", "完整精度、最优轮次、有效数和来源见 results.json。"])
    (OUT / "sta_codesign_results.md").write_text("\n".join(md)+"\n")
    print(json.dumps({"average_improvement":avg,"mean_cost_reduction_percent":{m:mean((1-data[c][m]['cost'])*100 for c in CASES) for m in METHODS}},indent=2))
    print(OUT)


if __name__ == "__main__": main()
