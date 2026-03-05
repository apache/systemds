#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------


"""Generate HTML benchmark report with charts and visualizations."""
import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# allow running from project root (python scripts/report.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import read_json, iter_run_dirs, manifest_timestamp, token_stats, ttft_stats


def cost_stats(samples_path: Path) -> Optional[float]:
    if not samples_path.exists():
        return None
    total_cost = 0.0
    found_any = False
    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    extra = obj.get("extra") or {}
                    cost = extra.get("cost_usd")
                    if cost is not None:
                        found_any = True
                        total_cost += float(cost)
                except Exception:
                    continue
    except Exception:
        return None
    # 0.0 for local backends, None if no cost data at all
    return total_cost if found_any else None


def safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def fmt(x: Any) -> str:
    if x is None:
        return "N/A"
    return html.escape(str(x))


def fmt_num(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def fmt_pct(x: Any, digits: int = 1) -> str:
    v = safe_float(x)
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}%"


def fmt_cost(x: Any) -> str:
    v = safe_float(x)
    if v is None:
        return "N/A"
    if v == 0:
        return "$0"
    if v < 0.0001:
        return f"${v:.6f}"
    if v < 0.01:
        return f"${v:.4f}"
    return f"${v:.2f}"


# Tableau 10 palette
BACKEND_COLORS = {
    "openai": "#4E79A7",
    "vllm": "#B07AA1",
    "systemds": "#E15759",
    "vllm (Qwen2.5-3B)": "#956B8E",
    "systemds (Qwen2.5-3B)": "#C94D4F",
}




def generate_grouped_bar_chart_svg(data: Dict[str, Dict[str, float]], title: str,
                                    group_colors: Dict[str, str],
                                    width: int = 600, height: int = 350,
                                    value_suffix: str = "") -> str:
    """Grouped horizontal bar chart as SVG."""
    if not data:
        return ""
    
    categories = list(data.keys())
    groups = set()
    for cat_data in data.values():
        groups.update(cat_data.keys())
    groups = sorted(groups)
    
    max_val = 0
    for cat_data in data.values():
        for v in cat_data.values():
            if v > max_val:
                max_val = v
    if max_val == 0:
        max_val = 1
    
    left_margin = 130
    right_margin = 20
    top_margin = 50
    bottom_margin = 60
    chart_width = width - left_margin - right_margin
    chart_height = height - top_margin - bottom_margin
    
    category_height = chart_height / len(categories) if categories else 1
    bar_height = min(20, (category_height - 10) / len(groups)) if groups else 20
    
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{width//2}" y="25" text-anchor="middle" font-size="14" font-weight="bold">{html.escape(title)}</text>')
    
    for i, category in enumerate(categories):
        cat_y = top_margin + i * category_height
        

        svg.append(f'<text x="{left_margin - 8}" y="{cat_y + category_height//2}" text-anchor="end" font-size="11">{html.escape(category[:18])}</text>')
        
        for j, group in enumerate(groups):
            value = data[category].get(group, 0)
            bar_y = cat_y + j * (bar_height + 2) + 5
            bar_width = (value / max_val) * chart_width if max_val > 0 else 0
            color = group_colors.get(group, "#999")
            
            svg.append(f'<rect x="{left_margin}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="{color}" rx="2"/>')
            
            if value > 0:
                val_text = f"{value:.1f}{value_suffix}" if isinstance(value, float) else f"{value}{value_suffix}"
                svg.append(f'<text x="{left_margin + bar_width + 3}" y="{bar_y + bar_height//2 + 4}" font-size="9">{val_text}</text>')
    
    svg.append('</svg>')
    
   
    legend = ['<div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px; justify-content: center;">']
    for group in groups:
        color = group_colors.get(group, "#999")
        legend.append(f'<div style="display: flex; align-items: center; gap: 5px;">')
        legend.append(f'<div style="width: 14px; height: 14px; background: {color}; border-radius: 3px;"></div>')
        legend.append(f'<span style="font-size: 12px;">{html.escape(group)}</span>')
        legend.append('</div>')
    legend.append('</div>')
    
    return '\n'.join(svg) + '\n' + '\n'.join(legend)


def _backend_model_key(r: Dict[str, Any]) -> str:
    """e.g. 'vllm (Qwen2.5-3B)' or just 'openai'."""
    backend = r.get("backend", "")
    model = r.get("backend_model", "")
    if not model or backend == "openai":
        return backend
    short = model.split("/")[-1]
    for suffix in ["-Instruct-v0.3", "-Instruct", "-Inst"]:
        short = short.replace(suffix, "")
    return f"{backend} ({short})"


def generate_accuracy_comparison_table(rows: List[Dict[str, Any]]) -> str:
    data: Dict[str, Dict[str, Dict[str, Any]]] = {} 
    
    for r in rows:
        workload = r.get("workload", "")
        bm_key = _backend_model_key(r)
        if not workload or not bm_key:
            continue
        
        if workload not in data:
            data[workload] = {}
        
        if bm_key not in data[workload]:
            data[workload][bm_key] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Accuracy Comparison by Workload</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">Percentage of correct answers per workload. Bold = 80%+. Hover a cell to see correct/total count.</p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Workload</th>')
    for b in backends:
        out.append(f'<th>{html.escape(b)}</th>')
    out.append('</tr></thead><tbody>')
    
    for wl in workloads:
        out.append(f'<tr><td><strong>{html.escape(wl)}</strong></td>')
        for b in backends:
            if b in data[wl]:
                acc = data[wl][b].get("accuracy_mean")
                n = int(safe_float(data[wl][b].get("n")) or 0)
                if acc is not None:
                    pct = acc * 100
                    acc_count = data[wl][b].get("accuracy_count", "")
                    tip = f"{acc_count} correct" if acc_count else ""
                    weight = "600" if pct >= 80 else "400"
                    out.append(f'<td style="font-weight: {weight};" title="{tip}">{pct:.0f}%</td>')
                else:
                    out.append('<td style="color:#bbb;">-</td>')
            else:
                out.append('<td style="color:#bbb;">-</td>')
        out.append('</tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_latency_comparison_table(rows: List[Dict[str, Any]]) -> str:
 
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for r in rows:
        workload = r.get("workload", "")
        bm_key = _backend_model_key(r)
        if not workload or not bm_key:
            continue
        if workload not in data:
            data[workload] = {}
        if bm_key not in data[workload]:
            data[workload][bm_key] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Latency Comparison (p50)</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">Median response time per query. Lower is better. p50 = half of all requests completed within this time.</p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Workload</th>')
    for b in backends:
        out.append(f'<th>{html.escape(b)}</th>')
    out.append('</tr></thead><tbody>')
    
    for wl in workloads:
        out.append(f'<tr><td><strong>{html.escape(wl)}</strong></td>')
        for b in backends:
            if b in data[wl]:
                lat = safe_float(data[wl][b].get("lat_p50"))
                if lat is not None:
                    display = f"{lat/1000:.1f}s" if lat >= 1000 else f"{lat:.0f}ms"
                    out.append(f'<td>{display}</td>')
                else:
                    out.append('<td style="color:#bbb;">-</td>')
            else:
                out.append('<td style="color:#bbb;">-</td>')
        out.append('</tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_latency_breakdown_table(rows: List[Dict[str, Any]]) -> str:
    # only include rows with TTFT data
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for r in rows:
        workload = r.get("workload", "")
        bm_key = _backend_model_key(r)
        ttft = r.get("ttft_mean")
        gen = r.get("gen_mean")
        
        if not workload or not bm_key:
            continue
        if ttft is None and gen is None:
            continue
            
        if workload not in data:
            data[workload] = {}
        if bm_key not in data[workload]:
            data[workload][bm_key] = r
    
    if not data:
        return '<p class="muted">No TTFT data available. Enable streaming mode for OpenAI to measure TTFT.</p>'
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Latency Breakdown: Prefill vs Decode</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">TTFT (Time-To-First-Token) = prompt processing. Generation = token decoding. Only available for streaming backends.</p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Workload</th><th>Backend</th><th>TTFT (ms)</th><th>Generation (ms)</th><th>Total (ms)</th><th>TTFT %</th></tr></thead><tbody>')
    
    for wl in workloads:
        for b in backends:
            if b in data[wl]:
                r = data[wl][b]
                ttft = safe_float(r.get("ttft_mean"))
                gen = safe_float(r.get("gen_mean"))
                total = safe_float(r.get("lat_mean"))
                
                def _fms(v):
                    if not v:
                        return '-'
                    return f'{v/1000:.1f}s' if v >= 1000 else f'{v:.0f}ms'
                
                pct_str = f'{(ttft / (ttft + gen)) * 100:.0f}%' if ttft and gen else '-'
                
                out.append(f'<tr><td>{html.escape(wl)}</td><td>{html.escape(b)}</td>')
                out.append(f'<td>{_fms(ttft)}</td><td>{_fms(gen)}</td><td>{_fms(total)}</td>')
                out.append(f'<td>{pct_str}</td></tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_consistency_metrics_table(rows: List[Dict[str, Any]]) -> str:
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for r in rows:
        workload = r.get("workload", "")
        bm_key = _backend_model_key(r)
        if not workload or not bm_key:
            continue
        if workload not in data:
            data[workload] = {}
        if bm_key not in data[workload]:
            data[workload][bm_key] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Consistency Metrics</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">How stable is response time across queries? CV (Coefficient of Variation) = std/mean. Lower = more consistent.</p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Workload</th><th>Backend</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>CV</th></tr></thead><tbody>')
    
    for wl in workloads:
        for b in backends:
            if b in data[wl]:
                r = data[wl][b]
                mean = safe_float(r.get("lat_mean"))
                std = safe_float(r.get("lat_std"))
                lat_min = safe_float(r.get("lat_min"))
                lat_max = safe_float(r.get("lat_max"))
                cv = safe_float(r.get("lat_cv"))
                
                def _fmt_ms(v):
                    if not v:
                        return '-'
                    return f'{v/1000:.1f}s' if v >= 1000 else f'{v:.0f}ms'
                
                cv_str = f'{cv:.0f}%' if cv is not None else '-'
                weight = 'font-weight:600' if cv and cv >= 50 else ''
                
                out.append(f'<tr><td>{html.escape(wl)}</td><td>{html.escape(b)}</td>')
                out.append(f'<td>{_fmt_ms(mean)}</td><td>{_fmt_ms(std)}</td><td>{_fmt_ms(lat_min)}</td><td>{_fmt_ms(lat_max)}</td>')
                out.append(f'<td style="{weight}">{cv_str}</td></tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_cost_efficiency_table(rows: List[Dict[str, Any]]) -> str:
  
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for r in rows:
        workload = r.get("workload", "")
        bm_key = _backend_model_key(r)
        if not workload or not bm_key:
            continue
        if workload not in data:
            data[workload] = {}
    
        if bm_key not in data[workload]:
            data[workload][bm_key] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Cost Efficiency</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">Cost per correct answer. API cost for OpenAI, compute cost (electricity + HW) for local backends. Lower = better value.</p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Workload</th>')
    for b in backends:
        out.append(f'<th>{html.escape(b)}</th>')
    out.append('</tr></thead><tbody>')
    
    for wl in workloads:
        out.append(f'<tr><td><strong>{html.escape(wl)}</strong></td>')
        for b in backends:
            if b in data[wl]:
                r = data[wl][b]
                api_cost = safe_float(r.get("cost")) or 0
                compute_cost = safe_float(r.get("total_compute_cost_usd")) or 0
                total_cost = api_cost if api_cost > 0 else compute_cost
                acc_mean = r.get("accuracy_mean")
                n = safe_float(r.get("n")) or 10
                
                if total_cost and total_cost > 0 and acc_mean is not None and acc_mean > 0:
                    correct_count = int(n * acc_mean)
                    cost_per_correct = total_cost / correct_count if correct_count > 0 else None
                    if cost_per_correct is not None:
                        out.append(f'<td>{fmt_cost(cost_per_correct)}</td>')
                    else:
                        out.append('<td style="color:#bbb;">-</td>')
                else:
                    out.append('<td style="color:#bbb;">-</td>')
            else:
                out.append('<td>-</td>')
        out.append('</tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_cost_analysis_section(rows: List[Dict[str, Any]]) -> str:
    

    openai_costs = []
    local_runs = []
    
    for r in rows:
        backend = r.get("backend", "")
        workload = r.get("workload", "")
        acc = r.get("accuracy_mean")
        n = safe_float(r.get("n")) or 10
        lat = safe_float(r.get("lat_p50"))
        
        row_cost = safe_float(r.get("cost")) or 0
        if backend == "openai" and row_cost > 0:
            openai_costs.append({
                "workload": workload,
                "cost": row_cost,
                "accuracy": acc,
                "n": n,
                "latency": lat,
                "total_tokens": r.get("total_tokens"),
            })
        elif backend in ["vllm", "systemds"]:
            local_runs.append({
                "backend": backend,
                "workload": workload,
                "accuracy": acc,
                "n": n,
                "latency": lat,
                "electricity_cost_usd": r.get("electricity_cost_usd"),
                "hardware_amortization_usd": r.get("hardware_amortization_usd"),
                "total_compute_cost_usd": r.get("total_compute_cost_usd"),
            })
    
    if not openai_costs:
        return ""
    
    out = ['<h2>Cost Analysis: Cloud vs Local Inference</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">OpenAI API costs vs estimated electricity + hardware amortization for local GPU inference.</p>')
    
  
    total_openai_cost = sum(c["cost"] for c in openai_costs)
    avg_cost_per_run = total_openai_cost / len(openai_costs) if openai_costs else 0
    total_queries = sum(c["n"] for c in openai_costs)
    cost_per_query = total_openai_cost / total_queries if total_queries > 0 else 0
    
    out.append('<div class="cost-analysis-grid">')
    

    out.append('''
    <div class="cost-card cloud">
        <h3>Cloud (OpenAI API)</h3>
        <div class="cost-stats">
    ''')
   
    total_tokens = sum(safe_float(c.get("total_tokens", 0)) or 0 for c in openai_costs)
    cost_per_1m_tokens = (total_openai_cost / total_tokens * 1_000_000) if total_tokens > 0 else None
    
    out.append(f'<div class="stat"><span class="label">Total Spent:</span> <span class="value">${total_openai_cost:.4f}</span></div>')
    out.append(f'<div class="stat"><span class="label">Runs with Cost:</span> <span class="value">{len(openai_costs)}</span></div>')
    out.append(f'<div class="stat"><span class="label">Avg Cost/Run:</span> <span class="value">${avg_cost_per_run:.4f}</span></div>')
    out.append(f'<div class="stat"><span class="label">Cost/Query:</span> <span class="value">${cost_per_query:.6f}</span></div>')
    if cost_per_1m_tokens:
        out.append(f'<div class="stat"><span class="label">Cost/1M Tokens:</span> <span class="value">${cost_per_1m_tokens:.2f}</span></div>')
    out.append('''
        </div>
        <div class="pros-cons">
            <div class="pros">+ Highest accuracy</div>
            <div class="pros">+ No hardware needed</div>
            <div class="cons">- Per-query costs</div>
            <div class="cons">- Network latency</div>
        </div>
    </div>
    ''')
    
 
    out.append('''
    <div class="cost-card local">
        <h3>Local Inference</h3>
        <div class="cost-stats">
    ''')
    out.append(f'<div class="stat"><span class="label">API Cost:</span> <span class="value">$0</span></div>')
    local_electricity = 0.0
    local_hw_cost = 0.0
    local_compute_total = 0.0
    for r in local_runs:
        local_electricity += safe_float(r.get("electricity_cost_usd")) or 0.0
        local_hw_cost += safe_float(r.get("hardware_amortization_usd")) or 0.0
        local_compute_total += safe_float(r.get("total_compute_cost_usd")) or 0.0
    if local_compute_total > 0:
        out.append(f'<div class="stat"><span class="label">Electricity:</span> <span class="value">${local_electricity:.4f}</span></div>')
        out.append(f'<div class="stat"><span class="label">HW Amortization:</span> <span class="value">${local_hw_cost:.4f}</span></div>')
        out.append(f'<div class="stat"><span class="label">Total Compute:</span> <span class="value">${local_compute_total:.4f}</span></div>')
    else:
        out.append(f'<div class="stat"><span class="label">Compute Cost:</span> <span class="value">Use --power-draw-w and --hardware-cost flags</span></div>')
    out.append(f'<div class="stat"><span class="label">Local Runs:</span> <span class="value">{len(local_runs)}</span></div>')
    out.append(f'<div class="stat"><span class="label">Backends:</span> <span class="value">{len(set(r["backend"] for r in local_runs))}</span></div>')
    out.append('''
        </div>
        <div class="pros-cons">
            <div class="pros">+  Zero API cost</div>
            <div class="pros">+  Privacy (data stays local)</div>
            <div class="cons">-  Hardware + electricity costs</div>
            <div class="cons">-  Lower accuracy on complex tasks</div>
        </div>
    </div>
    ''')
    
    out.append('</div>')  
    
  
    out.append('<h3>Cost Projection (1,000 queries)</h3>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Backend</th><th>Est. Cost (1000 queries)</th><th>Notes</th></tr></thead>')
    out.append('<tbody>')
    

    projected_1k = cost_per_query * 1000
    out.append(f'<tr><td>OpenAI (API)</td><td>${projected_1k:.2f}</td><td>Based on current usage (API cost)</td></tr>')
    
    local_backend_costs: Dict[str, List[float]] = {}
    for r in local_runs:
        b = r.get("backend", "unknown")
        tc = safe_float(r.get("total_compute_cost_usd")) or 0
        n = safe_float(r.get("n")) or 10
        if tc > 0 and n > 0:
            local_backend_costs.setdefault(b, []).append(tc / n)
    
    for b in sorted(local_backend_costs.keys()):
        per_query_costs = local_backend_costs[b]
        avg_per_query = sum(per_query_costs) / len(per_query_costs)
        proj = avg_per_query * 1000
        out.append(f'<tr><td>{html.escape(b)}</td><td>${proj:.2f}</td><td>Electricity + HW amortization</td></tr>')

    out.append('</tbody></table>')

    out.append('<p class="muted"><small>Note: Projections based on actual measured compute costs per query from benchmark runs '
               '(electricity + hardware amortization via --power-draw-w and --hardware-cost flags).</small></p>')
    
    return '\n'.join(out)



def generate_summary_section(rows: List[Dict[str, Any]]) -> str:

    backends = sorted(set(r.get("backend") for r in rows if r.get("backend")))
    workloads = sorted(set(r.get("workload") for r in rows if r.get("workload")))
    models = sorted(set(str(m) for m in (r.get("backend_model") for r in rows) if m))
    total_runs = len(rows)

    api_costs = [safe_float(r.get("cost")) for r in rows
                 if r.get("backend") == "openai" and safe_float(r.get("cost"))]
    total_api = sum(api_costs) if api_costs else 0
    total_compute = sum(safe_float(r.get("total_compute_cost_usd")) or 0
                        for r in rows if r.get("backend") != "openai")

    latencies = [safe_float(r.get("lat_p50")) for r in rows
                 if safe_float(r.get("lat_p50")) is not None]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    acc_by_wl: Dict[str, List[float]] = {}
    for r in rows:
        wl = r.get("workload", "")
        acc = r.get("accuracy_mean")
        if wl and acc is not None:
            acc_by_wl.setdefault(wl, []).append(acc * 100)

    best_wl = max(acc_by_wl, key=lambda w: sum(acc_by_wl[w])/len(acc_by_wl[w]), default="")
    worst_wl = min(acc_by_wl, key=lambda w: sum(acc_by_wl[w])/len(acc_by_wl[w]), default="")
    best_pct = sum(acc_by_wl[best_wl])/len(acc_by_wl[best_wl]) if best_wl else 0
    worst_pct = sum(acc_by_wl[worst_wl])/len(acc_by_wl[worst_wl]) if worst_wl else 0

    def _fmt_lat(ms):
        return f"{ms/1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"

    out = ['''
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px;">
    ''']

    cards = [
        ("Runs", str(total_runs), f"{len(workloads)} workloads, {len(backends)} backends"),
        ("Avg Latency", _fmt_lat(avg_lat), f"across all {total_runs} runs"),
        ("Best Accuracy", f"{best_pct:.0f}%", best_wl),
        ("Total Cost", f"${total_api + total_compute:.2f}", f"${total_api:.2f} API + ${total_compute:.2f} compute"),
    ]

    for title, value, subtitle in cards:
        out.append(f'''
        <div style="background: white; border-radius: 10px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
            <div style="font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">{title}</div>
            <div style="font-size: 28px; font-weight: 700; color: #1a1a2e;">{value}</div>
            <div style="font-size: 12px; color: #999; margin-top: 4px;">{subtitle}</div>
        </div>
        ''')

    out.append('</div>')

    out.append(f'''
    <div style="background: #f8f9fa; border-radius: 8px; padding: 14px 18px; margin-bottom: 28px; font-size: 13px; color: #555; line-height: 1.7;">
        <b>Models:</b> {", ".join(models)}<br>
        <b>Backends:</b> {", ".join(backends)}<br>
        <b>Workloads:</b> {", ".join(workloads)}
        &nbsp;&mdash;&nbsp; easiest: <b>{best_wl} ({best_pct:.0f}%)</b>,
        hardest: <b>{worst_wl} ({worst_pct:.0f}%)</b>
    </div>
    ''')

    return '\n'.join(out)


def generate_summary_cards(rows: List[Dict[str, Any]]) -> str:
    return generate_summary_section(rows)


def generate_backend_overview_table(rows: List[Dict[str, Any]]) -> str:
    """Compact one-row-per-backend table: avg accuracy, avg latency, total cost."""
    backends: Dict[str, Dict[str, list]] = {}
    for r in rows:
        bm = _backend_model_key(r)
        if not bm:
            continue
        backends.setdefault(bm, {"acc": [], "lat": [], "cost": 0.0, "workloads": set()})
        acc = r.get("accuracy_mean")
        lat = safe_float(r.get("lat_p50"))
        if acc is not None:
            backends[bm]["acc"].append(acc)
        if lat is not None:
            backends[bm]["lat"].append(lat)
        api = safe_float(r.get("cost")) or 0
        compute = safe_float(r.get("total_compute_cost_usd")) or 0
        backends[bm]["cost"] += api if api > 0 else compute
        wl = r.get("workload", "")
        if wl:
            backends[bm]["workloads"].add(wl)

    if not backends:
        return ""

    out = ['<h2>Backend Overview</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">One row per backend. Averages across all workloads. Quick comparison for presentations.</p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Backend</th><th>Workloads</th><th>Avg Accuracy</th><th>Avg Latency (p50)</th><th>Total Cost</th><th>Verdict</th></tr></thead><tbody>')

    best_acc_key = max(backends, key=lambda k: (sum(backends[k]["acc"]) / len(backends[k]["acc"])) if backends[k]["acc"] else 0)
    best_lat_key = min(backends, key=lambda k: (sum(backends[k]["lat"]) / len(backends[k]["lat"])) if backends[k]["lat"] else float('inf'))
    best_cost_key = min(backends, key=lambda k: backends[k]["cost"] if backends[k]["cost"] > 0 else float('inf'))

    for bm in sorted(backends.keys()):
        d = backends[bm]
        avg_acc = (sum(d["acc"]) / len(d["acc"]) * 100) if d["acc"] else 0
        avg_lat = sum(d["lat"]) / len(d["lat"]) if d["lat"] else 0
        total_cost = d["cost"]
        n_wl = len(d["workloads"])

        if avg_lat >= 1000:
            lat_str = f"{avg_lat / 1000:.1f}s"
        else:
            lat_str = f"{avg_lat:.0f}ms"

        badges = []
        if bm == best_acc_key:
            badges.append("Best accuracy")
        if bm == best_lat_key:
            badges.append("Fastest")
        if bm == best_cost_key:
            badges.append("Cheapest")
        verdict = ", ".join(badges) if badges else "-"

        color = BACKEND_COLORS.get(bm, BACKEND_COLORS.get(bm.split(" (")[0], "#666"))
        out.append(f'<tr>')
        out.append(f'<td><strong style="color:{color};">{html.escape(bm)}</strong></td>')
        out.append(f'<td>{n_wl}</td>')
        out.append(f'<td>{"<strong>" if bm == best_acc_key else ""}{avg_acc:.1f}%{"</strong>" if bm == best_acc_key else ""}</td>')
        out.append(f'<td>{"<strong>" if bm == best_lat_key else ""}{lat_str}{"</strong>" if bm == best_lat_key else ""}</td>')
        out.append(f'<td>{fmt_cost(total_cost)}</td>')
        out.append(f'<td style="font-size:12px;">{verdict}</td>')
        out.append(f'</tr>')

    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_systemds_vs_vllm_summary(rows: List[Dict[str, Any]]) -> str:
    """Compact SystemDS vs vLLM summary table -- one row per model."""
    by_model: Dict[str, Dict[str, Dict[str, list]]] = {}  # model -> backend -> metrics
    for r in rows:
        backend = r.get("backend", "")
        model = r.get("backend_model", "")
        if backend not in ("vllm", "systemds") or not model:
            continue
        short = model.split("/")[-1]
        for s in ["-Instruct-v0.3", "-Instruct"]:
            short = short.replace(s, "")
        by_model.setdefault(short, {}).setdefault(backend, {"acc": [], "lat": [], "wl": 0})
        acc = r.get("accuracy_mean")
        lat = safe_float(r.get("lat_p50"))
        if acc is not None:
            by_model[short][backend]["acc"].append(acc)
        if lat is not None:
            by_model[short][backend]["lat"].append(lat)
        by_model[short][backend]["wl"] += 1

    if not by_model:
        return ""

    out = ['<h2>SystemDS vs vLLM -- Summary</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">Condensed comparison for presentations. Same model + GPU, averaged across all workloads.</p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Model</th><th>Metric</th><th>vLLM</th><th>SystemDS JMLC</th><th>Delta</th></tr></thead><tbody>')

    for model_name in sorted(by_model.keys()):
        combos = by_model[model_name]
        v = combos.get("vllm", {"acc": [], "lat": []})
        s = combos.get("systemds", {"acc": [], "lat": []})

        v_acc = (sum(v["acc"]) / len(v["acc"]) * 100) if v["acc"] else 0
        s_acc = (sum(s["acc"]) / len(s["acc"]) * 100) if s["acc"] else 0
        v_lat = sum(v["lat"]) / len(v["lat"]) if v["lat"] else 0
        s_lat = sum(s["lat"]) / len(s["lat"]) if s["lat"] else 0

        acc_delta = s_acc - v_acc
        acc_delta_str = f"+{acc_delta:.1f}pp" if acc_delta >= 0 else f"{acc_delta:.1f}pp"
        lat_overhead = s_lat / v_lat if v_lat > 0 else 0
        lat_str = f"{lat_overhead:.1f}x slower" if lat_overhead > 1 else "faster"

        def fmt_lat(ms):
            return f"{ms/1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"

        # Accuracy row
        out.append(f'<tr>')
        out.append(f'<td rowspan="2"><strong>{html.escape(model_name)}</strong></td>')
        out.append(f'<td>Avg Accuracy</td>')
        out.append(f'<td>{v_acc:.1f}%</td>')
        out.append(f'<td>{s_acc:.1f}%</td>')
        color = "#59A14F" if acc_delta >= 0 else "#E15759"
        out.append(f'<td style="color:{color}; font-weight:600;">{acc_delta_str}</td>')
        out.append(f'</tr>')

        # Latency row
        out.append(f'<tr>')
        out.append(f'<td>Avg Latency (p50)</td>')
        out.append(f'<td>{fmt_lat(v_lat)}</td>')
        out.append(f'<td>{fmt_lat(s_lat)}</td>')
        out.append(f'<td style="color:#E15759; font-weight:600;">{lat_str}</td>')
        out.append(f'</tr>')

    out.append('</tbody></table>')

    out.append('<p style="color:#888; font-size:12px; margin-top:8px;">pp = percentage points. Latency overhead reflects the JMLC overhead. Accuracy deltas show SystemDS matches or slightly improves on reasoning/summarization tasks.</p>')

    return '\n'.join(out)


def generate_cost_tradeoff_table(rows: List[Dict[str, Any]]) -> str:
    """Tiny cost-accuracy tradeoff table for presentations."""
    cloud_cost = 0.0
    cloud_acc = []
    local_cost = 0.0
    local_acc = []
    local_runs = 0
    cloud_runs = 0
    cloud_queries = 0
    local_queries = 0

    for r in rows:
        backend = r.get("backend", "")
        acc = r.get("accuracy_mean")
        api = safe_float(r.get("cost")) or 0
        compute = safe_float(r.get("total_compute_cost_usd")) or 0
        n = int(safe_float(r.get("n")) or 0)

        if backend == "openai":
            cloud_cost += api
            cloud_runs += 1
            cloud_queries += n
            if acc is not None:
                cloud_acc.append(acc)
        elif backend in ("vllm", "systemds"):
            local_cost += compute
            local_runs += 1
            local_queries += n
            if acc is not None:
                local_acc.append(acc)

    if not cloud_acc and not local_acc:
        return ""

    cloud_avg = (sum(cloud_acc) / len(cloud_acc) * 100) if cloud_acc else 0
    local_avg = (sum(local_acc) / len(local_acc) * 100) if local_acc else 0

    cloud_per_q = (cloud_cost / cloud_queries) if cloud_queries else 0
    local_per_q = (local_cost / local_queries) if local_queries else 0

    out = ['<h2>Cost vs Accuracy Tradeoff</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">Cloud API vs local GPU inference. Key tradeoff for deployment decisions.</p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th></th><th>Cloud (OpenAI API)</th><th>Local GPU (vLLM + SystemDS)</th></tr></thead><tbody>')

    out.append(f'<tr><td><strong>Avg Accuracy</strong></td>')
    out.append(f'<td><strong>{cloud_avg:.1f}%</strong></td>')
    out.append(f'<td>{local_avg:.1f}%</td></tr>')

    out.append(f'<tr><td><strong>Total Cost ({cloud_runs + local_runs} runs)</strong></td>')
    out.append(f'<td>{fmt_cost(cloud_cost)}</td>')
    out.append(f'<td>{fmt_cost(local_cost)}</td></tr>')

    out.append(f'<tr><td><strong>Avg Cost / Query</strong></td>')
    out.append(f'<td>{fmt_cost(cloud_per_q)}</td>')
    out.append(f'<td>{fmt_cost(local_per_q)}</td></tr>')

    out.append(f'<tr><td><strong>Projected Cost (1K queries)</strong></td>')
    out.append(f'<td>{fmt_cost(cloud_per_q * 1000)}</td>')
    out.append(f'<td>{fmt_cost(local_per_q * 1000)}</td></tr>')

    out.append(f'<tr><td><strong>Advantage</strong></td>')
    out.append(f'<td style="font-size:12px;">Higher accuracy, zero setup</td>')
    out.append(f'<td style="font-size:12px;">Privacy, lower marginal cost</td></tr>')

    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_charts_section(rows: List[Dict[str, Any]]) -> str:
    """Generate a single throughput chart (accuracy/latency are already in comparison tables)."""
    latest: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        wl = r.get("workload", "")
        be = _backend_model_key(r)
        if not wl or not be:
            continue
        latest.setdefault(wl, {})
        if be not in latest[wl]:
            latest[wl][be] = r

    throughput_data: Dict[str, Dict[str, float]] = {}
    for wl, backends in latest.items():
        throughput_data[wl] = {}
        for be, r in backends.items():
            thr = safe_float(r.get("thr"))
            if thr is not None:
                throughput_data[wl][be] = thr

    if not throughput_data:
        return ""

    out = ['<h2>Throughput</h2>']
    out.append('<p style="color:#888; font-size:13px; margin-top:-8px;">Requests per second. Higher is better. Measures end-to-end query processing speed.</p>')
    out.append('<div class="charts-grid">')
    out.append('<div class="chart-container">')
    out.append(generate_grouped_bar_chart_svg(
        throughput_data, "Throughput by Workload (req/s)",
        BACKEND_COLORS, value_suffix=" req/s"
    ))
    out.append('</div>')
    out.append('</div>')
    return '\n'.join(out)


def generate_head_to_head_section(rows: List[Dict[str, Any]]) -> str:
    """Generate minimal head-to-head comparison: vLLM vs SystemDS JMLC."""

    by_model: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}
    for r in rows:
        backend = r.get("backend", "")
        model = r.get("backend_model", "")
        wl = r.get("workload", "")
        if backend not in ("vllm", "systemds") or not model or not wl:
            continue
        short = model.split("/")[-1]
        for s in ["-Instruct-v0.3", "-Instruct"]:
            short = short.replace(s, "")
        by_model.setdefault(short, {})[(wl, backend)] = r

    if not by_model:
        return ""

    out = []
    out.append('''
    <div style="margin: 32px 0;">
    <h2 style="margin-bottom: 4px;">Framework Comparison: vLLM vs SystemDS JMLC</h2>
    <p style="color: #666; margin-top: 0; font-size: 14px;">
        Same model, same NVIDIA H100 GPU, same prompts.
        Compares native llmPredict built-in overhead vs direct vLLM.
    </p>
    ''')

    for model_name in sorted(by_model.keys()):
        combos = by_model[model_name]
        workloads = sorted(set(wl for wl, _ in combos.keys()))

        overheads = []
        for wl in workloads:
            vr = combos.get((wl, "vllm"))
            sr = combos.get((wl, "systemds"))
            if vr and sr:
                vl = safe_float(vr.get("lat_p50")) or 0
                sl = safe_float(sr.get("lat_p50")) or 0
                if vl > 0:
                    overheads.append(sl / vl)
        avg_overhead = sum(overheads) / len(overheads) if overheads else 0

        max_lat = 1
        for wl in workloads:
            for be in ("vllm", "systemds"):
                r = combos.get((wl, be))
                if r:
                    v = safe_float(r.get("lat_p50")) or 0
                    if v > max_lat:
                        max_lat = v

        out.append(f'''
        <div style="background: #f8f9fa; border-radius: 10px; padding: 24px; margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px;">
            <h3 style="margin: 0; font-size: 17px;">{html.escape(model_name)}</h3>
            <span style="font-size: 24px; font-weight: 700; color: #444;">{avg_overhead:.1f}x
                <span style="font-size: 12px; font-weight: 400; color: #999;">avg overhead</span>
            </span>
        </div>
        ''')

        out.append('''
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead>
            <tr style="border-bottom: 1px solid #dee2e6; text-align: left;">
                <th style="padding: 8px 12px; width: 130px; font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.3px;">Workload</th>
                <th style="padding: 8px 12px; font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.3px;">Latency (p50)</th>
                <th style="padding: 8px 6px; width: 70px; text-align: right; font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.3px;">Overhead</th>
                <th style="padding: 8px 6px; width: 110px; text-align: center; font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.3px;">Accuracy</th>
            </tr>
        </thead>
        <tbody>
        ''')

        for wl in workloads:
            vr = combos.get((wl, "vllm"))
            sr = combos.get((wl, "systemds"))
            vl = safe_float(vr.get("lat_p50")) if vr else 0
            sl = safe_float(sr.get("lat_p50")) if sr else 0
            va = (vr.get("accuracy_mean") or 0) * 100 if vr else 0
            sa = (sr.get("accuracy_mean") or 0) * 100 if sr else 0

            def _fmt_lat(ms):
                if not ms:
                    return "-"
                return f"{ms/1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"

            ratio = sl / vl if vl > 0 else 0

            vl_pct = (vl / max_lat) * 100 if max_lat else 0
            sl_pct = (sl / max_lat) * 100 if max_lat else 0

            acc_html = f'{va:.0f}% vs {sa:.0f}%'

            out.append(f'''
            <tr style="border-bottom: 1px solid #f0f0f0;">
                <td style="padding: 10px 12px; font-weight: 600;">{html.escape(wl)}</td>
                <td style="padding: 10px 12px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="width: 55px; font-size: 11px; color: #4E79A7; font-weight: 600;">vLLM</span>
                        <div style="flex: 1; background: #e8eef4; border-radius: 3px; height: 12px;">
                            <div style="width: {vl_pct:.1f}%; background: #4E79A7; border-radius: 3px; height: 12px;"></div>
                        </div>
                        <span style="width: 55px; font-size: 12px; text-align: right; color: #555;">{_fmt_lat(vl)}</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="width: 55px; font-size: 11px; color: #E15759; font-weight: 600;">SystemDS</span>
                        <div style="flex: 1; background: #fae8e8; border-radius: 3px; height: 12px;">
                            <div style="width: {sl_pct:.1f}%; background: #E15759; border-radius: 3px; height: 12px;"></div>
                        </div>
                        <span style="width: 55px; font-size: 12px; text-align: right; color: #555;">{_fmt_lat(sl)}</span>
                    </div>
                </td>
                <td style="padding: 10px 6px; text-align: right; font-size: 16px; font-weight: 700; color: #444;">{ratio:.1f}x</td>
                <td style="padding: 10px 6px; text-align: center; font-size: 12px; color: #666;">{acc_html}</td>
            </tr>
            ''')

        out.append('</tbody></table>')
        out.append('</div>')  # card

    out.append('''
    <p style="color: #999; font-size: 12px; margin-top: 8px;">
        <b>Overhead</b> = SystemDS latency / vLLM latency. Same model produces same accuracy;
        small differences are from non-deterministic generation.
        The overhead measures the overhead that the JMLC + llmPredict pipeline adds
        in exchange for Java ecosystem integration.
    </p>
    </div>
    ''')

    return '\n'.join(out)


def fmt_cost_if_real(r: Dict[str, Any]) -> str:
    api_cost = safe_float(r.get("cost")) or 0
    if api_cost > 0:
        return fmt_cost(api_cost)
    return "$0"

def fmt_cost_per_1m_if_real(r: Dict[str, Any]) -> str:
    cost = r.get("cost_per_1m_tokens")
    backend = r.get("backend", "")
    if backend == "openai" and cost is not None:
        return fmt_cost(cost)
    return "-"

def fmt_compute_cost(r: Dict[str, Any]) -> str:
    tc = safe_float(r.get("total_compute_cost_usd"))
    if tc and tc > 0:
        return f"${tc:.4f}"
    return "-"


FULL_TABLE_COLUMNS = [
    ("backend", "Backend", lambda r: html.escape(r.get("backend", ""))),
    ("backend_model", "Model", lambda r: html.escape(str(r.get("backend_model", "")).split("/")[-1][:25])),
    ("workload", "Workload", lambda r: html.escape(r.get("workload", ""))),
    ("n", "n", lambda r: fmt(r.get("n"))),
    ("accuracy", "Accuracy", lambda r: f'{r.get("accuracy_mean", 0)*100:.1f}% ({r.get("accuracy_count", "")})' if r.get("accuracy_mean") is not None else "N/A"),
    ("rougeL_f", "ROUGE-L", lambda r: f'{r.get("rougeL_f")*100:.1f}%' if r.get("rougeL_f") is not None else ""),
    ("cost", "API Cost ($)", fmt_cost_if_real),
    ("compute_cost", "Compute ($)", fmt_compute_cost),
    ("lat_p50", "Latency p50 (ms)", lambda r: fmt_num(r.get("lat_p50"), 1)),
    ("lat_p95", "Latency p95 (ms)", lambda r: fmt_num(r.get("lat_p95"), 1)),
    ("ttft_mean", "TTFT (ms)", lambda r: fmt_num(r.get("ttft_mean"), 1)),
    ("thr", "Throughput (req/s)", lambda r: fmt_num(r.get("thr"), 2)),
    ("total_tokens", "Tokens", lambda r: fmt(r.get("total_tokens"))),
    ("toks_out", "tok/s (out)", lambda r: fmt_num(r.get("toks_out"), 1)),
]


def generate_full_table(title: str, table_rows: List[Dict[str, Any]], table_id: str = "", is_h3: bool = False) -> str:
    tag = "h3" if is_h3 else "h2"
    out = [f'<div class="table-header">']
    out.append(f'<{tag}>{html.escape(title)}</{tag}>')
    out.append(f'<div>')
    out.append(f'<button class="btn-small" onclick="printSection(\'{table_id}\')">Print</button>')
    out.append(f'<button class="btn-small" onclick="exportTableToCSV(\'{table_id}\', \'{table_id}.csv\')">CSV</button>')
    out.append(f'<button class="btn-small" onclick="copyTableToClipboard(\'{table_id}\')">Copy</button>')
    out.append(f'</div></div>')
    out.append(f'<div class="table-wrapper" id="{table_id}">')
    out.append('<table class="full-table">')
    out.append('<thead><tr>')
    for _, label, _ in FULL_TABLE_COLUMNS:
        out.append(f'<th>{html.escape(label)}</th>')
    out.append('</tr></thead><tbody>')
    
    for r in table_rows:
        out.append('<tr>')
        for _, _, render_fn in FULL_TABLE_COLUMNS:
            out.append(f'<td>{render_fn(r)}</td>')
        out.append('</tr>')
    
    out.append('</tbody></table></div>')
    return '\n'.join(out)


def generate_workload_tables(rows: List[Dict[str, Any]]) -> str:
    
    by_workload: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        wl = r.get("workload", "unknown")
        if wl not in by_workload:
            by_workload[wl] = []
        by_workload[wl].append(r)
    
    out = ['<h2>Performance by Workload Category</h2>']
    
    for wl in sorted(by_workload.keys()):
        wl_rows = by_workload[wl]
        table_id = f"workload-{wl.replace('_', '-')}"
        out.append(generate_full_table(
            wl.replace("_", " ").title(), 
            wl_rows, 
            table_id,
            is_h3=True
        ))
    
    return '\n'.join(out)


def generate_per_sample_results(results_dir: Path) -> str:
    """Generate expandable per-sample results for debugging."""
    run_dirs = iter_run_dirs(results_dir)
    
    out = ['<h2>Per-Sample Results (Debug)</h2>']
    out.append('<p class="muted">Click to expand individual predictions for each run.</p>')
    
    for run_dir in sorted(run_dirs, key=lambda x: x.name):
        samples_path = run_dir / "samples.jsonl"
        if not samples_path.exists():
            continue
        
        run_name = run_dir.name
        samples = []
        
        try:
            with open(samples_path, 'r') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        except Exception:
            continue
        
        if not samples:
            continue
        
        
        correct = sum(1 for s in samples if s.get("correct", False))
        total = len(samples)
        
        out.append(f'''
        <details class="sample-details">
            <summary>
                <strong>{html.escape(run_name)}</strong>
                <span class="sample-count">{correct}/{total} correct</span>
            </summary>
            <div class="sample-list">
        ''')
        
        for i, s in enumerate(samples[:20]):  # Limit to first 20 samples
            sid = s.get("id", s.get("sid", f"sample-{i}"))
            prediction = s.get("prediction", "")[:200]  # Truncate
            reference = s.get("reference", "")[:100]
            is_correct = s.get("correct", None)
            
            status_class = "correct" if is_correct else "incorrect" if is_correct is False else "unknown"
            status_icon = "✓" if is_correct else "✗" if is_correct is False else "?"
            
            out.append(f'''
                <div class="sample-item {status_class}">
                    <div class="sample-header">
                        <span class="status-icon">{status_icon}</span>
                        <span class="sample-id">{html.escape(str(sid))}</span>
                    </div>
                    <div class="sample-content">
                        <div class="prediction"><strong>Pred:</strong> {html.escape(prediction)}...</div>
                        <div class="reference"><strong>Ref:</strong> {html.escape(str(reference))}</div>
                    </div>
                </div>
            ''')
        
        if len(samples) > 20:
            out.append(f'<div class="muted">... and {len(samples) - 20} more samples</div>')
        
        out.append('</div></details>')
    
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate HTML benchmark report with charts.")
    ap.add_argument("--results-dir", default="results", help="Directory containing run folders")
    ap.add_argument("--out", default="report.html", help="Output HTML path")
    ap.add_argument("--latest", type=int, default=20, help="How many latest runs to show")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = iter_run_dirs(results_dir)
    
    if not run_dirs:
        print(f"Error: no valid run directories found under {results_dir}/", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            metrics = read_json(run_dir / "metrics.json")
            cfg = read_json(run_dir / "run_config.json")
            ts = manifest_timestamp(run_dir)
            total, avg, total_in, total_out = token_stats(run_dir / "samples.jsonl")
            cost = cost_stats(run_dir / "samples.jsonl")
            ttft_mean, gen_mean = ttft_stats(run_dir / "samples.jsonl")
            
            
            lat_mean = safe_float(metrics.get("latency_ms_mean"))
            lat_std = safe_float(metrics.get("latency_ms_std"))
            lat_cv = (lat_std / lat_mean * 100) if lat_mean and lat_std else None
            
            
            n = safe_float(metrics.get("n")) or 1
            total_time_s = (lat_mean * n / 1000) if lat_mean else None
            toks_total = (total / total_time_s) if total and total_time_s else None
            toks_out = (total_out / total_time_s) if total_out and total_time_s else None
            ms_per_tok_total = (1000 / toks_total) if toks_total else None
            ms_per_tok_out = (1000 / toks_out) if toks_out else None
            
        
            cost_per_1m = (cost / total * 1_000_000) if cost and total else None

            workload_base = cfg.get("workload", "")
            run_name = run_dir.name
            
            dataset_source = ""
            known_sources = ["gsm8k", "boolq", "xsum", "cnn", "logiqa", "ner", "json_struct", "stsb"]
            for src in known_sources:
                if f"_{src}" in run_name.lower():
                    dataset_source = src
                    break
            
            workload_with_source = f"{workload_base} ({dataset_source})" if dataset_source else workload_base
            
            rows.append({
                "run_dir": run_dir.name,
                "ts": ts,
                "backend": cfg.get("backend", ""),
                "backend_model": cfg.get("backend_model", ""),
                "workload": workload_base,  
                "workload_full": workload_with_source,  
                "n": metrics.get("n", ""),
                "lat_mean": metrics.get("latency_ms_mean"),
                "lat_p50": metrics.get("latency_ms_p50"),
                "lat_p95": metrics.get("latency_ms_p95"),
                "lat_std": lat_std,
                "lat_cv": lat_cv,
                "lat_min": metrics.get("latency_ms_min"),
                "lat_max": metrics.get("latency_ms_max"),
                "thr": metrics.get("throughput_req_per_s"),
                "accuracy_mean": metrics.get("accuracy_mean"),
                "accuracy_count": metrics.get("accuracy_count", ""),
                "rouge1_f": metrics.get("avg_rouge1_f"),
                "rouge2_f": metrics.get("avg_rouge2_f"),
                "rougeL_f": metrics.get("avg_rougeL_f"),
                "concurrency": metrics.get("concurrency"),
                "total_tokens": total,
                "avg_tokens": avg,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "cost": cost,
                "cost_per_1m_tokens": cost_per_1m,
                "electricity_cost_usd": metrics.get("electricity_cost_usd"),
                "hardware_amortization_usd": metrics.get("hardware_amortization_usd"),
                "total_compute_cost_usd": metrics.get("total_compute_cost_usd"),
                "mem_peak": metrics.get("memory_mb_peak"),
                "cpu_avg": metrics.get("cpu_percent_avg"),
                "ttft_mean": ttft_mean or metrics.get("ttft_ms_mean"),
                "gen_mean": gen_mean or metrics.get("generation_ms_mean"),
                "toks_total": toks_total,
                "toks_out": toks_out,
                "ms_per_tok_total": ms_per_tok_total,
                "ms_per_tok_out": ms_per_tok_out,
            })
        except Exception as e:
            print(f"Warning: skipping {run_dir.name}: {e}", file=sys.stderr)

    rows_sorted = sorted(rows, key=lambda r: r.get("ts", "") or "0000", reverse=True)
    latest_rows = rows_sorted[:args.latest]

    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>systemds-bench-gpt Benchmark Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
        margin: 0; padding: 24px; 
        background: #f8f9fa;
        color: #333;
    }}
    .container {{ max-width: 100%; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px 0; color: #1a1a2e; }}
    h2 {{ margin: 36px 0 12px 0; color: #1a1a2e; border-bottom: 1px solid #e8e8e8; padding-bottom: 8px; font-size: 20px; }}
    h3 {{ margin: 20px 0 10px 0; color: #333; }}
    .meta {{ color: #666; margin-bottom: 24px; font-size: 14px; }}
    
    
    @media (max-width: 900px) {{
        div[style*="grid-template-columns: repeat(4"] {{
            grid-template-columns: repeat(2, 1fr) !important;
        }}
    }}
    @media (max-width: 500px) {{
        div[style*="grid-template-columns: repeat(4"] {{
            grid-template-columns: 1fr !important;
        }}
    }}
    
    .charts-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
        gap: 24px;
        margin-bottom: 30px;
    }}
    .chart-container {{
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .comparison-table {{
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 24px;
        font-size: 13px;
    }}
    .comparison-table th, .comparison-table td {{
        padding: 10px 14px;
        text-align: center;
        border-bottom: 1px solid #f0f0f0;
    }}
    .comparison-table th {{
        background: #fafbfc;
        font-weight: 600;
        color: #555;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}
    .comparison-table td:first-child {{
        text-align: left;
    }}
    .comparison-table tbody tr:hover {{
        background: #f8f9fa;
    }}
    
    /* Cost Analysis Section */
    .cost-analysis-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
        margin-bottom: 24px;
    }}
    .cost-card {{
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-left: 3px solid #dee2e6;
    }}
    .cost-card h3 {{
        margin: 0 0 14px 0;
        font-size: 15px;
        color: #333;
    }}
    .cost-stats {{
        margin-bottom: 12px;
    }}
    .cost-stats .stat {{
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 1px solid #f5f5f5;
        font-size: 13px;
    }}
    .cost-stats .label {{
        color: #888;
    }}
    .cost-stats .value {{
        font-weight: 600;
        color: #333;
    }}
    .pros-cons {{
        font-size: 12px;
        color: #888;
        margin-top: 8px;
    }}
    .pros {{ margin: 3px 0; }}
    .cons {{ margin: 3px 0; }}
    
    @media (max-width: 768px) {{
        .cost-analysis-grid {{ grid-template-columns: 1fr; }}
    }}
    
    /* Full table with all columns - compact */
    .table-wrapper {{
        overflow-x: auto;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 24px;
    }}
    .table-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }}
    .table-header h2, .table-header h3 {{
        margin: 0;
    }}
    .btn-small {{
        padding: 5px 10px;
        background: #e9ecef;
        color: #555;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        cursor: pointer;
        font-size: 10px;
        margin-left: 6px;
    }}
    .btn-small:hover {{ background: #dee2e6; }}
    .full-table {{ 
        border-collapse: collapse; 
        width: max-content;
        min-width: 100%;
        font-size: 9px;
    }}
    .full-table th, .full-table td {{ 
        padding: 4px 6px; 
        text-align: left; 
        border: 1px solid #ddd;
        white-space: nowrap;
    }}
    .full-table th {{ 
        background: #f0f0f0; 
        font-weight: 600;
        color: #1a1a2e;
        position: sticky;
        top: 0;
        font-size: 8px;
    }}
    .full-table tr:nth-child(even) {{ background: #fafafa; }}
    .full-table tr:hover {{ background: #f5f5f5; }}
    
    code {{ 
        background: #f1f3f4; 
        padding: 2px 4px; 
        border-radius: 3px; 
        font-size: 10px;
    }}
    
    .btn {{
        padding: 6px 14px;
        background: #e9ecef;
        color: #555;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        cursor: pointer;
        font-size: 12px;
    }}
    .btn:hover {{ background: #dee2e6; }}
    
    @media print {{
        div[style*="display: flex; gap: 8px"] {{ display: none !important; }}
        body {{ 
            padding: 10px; 
            background: white;
            font-size: 9px;
        }}
        .charts-grid, .chart-container {{ 
            break-inside: avoid; 
        }}
        .table-wrapper {{
            overflow: visible;
            box-shadow: none;
        }}
        .full-table {{
            font-size: 8px;
        }}
        .full-table th, .full-table td {{
            padding: 3px 4px;
        }}
        h2 {{ 
            break-before: page;
            margin-top: 10px;
        }}
    }}
    
    @page {{
        size: landscape;
        margin: 0.5cm;
    }}
    
    @media (max-width: 768px) {{
        .charts-grid {{ grid-template-columns: 1fr; }}
    }}
    
    /* Per-Sample Results */
    .sample-details {{
        margin: 8px 0;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        background: #fafafa;
    }}
    .sample-details summary {{
        padding: 10px 15px;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 13px;
    }}
    .sample-details summary:hover {{
        background: #f0f0f0;
    }}
    .sample-count {{
        background: #e0e0e0;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
    }}
    .sample-list {{
        padding: 10px;
        max-height: 400px;
        overflow-y: auto;
    }}
    .sample-item {{
        margin: 5px 0;
        padding: 8px;
        border-radius: 4px;
        font-size: 11px;
        border-left: 3px solid #ccc;
    }}
    .sample-item.correct {{
        background: #f2f7f1;
        border-left-color: #59A14F;
    }}
    .sample-item.incorrect {{
        background: #fdf2f2;
        border-left-color: #E15759;
    }}
    .sample-item.unknown {{
        background: #fef8ef;
        border-left-color: #F28E2B;
    }}
    .sample-header {{
        display: flex;
        gap: 8px;
        margin-bottom: 4px;
    }}
    .status-icon {{
        font-weight: bold;
    }}
    .sample-id {{
        color: #666;
    }}
    .sample-content {{
        font-family: monospace;
        font-size: 10px;
        color: #444;
    }}
    .prediction, .reference {{
        margin: 2px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .muted {{
        color: #888;
        font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1 style="margin-bottom: 4px;">LLM Benchmark Report</h1>
    <p style="color: #666; font-size: 14px; margin: 0 0 4px 0;">
        Compares LLM inference backends (OpenAI API, vLLM, SystemDS JMLC)
        across accuracy, latency, throughput, and cost.
    </p>
    <div class="meta">Generated: {gen_ts} | {len(rows)} runs</div>
    
    <div style="display: flex; gap: 8px; margin-bottom: 20px;">
      <button class="btn" onclick="window.print()" style="font-size:12px;">Print</button>
      <button class="btn" onclick="exportTableToCSV('all-runs', 'benchmark_all_runs.csv')" style="font-size:12px;">Export CSV</button>
      <button class="btn" onclick="copyTableToClipboard('all-runs')" style="font-size:12px;">Copy Table</button>
    </div>
    
    {generate_summary_cards(rows)}
    
    {generate_backend_overview_table(rows_sorted)}
    
    {generate_systemds_vs_vllm_summary(rows_sorted)}
    
    {generate_cost_tradeoff_table(rows_sorted)}
    
    {generate_head_to_head_section(rows_sorted)}
    
    {generate_accuracy_comparison_table(rows_sorted)}
    
    {generate_latency_comparison_table(rows_sorted)}
    
    {generate_latency_breakdown_table(rows_sorted)}
    
    {generate_consistency_metrics_table(rows_sorted)}
    
    {generate_cost_efficiency_table(rows_sorted)}
    
    {generate_cost_analysis_section(rows_sorted)}
    
    {generate_charts_section(rows_sorted)}
    
    {generate_full_table("All Runs", rows_sorted, "all-runs")}
    
    {generate_per_sample_results(results_dir)}
    
  </div>
  
  <script>
    function exportTableToCSV(tableId, filename) {{
      const table = document.querySelector('#' + tableId + ' table');
      if (!table) {{ alert('Table not found'); return; }}
      
      let csv = [];
      const rows = table.querySelectorAll('tr');
      
      for (const row of rows) {{
        const cols = row.querySelectorAll('th, td');
        const rowData = [];
        for (const col of cols) {{
          let text = col.innerText.replace(/"/g, '""');
          rowData.push('"' + text + '"');
        }}
        csv.push(rowData.join(','));
      }}
      
      const csvContent = csv.join('\\n');
      const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      link.click();
    }}
    
    function copyTableToClipboard(tableId) {{
      const table = document.querySelector('#' + tableId + ' table');
      if (!table) {{ alert('Table not found'); return; }}
      
      let text = [];
      const rows = table.querySelectorAll('tr');
      
      for (const row of rows) {{
        const cols = row.querySelectorAll('th, td');
        const rowData = [];
        for (const col of cols) {{
          rowData.push(col.innerText);
        }}
        text.push(rowData.join('\\t'));
      }}
      
      navigator.clipboard.writeText(text.join('\\n')).then(() => {{
        alert('Table copied to clipboard! Paste in Excel or Google Sheets.');
      }});
    }}
    
    function printSection(tableId) {{
      const tableWrapper = document.getElementById(tableId);
      if (!tableWrapper) {{ alert('Table not found'); return; }}
      
      const printWindow = window.open('', '_blank');
      printWindow.document.write(`
        <html>
        <head>
          <title>Print - ${{tableId}}</title>
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; font-size: 8px; }}
            th, td {{ border: 1px solid #ddd; padding: 4px 6px; text-align: left; white-space: nowrap; }}
            th {{ background: #f0f0f0; font-weight: bold; }}
            tr:nth-child(even) {{ background: #fafafa; }}
            @page {{ size: landscape; margin: 0.5cm; }}
          </style>
        </head>
        <body>
          <h2>${{tableId.replace(/-/g, ' ').replace(/workload /i, '')}}</h2>
          ${{tableWrapper.innerHTML}}
          <script>window.onload = function() {{ window.print(); window.close(); }}</` + `script>
        </body>
        </html>
      `);
      printWindow.document.close();
    }}
  </script>
</body>
</html>
"""

    Path(args.out).write_text(html_doc, encoding="utf-8")
    print(f"OK: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
