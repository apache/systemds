
"""Generate HTML benchmark report with charts and visualizations."""
import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_run_dir(p: Path) -> bool:
    return p.is_dir() and (p / "metrics.json").exists() and (p / "run_config.json").exists()


def iter_run_dirs(results_dir: Path) -> List[Path]:
    if not results_dir.exists():
        return []
    seen = set()
    runs: List[Path] = []
    for p in results_dir.iterdir():
        if is_run_dir(p):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                runs.append(p)
    for group in results_dir.iterdir():
        if not group.is_dir():
            continue
        for p in group.iterdir():
            if is_run_dir(p):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    runs.append(p)
    return runs


def manifest_timestamp(run_dir: Path) -> str:
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return ""
    try:
        m = read_json(mpath)
        ts = m.get("timestamp_utc")
        return "" if ts is None else str(ts)
    except Exception:
        return ""


def token_stats(samples_path: Path) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    if not samples_path.exists():
        return (None, None, None, None)
    total_tokens = 0
    total_in = 0
    total_out = 0
    count = 0
    saw_any = False
    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                usage = (obj.get("extra") or {}).get("usage") or {}
                tt = usage.get("total_tokens")
                it = usage.get("input_tokens")
                ot = usage.get("output_tokens")
                if tt is None and it is None and ot is None:
                    continue
                saw_any = True
                if tt is not None:
                    total_tokens += int(tt)
                if it is not None:
                    total_in += int(it)
                if ot is not None:
                    total_out += int(ot)
                count += 1
    except Exception:
        return (None, None, None, None)
    if not saw_any or count == 0:
        return (None, None, None, None)
    avg = (total_tokens / count) if total_tokens > 0 else None
    return (
        total_tokens if total_tokens > 0 else None,
        avg,
        total_in if total_in > 0 else None,
        total_out if total_out > 0 else None,
    )


def cost_stats(samples_path: Path) -> Optional[float]:
    """Calculate total cost from samples."""
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
    # return 0.0 for local backends (they report cost_usd: 0.0)
    return total_cost if found_any else None


def timing_stats(samples_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Calculate TTFT and generation time means from samples."""
    if not samples_path.exists():
        return (None, None)
    ttft_vals = []
    gen_vals = []
    try:
        with samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ttft = obj.get("ttft_ms")
                    gen = obj.get("generation_ms")
                    if ttft is not None:
                        ttft_vals.append(float(ttft))
                    if gen is not None:
                        gen_vals.append(float(gen))
                except Exception:
                    continue
    except Exception:
        return (None, None)
    
    ttft_mean = sum(ttft_vals) / len(ttft_vals) if ttft_vals else None
    gen_mean = sum(gen_vals) / len(gen_vals) if gen_vals else None
    return (ttft_mean, gen_mean)


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
        return "$0.00"
    return f"${v:.4f}"


# colors for backends
BACKEND_COLORS = {
    "openai": "#10a37f",
    "mlx": "#ff6b6b",
    "ollama": "#4ecdc4",
    "vllm": "#9b59b6",
}

# colors for workloads
WORKLOAD_COLORS = {
    "math": "#3498db",
    "reasoning": "#e74c3c",
    "summarization": "#2ecc71",
    "json_extraction": "#f39c12",
}


def generate_bar_chart_svg(data: List[Tuple[str, float, str]], title: str, 
                            width: int = 500, height: int = 300,
                            value_suffix: str = "", show_values: bool = True) -> str:
    """Generate SVG bar chart. data = [(label, value, color), ...]"""
    if not data:
        return ""
    
    max_val = max(d[1] for d in data) if data else 1
    bar_height = 28
    gap = 8
    left_margin = 120
    right_margin = 80
    top_margin = 40
    chart_width = width - left_margin - right_margin
    chart_height = len(data) * (bar_height + gap)
    total_height = chart_height + top_margin + 20
    
    svg = [f'<svg width="{width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{width//2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{html.escape(title)}</text>')
    
    for i, (label, value, color) in enumerate(data):
        y = top_margin + i * (bar_height + gap)
        bar_width = (value / max_val) * chart_width if max_val > 0 else 0
        
  
        svg.append(f'<text x="{left_margin - 8}" y="{y + bar_height//2 + 4}" text-anchor="end" font-size="11">{html.escape(label[:15])}</text>')
        
     
        svg.append(f'<rect x="{left_margin}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" rx="3"/>')
        
  
        if show_values:
            val_text = f"{value:.1f}{value_suffix}" if isinstance(value, float) else f"{value}{value_suffix}"
            svg.append(f'<text x="{left_margin + bar_width + 5}" y="{y + bar_height//2 + 4}" font-size="11">{val_text}</text>')
    
    svg.append('</svg>')
    return '\n'.join(svg)


def generate_grouped_bar_chart_svg(data: Dict[str, Dict[str, float]], title: str,
                                    group_colors: Dict[str, str],
                                    width: int = 600, height: int = 350,
                                    value_suffix: str = "") -> str:
    """Generate grouped bar chart. data = {category: {group: value}}"""
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


def generate_accuracy_comparison_table(rows: List[Dict[str, Any]]) -> str:
    """Generate accuracy comparison table by workload and backend."""
    # group by base workload and backend, take latest run only
    # this avoids duplicates like "reasoning" and "reasoning (toy)"
    data: Dict[str, Dict[str, Dict[str, Any]]] = {} 
    
    for r in rows:

        workload = r.get("workload", "")
        backend = r.get("backend", "")
        if not workload or not backend:
            continue
        
        if workload not in data:
            data[workload] = {}
        
        # keep latest
        if backend not in data[workload]:
            data[workload][backend] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Accuracy Comparison by Workload</h2>']
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
                acc_count = data[wl][b].get("accuracy_count", "")
                if acc is not None:
                    pct = acc * 100
                    color = "#2ecc71" if pct >= 80 else "#f39c12" if pct >= 50 else "#e74c3c"
                    out.append(f'<td style="background: {color}22; color: {color}; font-weight: bold;">{pct:.0f}%<br><small>{acc_count}</small></td>')
                else:
                    out.append('<td>-</td>')
            else:
                out.append('<td>-</td>')
        out.append('</tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_latency_comparison_table(rows: List[Dict[str, Any]]) -> str:
    """Generate latency comparison table by workload and backend."""
 
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for r in rows:

        workload = r.get("workload", "")
        backend = r.get("backend", "")
        if not workload or not backend:
            continue
        if workload not in data:
            data[workload] = {}

        if backend not in data[workload]:
            data[workload][backend] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Latency Comparison (p50 ms)</h2>']
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
                    out.append(f'<td>{lat:.0f}ms</td>')
                else:
                    out.append('<td>-</td>')
            else:
                out.append('<td>-</td>')
        out.append('</tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_latency_breakdown_table(rows: List[Dict[str, Any]]) -> str:
    """Generate latency breakdown table showing TTFT vs Generation time (like prefill vs decode)."""
    # only include rows with TTFT data
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for r in rows:
        workload = r.get("workload", "")
        backend = r.get("backend", "")
        ttft = r.get("ttft_mean")
        gen = r.get("gen_mean")
        
        if not workload or not backend:
            continue
        if ttft is None and gen is None:
            continue
            
        if workload not in data:
            data[workload] = {}
        if backend not in data[workload]:
            data[workload][backend] = r
    
    if not data:
        return '<p class="muted">No TTFT data available. Enable streaming mode for OpenAI to measure TTFT.</p>'
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>⏱️ Latency Breakdown (TTFT vs Generation)</h2>']
    out.append('<p><small>Time-To-First-Token (TTFT) = prefill/prompt processing. Generation = token decoding. Only available for streaming backends.</small></p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Workload</th><th>Backend</th><th>TTFT (ms)</th><th>Generation (ms)</th><th>Total (ms)</th><th>TTFT %</th></tr></thead><tbody>')
    
    for wl in workloads:
        for b in backends:
            if b in data[wl]:
                r = data[wl][b]
                ttft = safe_float(r.get("ttft_mean"))
                gen = safe_float(r.get("gen_mean"))
                total = safe_float(r.get("lat_mean"))
                
                ttft_str = f'{ttft:.0f}' if ttft else '-'
                gen_str = f'{gen:.0f}' if gen else '-'
                total_str = f'{total:.0f}' if total else '-'
                
                if ttft and gen:
                    ttft_pct = (ttft / (ttft + gen)) * 100
                    pct_str = f'{ttft_pct:.0f}%'
                    # color based on TTFT proportion
                    color = '#2ecc71' if ttft_pct < 30 else '#f39c12' if ttft_pct < 60 else '#e74c3c'
                else:
                    pct_str = '-'
                    color = '#666'
                
                out.append(f'<tr><td>{html.escape(wl)}</td><td>{html.escape(b)}</td>')
                out.append(f'<td>{ttft_str}</td><td>{gen_str}</td><td>{total_str}</td>')
                out.append(f'<td style="color: {color}; font-weight: bold;">{pct_str}</td></tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_consistency_metrics_table(rows: List[Dict[str, Any]]) -> str:
    """Generate consistency metrics table showing latency variance across backends."""
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for r in rows:
        workload = r.get("workload", "")
        backend = r.get("backend", "")
        if not workload or not backend:
            continue
        if workload not in data:
            data[workload] = {}
        if backend not in data[workload]:
            data[workload][backend] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>📊 Consistency Metrics (Latency Variance)</h2>']
    out.append('<p><small>CV (Coefficient of Variation) = std/mean × 100%. Lower CV = more consistent performance.</small></p>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Workload</th><th>Backend</th><th>Mean (ms)</th><th>Std (ms)</th><th>Min (ms)</th><th>Max (ms)</th><th>CV (%)</th></tr></thead><tbody>')
    
    for wl in workloads:
        for b in backends:
            if b in data[wl]:
                r = data[wl][b]
                mean = safe_float(r.get("lat_mean"))
                std = safe_float(r.get("lat_std"))
                lat_min = safe_float(r.get("lat_min"))
                lat_max = safe_float(r.get("lat_max"))
                cv = safe_float(r.get("lat_cv"))
                
                mean_str = f'{mean:.0f}' if mean else '-'
                std_str = f'{std:.0f}' if std else '-'
                min_str = f'{lat_min:.0f}' if lat_min else '-'
                max_str = f'{lat_max:.0f}' if lat_max else '-'
                
                if cv is not None:
                    cv_str = f'{cv:.1f}%'
                 
                    color = '#2ecc71' if cv < 20 else '#f39c12' if cv < 50 else '#e74c3c'
                else:
                    cv_str = '-'
                    color = '#666'
                
                out.append(f'<tr><td>{html.escape(wl)}</td><td>{html.escape(b)}</td>')
                out.append(f'<td>{mean_str}</td><td>{std_str}</td><td>{min_str}</td><td>{max_str}</td>')
                out.append(f'<td style="color: {color}; font-weight: bold;">{cv_str}</td></tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_cost_efficiency_table(rows: List[Dict[str, Any]]) -> str:
    """Generate cost efficiency comparison table (cost per correct answer)."""
  
    data: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for r in rows:
        workload = r.get("workload", "")
        backend = r.get("backend", "")
        if not workload or not backend:
            continue
        if workload not in data:
            data[workload] = {}
    
        if backend not in data[workload]:
            data[workload][backend] = r
    
    if not data:
        return ""
    
    workloads = sorted(data.keys())
    backends = sorted(set(b for w in data.values() for b in w.keys()))
    
    out = ['<h2>Cost Efficiency ($ per correct answer)</h2>']
    out.append('<p><small>Lower is better. Shows cost divided by number of correct answers. Only for OpenAI (local backends have no API cost).</small></p>')
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
                cost = safe_float(r.get("cost"))
                acc_mean = r.get("accuracy_mean")
                n = safe_float(r.get("n")) or 10
                
                if cost and cost > 0 and acc_mean is not None and acc_mean > 0:
                    correct_count = int(n * acc_mean)
                    cost_per_correct = cost / correct_count if correct_count > 0 else None
                    if cost_per_correct is not None:
                       
                        color = "#2ecc71" if cost_per_correct < 0.001 else "#f39c12" if cost_per_correct < 0.01 else "#e74c3c"
                        out.append(f'<td style="background: {color}22; color: {color}; font-weight: bold;">${cost_per_correct:.4f}</td>')
                    else:
                        out.append('<td>-</td>')
                elif b != "openai":
                  
                    out.append('<td style="color: #2ecc71;">$0 (local)</td>')
                else:
                    out.append('<td>-</td>')
            else:
                out.append('<td>-</td>')
        out.append('</tr>')
    
    out.append('</tbody></table>')
    return '\n'.join(out)


def generate_cost_analysis_section(rows: List[Dict[str, Any]]) -> str:
    """Generate comprehensive cost analysis comparing cloud vs local inference."""
    

    openai_costs = []
    local_runs = []
    
    for r in rows:
        backend = r.get("backend", "")
        cost = safe_float(r.get("cost"))
        workload = r.get("workload", "")
        acc = r.get("accuracy_mean")
        n = safe_float(r.get("n")) or 10
        lat = safe_float(r.get("lat_p50"))
        
        if backend == "openai" and cost and cost > 0:
            openai_costs.append({
                "workload": workload,
                "cost": cost,
                "accuracy": acc,
                "n": n,
                "latency": lat,
                "total_tokens": r.get("total_tokens"),
            })
        elif backend in ["ollama", "mlx", "vllm"]:
            local_runs.append({
                "backend": backend,
                "workload": workload,
                "accuracy": acc,
                "n": n,
                "latency": lat,
            })
    
    if not openai_costs:
        return ""
    
    out = ['<h2>💰 Cost Analysis: Cloud vs Local Inference</h2>']
    
  
    total_openai_cost = sum(c["cost"] for c in openai_costs)
    avg_cost_per_run = total_openai_cost / len(openai_costs) if openai_costs else 0
    total_queries = sum(c["n"] for c in openai_costs)
    cost_per_query = total_openai_cost / total_queries if total_queries > 0 else 0
    
    out.append('<div class="cost-analysis-grid">')
    

    out.append('''
    <div class="cost-card cloud">
        <h3>☁️ Cloud (OpenAI API)</h3>
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
            <div class="pros">✅ Highest accuracy</div>
            <div class="pros">✅ No hardware needed</div>
            <div class="cons">❌ Per-query costs</div>
            <div class="cons">❌ Network latency</div>
        </div>
    </div>
    ''')
    
 
    out.append('''
    <div class="cost-card local">
        <h3>🖥️ Local Inference</h3>
        <div class="cost-stats">
    ''')
    out.append(f'<div class="stat"><span class="label">API Cost:</span> <span class="value highlight">$0</span></div>')
    out.append(f'<div class="stat"><span class="label">Local Runs:</span> <span class="value">{len(local_runs)}</span></div>')
    out.append(f'<div class="stat"><span class="label">Backends:</span> <span class="value">{len(set(r["backend"] for r in local_runs))}</span></div>')
    out.append('''
        </div>
        <div class="pros-cons">
            <div class="pros">✅ Zero API cost</div>
            <div class="pros">✅ Privacy (data stays local)</div>
            <div class="cons">❌ Hardware required</div>
            <div class="cons">❌ Lower accuracy on complex tasks</div>
        </div>
    </div>
    ''')
    
    out.append('</div>')  
    
  
    out.append('<h3>📊 Cost Projection (1000 queries)</h3>')
    out.append('<table class="comparison-table">')
    out.append('<thead><tr><th>Backend</th><th>Est. Cost (1000 queries)</th><th>Notes</th></tr></thead>')
    out.append('<tbody>')
    

    projected_1k = cost_per_query * 1000
    out.append(f'<tr><td>OpenAI (gpt-4.1-mini)</td><td style="color: #e74c3c; font-weight: bold;">${projected_1k:.2f}</td><td>Based on current usage</td></tr>')
    

    out.append('<tr><td>Ollama (local)</td><td style="color: #2ecc71; font-weight: bold;">$0</td><td>Requires Mac/Linux, ~4GB RAM</td></tr>')
    out.append('<tr><td>MLX (Apple Silicon)</td><td style="color: #2ecc71; font-weight: bold;">$0</td><td>Requires M1/M2/M3 Mac</td></tr>')
    out.append('<tr><td>vLLM (GPU server)</td><td style="color: #f39c12; font-weight: bold;">~$5-20</td><td>Cloud GPU: ~$0.20-0.50/hour</td></tr>')
    
    out.append('</tbody></table>')
    
    out.append('<p class="muted"><small>Note: Local backend costs exclude hardware purchase/depreciation and electricity. vLLM cost estimate based on cloud GPU rental.</small></p>')
    
    return '\n'.join(out)


def generate_scatter_plot_svg(data: List[Tuple[float, float, str, str]], 
                               title: str, x_label: str, y_label: str,
                               width: int = 400, height: int = 300) -> str:
    """Generate SVG scatter plot. data = [(x, y, label, color), ...]"""
    if not data:
        return '<p class="muted">No data with both cost and accuracy</p>'
    
    
    valid_data = [(x, y, l, c) for x, y, l, c in data if x > 0 and y is not None]
    if not valid_data:
        return '<p class="muted">No runs with cost data</p>'
    
    left_margin = 60
    right_margin = 120
    top_margin = 40
    bottom_margin = 50
    
    chart_width = width - left_margin - right_margin
    chart_height = height - top_margin - bottom_margin
    
    x_vals = [d[0] for d in valid_data]
    y_vals = [d[1] for d in valid_data]
    
    x_min, x_max = 0, max(x_vals) * 1.1
    y_min, y_max = 0, min(100, max(y_vals) * 1.1)
    
    def scale_x(v):
        return left_margin + (v - x_min) / (x_max - x_min) * chart_width if x_max > x_min else left_margin
    def scale_y(v):
        return top_margin + chart_height - (v - y_min) / (y_max - y_min) * chart_height if y_max > y_min else top_margin + chart_height
    
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<text x="{width//2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{html.escape(title)}</text>')
    
  
    svg.append(f'<line x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{top_margin + chart_height}" stroke="#ccc" stroke-width="1"/>')
    svg.append(f'<line x1="{left_margin}" y1="{top_margin + chart_height}" x2="{left_margin + chart_width}" y2="{top_margin + chart_height}" stroke="#ccc" stroke-width="1"/>')
    

    svg.append(f'<text x="{left_margin + chart_width//2}" y="{height - 10}" text-anchor="middle" font-size="11">{html.escape(x_label)}</text>')
    svg.append(f'<text x="15" y="{top_margin + chart_height//2}" text-anchor="middle" font-size="11" transform="rotate(-90, 15, {top_margin + chart_height//2})">{html.escape(y_label)}</text>')
    
    
    for pct in [25, 50, 75, 100]:
        if pct <= y_max:
            y = scale_y(pct)
            svg.append(f'<line x1="{left_margin}" y1="{y}" x2="{left_margin + chart_width}" y2="{y}" stroke="#eee" stroke-width="1"/>')
            svg.append(f'<text x="{left_margin - 5}" y="{y + 4}" text-anchor="end" font-size="9">{pct}%</text>')
    
    
    seen_workloads = {}
    for x, y, label, color in valid_data:
        px, py = scale_x(x), scale_y(y)
        svg.append(f'<circle cx="{px}" cy="{py}" r="8" fill="{color}" fill-opacity="0.7" stroke="{color}" stroke-width="2"/>')
        if label not in seen_workloads:
            seen_workloads[label] = color
    
    
    legend_x = left_margin + chart_width + 15
    legend_y = top_margin + 10
    for i, (label, color) in enumerate(seen_workloads.items()):
        y_pos = legend_y + i * 20
        svg.append(f'<circle cx="{legend_x + 6}" cy="{y_pos}" r="6" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 18}" y="{y_pos + 4}" font-size="10">{html.escape(label[:12])}</text>')
    
    svg.append('</svg>')
    return '\n'.join(svg)


def generate_summary_section(rows: List[Dict[str, Any]]) -> str:
    """Generate comprehensive summary statistics section."""
  
    backends = set(r.get("backend") for r in rows if r.get("backend"))
    workloads = set(r.get("workload") for r in rows if r.get("workload"))
    models = set(r.get("backend_model") for r in rows if r.get("backend_model"))
    total_runs = len(rows)
    

    costs = [safe_float(r.get("cost")) for r in rows if r.get("backend") == "openai" and safe_float(r.get("cost"))]
    total_cost = sum(costs) if costs else 0
    runs_with_cost = len(costs)
    avg_cost = total_cost / runs_with_cost if runs_with_cost > 0 else 0
    
    latencies = [safe_float(r.get("lat_p50")) for r in rows if safe_float(r.get("lat_p50")) is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    

    acc_by_workload: Dict[str, List[float]] = {}
    for r in rows:
        wl = r.get("workload", "")
        acc = r.get("accuracy_mean")
        if wl and acc is not None:
            if wl not in acc_by_workload:
                acc_by_workload[wl] = []
            acc_by_workload[wl].append(acc * 100)
    
    best_workload = ""
    worst_workload = ""
    best_acc = 0
    worst_acc = 100
    for wl, accs in acc_by_workload.items():
        avg = sum(accs) / len(accs)
        if avg > best_acc:
            best_acc = avg
            best_workload = wl
        if avg < worst_acc:
            worst_acc = avg
            worst_workload = wl
    
    out = ['<div class="summary-section">']
    out.append('<h2>📊 Summary Statistics</h2>')
    out.append('<div class="summary-grid">')
    
    
    out.append('''
    <div class="summary-card">
        <div class="card-header">OVERVIEW</div>
        <div class="card-content">
    ''')
    out.append(f'<div class="stat-row"><span class="stat-label">Total Runs:</span> <span class="stat-value">{total_runs}</span></div>')
    out.append(f'<div class="stat-row"><span class="stat-label">Workloads:</span> <span class="stat-value">{", ".join(sorted(workloads))}</span></div>')
    out.append(f'<div class="stat-row"><span class="stat-label">Models:</span> <span class="stat-value">{", ".join(sorted(str(m) for m in models if m))}</span></div>')
    out.append(f'<div class="stat-row"><span class="stat-label">Backends:</span> <span class="stat-value">{", ".join(sorted(backends))}</span></div>')
    out.append('</div></div>')
    

    out.append('''
    <div class="summary-card">
        <div class="card-header">💰 COST</div>
        <div class="card-content">
    ''')
    out.append(f'<div class="stat-row"><span class="stat-label">Total Cost:</span> <span class="stat-value">${total_cost:.4f}</span></div>')
    out.append(f'<div class="stat-row"><span class="stat-label">Runs with Cost:</span> <span class="stat-value">{runs_with_cost}/{total_runs}</span></div>')
    out.append(f'<div class="stat-row"><span class="stat-label">Avg Cost/Run:</span> <span class="stat-value">${avg_cost:.4f}</span></div>')
    out.append('</div></div>')
    
 
    out.append('''
    <div class="summary-card">
        <div class="card-header">🎯 ACCURACY</div>
        <div class="card-content">
    ''')
    out.append(f'<div class="stat-row"><span class="stat-label">Best Workload:</span> <span class="stat-value">{best_workload} ({best_acc:.1f}%)</span></div>')
    out.append(f'<div class="stat-row"><span class="stat-label">Hardest Workload:</span> <span class="stat-value">{worst_workload} ({worst_acc:.1f}%)</span></div>')
    out.append('</div></div>')
    
    
    out.append('''
    <div class="summary-card">
        <div class="card-header">⚡ LATENCY</div>
        <div class="card-content">
    ''')
    out.append(f'<div class="stat-row"><span class="stat-label">Avg Latency:</span> <span class="stat-value">{avg_latency:.2f} ms</span></div>')
    out.append(f'<div class="stat-row"><span class="stat-label">Min:</span> <span class="stat-value">{min_latency:.2f} ms</span></div>')
    out.append(f'<div class="stat-row"><span class="stat-label">Max:</span> <span class="stat-value">{max_latency:.2f} ms</span></div>')
    out.append('</div></div>')
    
    out.append('</div>') 
    
  
    out.append('<h2>📈 Visualizations</h2>')
    out.append('<div class="viz-grid">')
    
    
    accuracy_bars = []
    for wl, accs in sorted(acc_by_workload.items()):
        avg = sum(accs) / len(accs)
        color = WORKLOAD_COLORS.get(wl, "#999")
        accuracy_bars.append((wl, avg, color))
    
    out.append('<div class="viz-container">')
    out.append(generate_bar_chart_svg(accuracy_bars, "Accuracy by Workload", width=350, height=250, value_suffix="%"))
    out.append('</div>')
    
   
    scatter_data = []
    for r in rows:
        cost = safe_float(r.get("cost"))
        acc = r.get("accuracy_mean")
        wl = r.get("workload", "")
        if cost and cost > 0 and acc is not None and wl:
            color = WORKLOAD_COLORS.get(wl, "#999")
            scatter_data.append((cost, acc * 100, wl, color))
    
    out.append('<div class="viz-container">')
    out.append(generate_scatter_plot_svg(scatter_data, "Cost vs Accuracy", "Cost ($)", "Accuracy (%)", width=450, height=250))
    out.append('</div>')
    
    out.append('</div>')  
    out.append('</div>') 
    
    return '\n'.join(out)


def generate_summary_cards(rows: List[Dict[str, Any]]) -> str:
    """Generate summary section - wrapper for generate_summary_section."""
    return generate_summary_section(rows)


def generate_charts_section(rows: List[Dict[str, Any]]) -> str:
    """Generate all charts."""
    out = ['<h2>Performance Charts</h2>', '<div class="charts-grid">']
    
    
    latest: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        wl = r.get("workload", "")
        be = r.get("backend", "")
        if not wl or not be:
            continue
        if wl not in latest:
            latest[wl] = {}
        latest[wl][be] = r
    
    
    accuracy_data: Dict[str, Dict[str, float]] = {}
    for wl, backends in latest.items():
        accuracy_data[wl] = {}
        for be, r in backends.items():
            acc = r.get("accuracy_mean")
            if acc is not None:
                accuracy_data[wl][be] = acc * 100
    
    if accuracy_data:
        out.append('<div class="chart-container">')
        out.append(generate_grouped_bar_chart_svg(
            accuracy_data, "Accuracy by Workload (%)", 
            BACKEND_COLORS, value_suffix="%"
        ))
        out.append('</div>')
    
   
    latency_data: Dict[str, Dict[str, float]] = {}
    for wl, backends in latest.items():
        latency_data[wl] = {}
        for be, r in backends.items():
            lat = safe_float(r.get("lat_p50"))
            if lat is not None:
                latency_data[wl][be] = lat / 1000 
    
    if latency_data:
        out.append('<div class="chart-container">')
        out.append(generate_grouped_bar_chart_svg(
            latency_data, "Latency by Workload (p50, seconds)",
            BACKEND_COLORS, value_suffix="s"
        ))
        out.append('</div>')
    
  
    throughput_data: Dict[str, Dict[str, float]] = {}
    for wl, backends in latest.items():
        throughput_data[wl] = {}
        for be, r in backends.items():
            thr = safe_float(r.get("thr"))
            if thr is not None:
                throughput_data[wl][be] = thr
    
    if throughput_data:
        out.append('<div class="chart-container">')
        out.append(generate_grouped_bar_chart_svg(
            throughput_data, "Throughput by Workload (req/s)",
            BACKEND_COLORS, value_suffix=" req/s"
        ))
        out.append('</div>')
    
    out.append('</div>')
    return '\n'.join(out)



def fmt_cost_if_real(r: Dict[str, Any]) -> str:
    cost = r.get("cost")
    backend = r.get("backend", "")
    if backend == "openai" and cost is not None:
        return fmt_cost(cost)
    return "-"

def fmt_cost_per_1m_if_real(r: Dict[str, Any]) -> str:
    cost = r.get("cost_per_1m_tokens")
    backend = r.get("backend", "")
    if backend == "openai" and cost is not None:
        return fmt_cost(cost)
    return "-"


FULL_TABLE_COLUMNS = [
    ("run_dir", "Run", lambda r: f'<code>{html.escape(str(r.get("run_dir", ""))[:25])}</code>'),
    ("ts", "Timestamp (UTC)", lambda r: html.escape((r.get("ts", "") or "")[:19].replace("T", " "))),
    ("backend", "Backend", lambda r: html.escape(r.get("backend", ""))),
    ("backend_model", "Model", lambda r: html.escape(str(r.get("backend_model", ""))[:20])),
    ("workload", "Workload", lambda r: html.escape(r.get("workload", ""))),
    ("n", "n", lambda r: fmt(r.get("n"))),
    ("accuracy", "Accuracy", lambda r: f'{r.get("accuracy_mean", 0)*100:.1f}% ({r.get("accuracy_count", "")})' if r.get("accuracy_mean") is not None else "N/A"),
    ("cost", "Cost ($)", fmt_cost_if_real),
    ("cost_per_1m", "$/1M tok", fmt_cost_per_1m_if_real),
    ("mem_peak", "Mem Peak (MB)", lambda r: fmt_num(r.get("mem_peak"), 1)),
    ("cpu_avg", "CPU Avg (%)", lambda r: fmt_num(r.get("cpu_avg"), 1)),
    ("lat_mean", "lat mean (ms)", lambda r: fmt_num(r.get("lat_mean"), 2)),
    ("lat_p50", "p50 (ms)", lambda r: fmt_num(r.get("lat_p50"), 2)),
    ("lat_p95", "p95 (ms)", lambda r: fmt_num(r.get("lat_p95"), 2)),
    ("lat_std", "Lat Std (ms)", lambda r: fmt_num(r.get("lat_std"), 2)),
    ("lat_cv", "Lat CV (%)", lambda r: fmt_pct(r.get("lat_cv"))),
    ("lat_min", "Lat Min (ms)", lambda r: fmt_num(r.get("lat_min"), 2)),
    ("lat_max", "Lat Max (ms)", lambda r: fmt_num(r.get("lat_max"), 2)),
    ("ttft_mean", "TTFT (ms)", lambda r: fmt_num(r.get("ttft_mean"), 2)),
    ("gen_mean", "Gen (ms)", lambda r: fmt_num(r.get("gen_mean"), 2)),
    ("thr", "throughput (req/s)", lambda r: fmt_num(r.get("thr"), 4)),
    ("total_tokens", "total tok", lambda r: fmt(r.get("total_tokens"))),
    ("avg_tokens", "avg tok", lambda r: fmt_num(r.get("avg_tokens"), 1)),
    ("total_input_tokens", "in tok", lambda r: fmt(r.get("total_input_tokens"))),
    ("total_output_tokens", "out tok", lambda r: fmt(r.get("total_output_tokens"))),
    ("toks_total", "tok/s (total)", lambda r: fmt_num(r.get("toks_total"), 2)),
    ("ms_per_tok_total", "ms/tok (total)", lambda r: fmt_num(r.get("ms_per_tok_total"), 2)),
    ("toks_out", "tok/s (out)", lambda r: fmt_num(r.get("toks_out"), 2)),
    ("ms_per_tok_out", "ms/tok (out)", lambda r: fmt_num(r.get("ms_per_tok_out"), 2)),
]


def generate_full_table(title: str, table_rows: List[Dict[str, Any]], table_id: str = "", is_h3: bool = False) -> str:
    """Generate full results table with all columns."""
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
    """Generate separate tables for each workload category."""
    
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
            ttft_mean, gen_mean = timing_stats(run_dir / "samples.jsonl")
            
            
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
            known_sources = ["toy", "gsm8k", "boolq", "xsum", "cnn", "logiqa", "ner"]
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
                "total_tokens": total,
                "avg_tokens": avg,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "cost": cost,
                "cost_per_1m_tokens": cost_per_1m,
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
    h2 {{ margin: 30px 0 15px 0; color: #1a1a2e; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
    h3 {{ margin: 20px 0 10px 0; color: #333; }}
    .meta {{ color: #666; margin-bottom: 24px; font-size: 14px; }}
    
    .summary-cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 16px;
        margin-bottom: 30px;
    }}
    .card {{
        background: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .card-value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
    .card-label {{ font-size: 12px; color: #666; margin-top: 4px; }}
    
    /* Summary Section */
    .summary-section {{
        background: #1a1a2e;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 30px;
        color: white;
    }}
    .summary-section h2 {{
        color: white;
        border-bottom: 1px solid rgba(255,255,255,0.2);
        margin: 0 0 20px 0;
    }}
    .summary-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 30px;
    }}
    .summary-card {{
        background: rgba(255,255,255,0.1);
        padding: 16px;
        border-radius: 8px;
    }}
    .card-header {{
        font-size: 11px;
        font-weight: 600;
        color: rgba(255,255,255,0.7);
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .card-content {{
        font-size: 12px;
    }}
    .stat-row {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }}
    .stat-label {{
        color: rgba(255,255,255,0.8);
    }}
    .stat-value {{
        font-weight: 600;
        color: white;
    }}
    .viz-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
    }}
    .viz-container {{
        background: white;
        padding: 20px;
        border-radius: 8px;
        min-height: 280px;
    }}
    
    @media (max-width: 1200px) {{
        .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
        .viz-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 768px) {{
        .summary-grid {{ grid-template-columns: 1fr; }}
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 24px;
    }}
    .comparison-table th, .comparison-table td {{
        padding: 12px 16px;
        text-align: center;
        border-bottom: 1px solid #eee;
    }}
    .comparison-table th {{
        background: #f8f9fa;
        font-weight: 600;
        color: #1a1a2e;
    }}
    .comparison-table td:first-child {{
        text-align: left;
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
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    .cost-card.cloud {{
        border-left: 4px solid #3498db;
    }}
    .cost-card.local {{
        border-left: 4px solid #2ecc71;
    }}
    .cost-card h3 {{
        margin: 0 0 16px 0;
        font-size: 16px;
    }}
    .cost-stats {{
        margin-bottom: 16px;
    }}
    .cost-stats .stat {{
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #f0f0f0;
    }}
    .cost-stats .label {{
        color: #666;
    }}
    .cost-stats .value {{
        font-weight: 600;
        color: #1a1a2e;
    }}
    .cost-stats .value.highlight {{
        color: #2ecc71;
        font-size: 18px;
    }}
    .pros-cons {{
        font-size: 12px;
    }}
    .pros {{ color: #2ecc71; margin: 4px 0; }}
    .cons {{ color: #e74c3c; margin: 4px 0; }}
    
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
        padding: 6px 12px;
        background: #3498db;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 11px;
        margin-left: 8px;
    }}
    .btn-small:hover {{ background: #2980b9; }}
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
    .full-table tr:hover {{ background: #f0f7ff; }}
    
    code {{ 
        background: #f1f3f4; 
        padding: 2px 4px; 
        border-radius: 3px; 
        font-size: 10px;
    }}
    
    /* Print/Export buttons */
    .toolbar {{
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }}
    .btn {{
        padding: 10px 20px;
        background: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .btn:hover {{ background: #2980b9; }}
    .btn-green {{ background: #2ecc71; }}
    .btn-green:hover {{ background: #27ae60; }}
    .btn-purple {{ background: #9b59b6; }}
    .btn-purple:hover {{ background: #8e44ad; }}
    
    /* Print styles for better screenshot/print */
    @media print {{
        .toolbar {{ display: none !important; }}
        body {{ 
            padding: 10px; 
            background: white;
            font-size: 9px;
        }}
        .summary-cards, .charts-grid, .chart-container {{ 
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
        .summary-cards {{ grid-template-columns: repeat(2, 1fr); }}
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
        background: #e8f5e9;
        border-left-color: #4caf50;
    }}
    .sample-item.incorrect {{
        background: #ffebee;
        border-left-color: #f44336;
    }}
    .sample-item.unknown {{
        background: #fff8e1;
        border-left-color: #ff9800;
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
    <h1>systemds-bench-gpt Benchmark Report</h1>
    <div class="meta">Generated: {gen_ts} | Total Runs: {len(rows)}</div>
    
    <div class="toolbar">
      <button class="btn" onclick="window.print()">
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M2.5 8a.5.5 0 1 0 0-1 .5.5 0 0 0 0 1z"/><path d="M5 1a2 2 0 0 0-2 2v2H2a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h1v1a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-1h1a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-1V3a2 2 0 0 0-2-2H5zM4 3a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2H4V3zm1 5a2 2 0 0 0-2 2v1H2a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v-1a2 2 0 0 0-2-2H5zm7 2v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1z"/></svg>
        Print Report
      </button>
      <button class="btn btn-green" onclick="exportTableToCSV('all-runs', 'benchmark_all_runs.csv')">
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/><path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/></svg>
        Export CSV
      </button>
      <button class="btn btn-purple" onclick="copyTableToClipboard('all-runs')">
        <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/><path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/></svg>
        Copy Table
      </button>
    </div>
    
    {generate_summary_cards(rows)}
    
    {generate_accuracy_comparison_table(rows_sorted)}
    
    {generate_latency_comparison_table(rows_sorted)}
    
    {generate_latency_breakdown_table(rows_sorted)}
    
    {generate_consistency_metrics_table(rows_sorted)}
    
    {generate_cost_efficiency_table(rows_sorted)}
    
    {generate_cost_analysis_section(rows_sorted)}
    
    {generate_charts_section(rows_sorted)}
    
    {generate_full_table("Latest Runs", latest_rows, "latest-runs")}
    
    {generate_full_table("All Runs", rows_sorted, "all-runs")}
    
    {generate_workload_tables(rows_sorted)}
    
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
