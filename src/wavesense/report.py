"""Portable HTML inspection report with embedded interactive plots."""
import html
import json
from datetime import datetime, timezone


def export_html(result, figures):
    details = html.escape(json.dumps(result, indent=2))
    plots = ''.join(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
                    for i, fig in enumerate(figures))
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>WaveSense report</title>
    <style>body{{background:#0a1220;color:#dce7f5;font:16px system-ui;max-width:1200px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}h1{{color:#67e8f9}}</style>
    </head><body><h1>WaveSense-CFRP inspection report</h1><p>{html.escape(result['result'])} · score {result['score']:.3f}</p>
    <p>Created {datetime.now(timezone.utc).isoformat()} · Uncalibrated prototype; no defect localization.</p>
    {plots}<h2>Input, processing, model, evidence and warnings</h2><pre>{details}</pre></body></html>'''
