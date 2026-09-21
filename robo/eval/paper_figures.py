"""Vector presentation of already authenticated paper statistics; no estimator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def render_agentic_paired(payload, source, destination):
    """Plot all declared intervals, retaining conditional support and null CIs.

    The canonical paper pipeline authenticates the full completion audit before
    calling this renderer. This function never fits or bootstraps anything.
    """
    from robo.eval.agentic_ablation import AGENTIC_UNCERTAINTY_PROTOCOL
    if payload['protocol'] != AGENTIC_UNCERTAINTY_PROTOCOL:
        raise ValueError('paired figure requires the declared statistical protocol')
    expected = [(f'{b}-{a}', m) for a, b in payload['protocol']['contrasts']
                for m in payload['protocol']['metrics']]
    if [(r['contrast'], r['metric']) for r in payload['rows']] != expected:
        raise ValueError('paired figure requires every declared contrast in order')
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {
        'build_coverage': ('Coverage', 'All planned jobs', 'Fraction difference'),
        'f1_20': ('F1@20 mm', 'Common accepted, matched jobs', 'Fraction difference'),
        'cd_cm': ('Chamfer distance', 'Common accepted, matched jobs', 'Difference (cm; lower is better)'),
        'stable_fraction': ('Stability probe', 'Common accepted, tested jobs', 'Fraction difference'),
    }
    plotted = []
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':8, 'pdf.fonttype':42,
                         'svg.fonttype':'none', 'svg.hashsalt':'simanyroom-paired-v1',
                         'axes.spines.top':False, 'axes.spines.right':False}):
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.5), layout='constrained')
        try:
            for ax, metric in zip(axes.flat, payload['protocol']['metrics']):
                rows = [(i,r) for i,r in enumerate(payload['rows']) if r['metric']==metric]
                title, support, unit = styles[metric]
                ax.set_title(title + '\n' + support, loc='left', fontsize=9, pad=9)
                ax.axvline(0, color='#7a838c', linewidth=.8, linestyle='--', zorder=0)
                labels = []
                for y,(index,row) in enumerate(rows):
                    labels.append(f"{row['contrast']}   {row['paired_jobs']}/{row['planned_jobs']}; s={row['paired_scenes']}")
                    delta = row['delta']; low, high = row['ci95']
                    if row['status']=='ESTIMATED':
                        # Percentile intervals need not contain the point estimate.
                        ax.hlines(y, low, high, color='#246b92', linewidth=2)
                        ax.vlines([low,high], y-.07, y+.07, color='#246b92', linewidth=1)
                        ax.plot(delta, y, 'o', color='#153b55', markersize=4)
                    elif row['status']=='NOT_ESTIMABLE':
                        if delta is not None:
                            ax.plot(delta, y, 'D', markerfacecolor='white', markeredgecolor='#646b73', markersize=4)
                        ax.text(.98,y,'no CI' if delta is not None else 'unmeasured',
                                transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=7,color='#646b73')
                    else:
                        raise ValueError('unknown paired interval status')
                    plotted.append({'source_row':index,'contrast':row['contrast'],'metric':metric,
                        'delta':delta,'ci95':row['ci95'],'paired_jobs':row['paired_jobs'],
                        'planned_jobs':row['planned_jobs'],'paired_scenes':row['paired_scenes'],
                        'status':row['status'],'source_fields':[f'rows[{index}].{key}' for key in
                            ('delta','ci95','paired_jobs','planned_jobs','paired_scenes','status')]})
                ax.set_yticks(range(len(rows)),labels,fontsize=7)
                ax.set_ylim(len(rows)-.5,-.5)
                ax.set_xlabel(unit,fontsize=8)
                ax.grid(axis='x',alpha=.12)
                ax.tick_params(axis='both',length=2)
            fig.suptitle('Treatment minus baseline | paired jobs/planned; s = supported scenes',fontsize=10)
            fig.savefig(destination/'agentic_paired_uncertainty.pdf',metadata={'CreationDate':None,'ModDate':None})
            fig.savefig(destination/'agentic_paired_uncertainty.svg',metadata={'Date':None})
            fig.savefig(destination/'agentic_paired_uncertainty.png',dpi=220)
        finally:
            plt.close(fig)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record={'schema_version':1,'scope':'authenticated_paired_interval_visualization','source':source,
            'protocol':payload['protocol'],'plotted_rows':plotted,'metric_recomputation':False,
            'bootstrap_repeated':False,'paper_ready':False,'headline_eligible':False,
            'limitations':'Pointwise descriptive intervals. Conditional geometry/probes are not population-wide quality or independent physics validation.',
            'matplotlib_version':matplotlib.__version__,
            'files':{p.name:{'sha256':digest(p),'size_bytes':p.stat().st_size} for p in sorted(destination.iterdir())}}
    (destination/'figure_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return {p.name:{'sha256':digest(p),'size_bytes':p.stat().st_size} for p in sorted(destination.iterdir())}
