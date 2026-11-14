"""
figures.py - the report's charts, saved as PNG files by classify_depreciation.py.

Colours: Faster = red, Normal = grey, Slower = blue (checked to stay distinct
for colour-blind readers); one blue for single-series charts.
"""

import logging
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter, MultipleLocator,  # noqa: E402
                               NullLocator, PercentFormatter)

from tdidt import number  # noqa: E402

# The fallback font has no semibold weight; matplotlib warns even when it isn't used
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

CLASSES = ['Faster', 'Normal', 'Slower']
CLASS_COLORS = {'Faster': '#e34948', 'Normal': '#a8a7a0', 'Slower': '#2a78d6'}
CLASS_TEXT = {'Faster': '#ffffff', 'Normal': '#0b0b0b', 'Slower': '#ffffff'}
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK_2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#ffffff'
BLUE_RAMP = ['#f3f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'semibold',
    'axes.titlelocation': 'left',
    'axes.titlecolor': INK,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK_2,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': AXIS,
    'ytick.color': AXIS,
    'xtick.labelcolor': INK_2,
    'ytick.labelcolor': INK_2,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'savefig.dpi': 200,
    'legend.frameon': False,
    'text.color': INK,
})


def header(fig, title, subtitle=None):
    """Title and subtitle at fixed distances (in inches) from the top edge."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.12 / h, title, fontsize=13, fontweight='semibold', va='top', color=INK)
    if subtitle:
        fig.text(0.012, 1 - 0.47 / h, subtitle, fontsize=9.5, va='top', color=INK_2)


def save(fig, path, reserve=0.85, right=1.0):
    """Lay out the plot below `reserve` inches kept free for the header."""
    fig.tight_layout(rect=[0, 0, right, 1 - reserve / fig.get_figheight()])
    fig.savefig(path)
    plt.close(fig)


def legend_y(fig, below=0.78):
    """Figure-fraction y for a legend placed `below` inches under the top edge."""
    return 1 - below / fig.get_figheight()


def euro(v, _=None):
    return f'€{v / 1000:,.0f}K' if v >= 1000 else f'€{v:,.0f}'


# ---------------------------------------------------------------------------
# 1. How the class label is made
# ---------------------------------------------------------------------------
def label_scatter(d, band, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True,
                             gridspec_kw={'width_ratios': [3, 2]})
    rng = np.random.default_rng(0)
    for ax, group in zip(axes, ['Regular', 'Performance']):
        g = d[d.group == group]
        curve = g.groupby('year').year_price.first().sort_index()
        ax.fill_between(curve.index, curve * (1 - band), curve * (1 + band),
                        color=MUTED, alpha=0.13, linewidth=0, zorder=1)
        for c in ['Normal', 'Faster', 'Slower']:
            s = g[g.label == c]
            ax.scatter(s.year + rng.uniform(-0.3, 0.3, len(s)), s.adjusted_price, s=16,
                       color=CLASS_COLORS[c], edgecolors=SURFACE, linewidths=0.6, zorder=2)
        ax.plot(curve.index, curve, color=INK, lw=2, solid_capstyle='round', zorder=3)
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(FixedLocator([1000, 2000, 5000, 10000, 20000, 50000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(euro))
        ax.set_ylim(600, 110000)
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        ax.set_xlabel('Year of the car')
        ax.set_title(f'{group} Golfs ({len(g):,} cars)' if group == 'Regular'
                     else f'Performance Golfs: GTI, GTD, GTE, R ({len(g)} cars)')
    axes[0].set_ylabel('Asking price, adjusted for trim level (log scale)')
    handles = [Line2D([], [], marker='o', ls='', ms=7, color=CLASS_COLORS[c],
                      label=f'{c}: {"more than 15% below" if c == "Faster" else "more than 15% above" if c == "Slower" else "within 15% of"} typical')
               for c in CLASSES]
    handles += [Line2D([], [], color=INK, lw=2, label='Typical price for its year'),
                Patch(color=MUTED, alpha=0.25, label='±15% band')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.005, legend_y(fig)), ncol=5,
               fontsize=8.5, handletextpad=0.3, columnspacing=1.0)
    header(fig, 'Class label: has the car lost value faster or slower than similar Golfs of its age?',
           "Each dot is one car, compared with the median price of Golfs from the same year, group and trim level.")
    save(fig, path, reserve=1.1)


# ---------------------------------------------------------------------------
# 2. Choosing the depth
# ---------------------------------------------------------------------------
def depth_curve(search, chosen, baseline, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = search.depth
    ax.fill_between(x, search.cv_accuracy - search.cv_sd, search.cv_accuracy + search.cv_sd,
                    color=BLUE, alpha=0.10, linewidth=0)
    ax.plot(x, search.train_accuracy, color=ORANGE, lw=2, marker='o', ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label='Training cars (seen)')
    ax.plot(x, search.cv_accuracy, color=BLUE, lw=2, marker='o', ms=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label='Held-out cars (cross-validation)')
    ax.axhline(baseline, color=MUTED, lw=1)
    ax.text(x.max(), baseline + 0.004, f"Always guess 'Normal' ({baseline:.0%})",
            ha='right', va='bottom', fontsize=9, color=INK_2)
    ax.set_ylim(baseline - 0.03, max(search.train_accuracy.max(), 0.8) + 0.02)
    best = search.set_index('depth').loc[chosen]
    ax.axvline(chosen, color=AXIS, lw=1, zorder=0)
    ax.annotate(f'Chosen: depth {chosen}\n{best.cv_accuracy:.1%} on held-out cars',
                xy=(chosen, best.cv_accuracy), xytext=(chosen + 0.4, best.cv_accuracy - 0.075),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=1))
    for col, label in [('train_accuracy', 'Training'), ('cv_accuracy', 'Held-out')]:
        ax.text(x.max() + 0.25, search[col].iloc[-1], label, va='center', fontsize=9, color=INK_2)
    ax.set_xlim(0.6, x.max() + 1.4)
    ax.set_xticks(list(x))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.set_xlabel('Maximum tree depth')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.02), ncol=2, fontsize=9)
    header(fig, 'Deeper trees memorise the training cars but stop improving on unseen ones',
           'Mean of 5-fold cross-validation on the training cars; shaded band = ±1 standard deviation.')
    save(fig, path)


# ---------------------------------------------------------------------------
# 3. Confusion matrix
# ---------------------------------------------------------------------------
def confusion_matrix(cm, path):
    values = cm.to_numpy()
    share = values / values.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    cmap = LinearSegmentedColormap.from_list('blues', BLUE_RAMP)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i - 0.08, f'{values[i, j]}', ha='center', va='center', fontsize=15,
                    fontweight='semibold', color=SURFACE if share[i, j] > 0.45 else INK)
            ax.text(j, i + 0.2, f'{share[i, j]:.0%} of row', ha='center', va='center', fontsize=8.5,
                    color=SURFACE if share[i, j] > 0.45 else INK_2)
    ax.set_xticks(range(3), [f'{c}' for c in CLASSES])
    ax.set_yticks(range(3), [f'{c} ({values[i].sum()})' for i, c in enumerate(CLASSES)])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, 3), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=3)
    ax.tick_params(which='minor', length=0)
    ax.set_xlabel('Predicted class', labelpad=10)
    ax.set_ylabel('Actual class (cars)', labelpad=10)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    acc = np.trace(values) / values.sum()
    header(fig, f'Decision tree on the test cars: {acc:.1%} correct',
           'Diagonal = correct. Shade = share of each actual class.')
    save(fig, path, reserve=0.8)


# ---------------------------------------------------------------------------
# 4. Attribute importance
# ---------------------------------------------------------------------------
def importance(imp, path):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(imp))
    ax.barh(y, imp.to_numpy(), height=0.55, color=BLUE)
    for yi, v in zip(y, imp.to_numpy()):
        ax.text(v + 0.006, yi, f'{v:.1%}' if v > 0 else 'not used', va='center', fontsize=9,
                color=INK_2 if v > 0 else MUTED)
    ax.set_yticks(y, imp.index)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlim(0, imp.max() * 1.15)
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Share of the tree\'s total drop in Gini impurity')
    header(fig, 'Which attributes the decision tree relies on',
           'Each split\'s drop in impurity, weighted by the number of cars it splits, summed per attribute.')
    save(fig, path)


# ---------------------------------------------------------------------------
# 5. The tree
# ---------------------------------------------------------------------------
def tree_diagram(tree, names, path, levels=3):
    """Top `levels` levels of the tree. Nodes cut off at the bottom show their
    majority class and are marked as splitting further."""
    slots = []

    def place(node, depth):
        stop = node.attribute is None or depth >= levels
        if stop:
            node._x = len(slots)
            slots.append(node)
        else:
            place(node.left, depth + 1)
            place(node.right, depth + 1)
            node._x = (node.left._x + node.right._x) / 2
        node._stop = stop

    place(tree.root, 0)
    width = len(slots)
    fig, ax = plt.subplots(figsize=(max(8, 1.85 * width), 1.55 * (levels + 1) + 0.9))
    ax.set_xlim(-0.55, width - 0.45)
    ax.set_ylim(-levels - 0.4, 0.36)
    ax.axis('off')
    box_w, box_h = 0.9, 0.62

    def question(node):
        name = names.get(node.attribute, node.attribute)
        if node.threshold is not None:
            unit = ' km' if 'km' in node.attribute else ' years' if node.attribute == 'age' else ''
            text = f'{name.split(" (")[0]} ≤ {number(node.threshold)}{unit}?'
        else:
            values = sorted(node.left_values)
            text = f'{name} = {values[0]}?' if len(values) == 1 else f'{name} in {{{", ".join(values)}}}?'
        return textwrap.fill(text, 24)

    def draw(node, depth):
        x, y = node._x, -depth
        shares = node.counts / node.n
        leaf_class = CLASSES[int(shares.argmax())]
        ax.add_patch(FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                    boxstyle='round,pad=0,rounding_size=0.06',
                                    facecolor=SURFACE, edgecolor=AXIS, linewidth=1, zorder=2))
        if node._stop:
            ax.text(x, y + 0.17, leaf_class, ha='center', va='center', fontsize=10,
                    fontweight='semibold', zorder=3)
            more = ' · splits further' if node.attribute is not None else ''
            ax.text(x, y - 0.01, f'{node.n} cars · {shares.max():.0%}{more}', ha='center',
                    va='center', fontsize=7.5, color=INK_2, zorder=3)
        else:
            ax.text(x, y + 0.12, question(node), ha='center', va='center', fontsize=8.5,
                    linespacing=1.1, zorder=3)
            ax.text(x, y - 0.11, f'{node.n} cars', ha='center', va='center', fontsize=7.5,
                    color=INK_2, zorder=3)
        # thin segmented bar of class shares
        left, bar_w, gap = x - box_w / 2 + 0.07, box_w - 0.14, 0.008
        for c, s in zip(CLASSES, shares):
            if s > 0:
                ax.add_patch(Rectangle((left, y - 0.24), max(bar_w * s - gap, 0.001), 0.05,
                                       facecolor=CLASS_COLORS[c], edgecolor='none', zorder=3))
                left += bar_w * s
        if not node._stop:
            for child, word in [(node.left, 'yes'), (node.right, 'no')]:
                ax.plot([x, child._x], [y - box_h / 2, -depth - 1 + box_h / 2],
                        color=AXIS, lw=1, zorder=1)
                mx, my = (x + child._x) / 2, (y - box_h / 2 + (-depth - 1 + box_h / 2)) / 2
                ax.text(mx, my, word, ha='center', va='center', fontsize=8, color=INK_2,
                        bbox=dict(facecolor=SURFACE, edgecolor='none', pad=1), zorder=2)
                draw(child, depth + 1)

    draw(tree.root, 0)
    handles = [Patch(color=CLASS_COLORS[c], label=c) for c in CLASSES]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.995, 0.99), ncol=3, fontsize=9,
               title='Bar = share of cars in each class', title_fontsize=8.5)
    header(fig, f'Decision tree (Gini): top {levels} levels',
           'Read from the top: answer each question to reach a box. The bold class is the prediction.')
    save(fig, path, reserve=0.75)


# ---------------------------------------------------------------------------
# 6. Model comparison
# ---------------------------------------------------------------------------
def model_comparison(summary, path):
    s = summary.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, r in s.iterrows():
        color = BLUE if r.model == 'Decision tree (Gini)' else '#a8a7a0'
        ax.plot([r['min'], r['max']], [i, i], color=color, lw=2, solid_capstyle='round')
        ax.plot(r['mean'], i, 'o', ms=9, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(r['max'] + 0.006, i, f"{r['mean']:.1%} average", va='center', fontsize=9, color=INK_2)
    ax.set_yticks(range(len(s)), s.model)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlim(s['min'].min() - 0.03, s['max'].max() + 0.1)
    ax.set_ylim(-0.6, len(s) - 0.4)
    ax.grid(axis='x')
    ax.set_axisbelow(True)
    ax.set_xlabel('Accuracy on unseen test cars')
    header(fig, 'Accuracy over 30 different random train/test splits',
           'Dot = average; line = lowest to highest.')
    save(fig, path)


# ---------------------------------------------------------------------------
# 7. Class shares by attribute value
# ---------------------------------------------------------------------------
ORDER = {
    'Trim level': ['Base', 'Mid', 'High', 'Performance', 'Unknown'],
    'Km per year': ['<10k', '10-15k', '15-20k', '20-25k', '25k+'],
    'Mileage (km)': ['<50k', '50-100k', '100-150k', '150-200k', '200k+'],
    'Previous owners': ['0-1', '2-3', '4+'],
    'Age (years)': ['0-3', '4-7', '8-11', '12+'],
}


def class_shares(shares, attributes, path):
    cols = 3
    rows = int(np.ceil(len(attributes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 2.7 * rows + 0.9))
    for ax, a in zip(axes.flat, attributes):
        t = shares[shares.attribute == a].set_index('value')
        t = t.reindex([v for v in ORDER.get(a, sorted(t.index)) if v in t.index])[::-1]
        y = np.arange(len(t))
        left = np.zeros(len(t))
        for c in CLASSES:
            ax.barh(y, t[c], left=left, height=0.62, color=CLASS_COLORS[c],
                    edgecolor=SURFACE, linewidth=1.5)
            for yi, l, v in zip(y, left, t[c]):
                if v >= 0.12:
                    ax.text(l + v / 2, yi, f'{v:.0%}', ha='center', va='center', fontsize=8,
                            color=CLASS_TEXT[c])
            left += t[c].to_numpy()
        for yi, n in zip(y, t.cars):
            ax.text(1.02, yi, f'{n} cars', va='center', fontsize=8, color=MUTED)
        ax.set_yticks(y, t.index)
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(len(t) - 5.4, len(t) - 0.4)    # same bar thickness in every panel
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        for side in ['left', 'bottom']:
            ax.spines[side].set_visible(False)
        ax.set_title(a, pad=4)
    for ax in list(axes.flat)[len(attributes):]:
        ax.axis('off')
    handles = [Patch(color=CLASS_COLORS[c], label=c) for c in CLASSES]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.005, legend_y(fig)), ncol=3, fontsize=9)
    header(fig, 'Share of cars in each class, by attribute value',
           f'All {int(shares[shares.attribute == attributes[0]].cars.sum()):,} cars.')
    save(fig, path, reserve=1.1, right=0.97)


# ---------------------------------------------------------------------------
# 8. Value kept by age
# ---------------------------------------------------------------------------
def value_kept(kept, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    styles = {'Regular': (BLUE, 'Regular Golfs')}
    for group, (color, label) in styles.items():
        g = kept[kept.group == group].reset_index(drop=True)
        x = np.arange(len(g))
        ax.plot(x, g.value_kept, color=color, lw=2, marker='o', ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        if group == 'Regular':
            for xi, v in zip(x, g.value_kept):
                ax.text(xi, v - 0.035, f'{v:.0%}', ha='center', va='top', fontsize=8.5, color=INK_2)
            ax.set_xticks(x, g.age_band)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.set_xlabel('Age of the car (years)')
    ax.set_ylabel('Median price as a share of a 0-1 year old Golf')
    header(fig, 'How much of its value a Golf keeps as it ages',
           'Regular Golfs: median asking price of each age band, divided by the median price at 0-1 years old.')
    save(fig, path)


# ---------------------------------------------------------------------------
# 9. ROC curves
# ---------------------------------------------------------------------------
def roc_curves(roc, path):
    """roc: {model: {class: (fpr, tpr, auc)}}. One panel per class."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    colors = {'Decision tree (Gini)': BLUE, 'Naive Bayes': ORANGE}
    for ax, c in zip(axes, CLASSES):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--')
        for model, curves in roc.items():
            fpr, tpr, auc = curves[c]
            ax.plot(fpr, tpr, color=colors[model], lw=2, label=f'{model} (area {auc:.2f})')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel('False positive rate')
        ax.set_title(f'{c} against the rest')
        ax.legend(loc='lower right', fontsize=8.5)
    axes[0].set_ylabel('True positive rate')
    header(fig, 'ROC curves on the test cars',
           'The closer a curve is to the top-left corner, the better. Dashed line = guessing.')
    save(fig, path)
