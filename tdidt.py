"""
tdidt.py - the two classifiers from the course notes, written from scratch.

DecisionTree  Top-Down Induction of Decision Trees (TDIDT). Each node is split
              in two, using the attribute and test that give the lowest Gini
              index (or, with criterion='entropy', the highest information gain).
              Both measures use binary splits here (the notes' third split
              case: "A is discrete-valued and a binary tree must be produced"),
              so the two are compared on equal terms. Information gain is not
              given one branch per value as in the notes' worked example.
NaiveBayes    Naive Bayes for categorical attributes.

Both take a pandas DataFrame of attributes (one row per car) and a Series of
class labels.
"""

from itertools import combinations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Impurity measures
# ---------------------------------------------------------------------------
def impurity(counts, criterion='gini'):
    """Impurity of each row of an array of class counts.

    gini:     Gini(D) = 1 - sum(p_i^2)
    entropy:  Info(D) = -sum(p_i * log2(p_i))
    """
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    n = counts.sum(axis=1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
    if criterion == 'gini':
        return 1.0 - (p ** 2).sum(axis=1)
    logs = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logs).sum(axis=1)


# ---------------------------------------------------------------------------
# Decision tree (TDIDT)
# ---------------------------------------------------------------------------
class Node:
    """One node of the tree. A node without an `attribute` is a leaf."""

    def __init__(self, counts, depth):
        self.counts = counts        # training cars of each class that reach this node
        self.depth = depth
        self.attribute = None       # attribute tested here (None for a leaf)
        self.threshold = None       # numeric test:     value <= threshold -> left
        self.left_values = None     # categorical test: value in left_values -> left
        self.right_values = None
        self.gain = 0.0             # drop in impurity achieved by this split
        self.left = None
        self.right = None

    @property
    def n(self):
        return int(self.counts.sum())

    def goes_left(self, value):
        if self.threshold is not None:
            return float(value) <= self.threshold
        return value in self.left_values


class DecisionTree:
    """TDIDT with binary splits.

    At each node, every attribute is tried:
      * numeric attribute:     test 'A <= split-point', where the split-point is
                               the midpoint of two adjacent known values
      * categorical attribute: test 'A in S' for every subset S of its values
    The test with the lowest weighted impurity of the two children is chosen.

    Growing stops when a node is pure, reaches `max_depth`, is too small to give
    two children of at least `min_leaf` cars, or no test lowers the impurity.
    Our data does not meet the notes' adequacy condition (two cars with the same
    attribute values can have different classes), so a leaf predicts the
    majority class of the training cars that reach it.

    Because splits are binary, an attribute can be tested again lower down the
    same branch (e.g. mileage at a second split-point).
    """

    def __init__(self, criterion='gini', max_depth=None, min_leaf=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    # -- training -----------------------------------------------------------
    def fit(self, X, y, classes=None):
        self.classes = list(classes) if classes is not None else sorted(y.unique())
        self.attributes = list(X.columns)
        self.numeric = {a for a in X.columns if pd.api.types.is_numeric_dtype(X[a])}
        cols = {a: X[a].to_numpy() for a in self.attributes}
        labels = y.map({c: i for i, c in enumerate(self.classes)}).to_numpy()
        self.root = self._grow(cols, labels, depth=0)
        return self

    def _grow(self, cols, labels, depth):
        node = Node(np.bincount(labels, minlength=len(self.classes)), depth)
        if (node.counts.max() == node.n
                or (self.max_depth is not None and depth >= self.max_depth)
                or node.n < 2 * self.min_leaf):
            return node

        best = self._best_split(cols, labels)
        if best is None:
            return node
        (node.attribute, node.threshold, node.left_values,
         node.right_values, node.gain) = best

        values = cols[node.attribute]
        if node.threshold is not None:
            left = values.astype(float) <= node.threshold
        else:
            left = np.isin(values, list(node.left_values))
        node.left = self._grow({a: v[left] for a, v in cols.items()}, labels[left], depth + 1)
        node.right = self._grow({a: v[~left] for a, v in cols.items()}, labels[~left], depth + 1)
        return node

    def _best_split(self, cols, labels):
        """(attribute, threshold, left_values, right_values, gain) of the best
        split, or None if no split lowers the impurity."""
        k = len(self.classes)
        parent = impurity(np.bincount(labels, minlength=k), self.criterion)[0]
        best, best_score = None, parent - 1e-9
        for a in self.attributes:
            if a in self.numeric:
                found = self._numeric_split(cols[a].astype(float), labels, k)
            else:
                found = self._categorical_split(cols[a], labels, k)
            if found is not None and found[0] < best_score:
                best_score = found[0]
                best = (a,) + found[1:]
        if best is None:
            return None
        return best + (parent - best_score,)

    def _numeric_split(self, values, labels, k):
        order = np.argsort(values, kind='stable')
        v, y = values[order], labels[order]
        n = len(y)
        one_hot = np.eye(k)[y]
        left = np.cumsum(one_hot, axis=0)[:-1]          # class counts left of each gap
        right = one_hot.sum(axis=0) - left
        n_left = np.arange(1, n)
        ok = (v[:-1] < v[1:]) & (n_left >= self.min_leaf) & (n - n_left >= self.min_leaf)
        if not ok.any():
            return None
        score = (n_left * impurity(left, self.criterion)
                 + (n - n_left) * impurity(right, self.criterion)) / n
        score[~ok] = np.inf
        i = int(np.argmin(score))
        return score[i], (v[i] + v[i + 1]) / 2, None, None

    def _categorical_split(self, values, labels, k):
        cats, codes = np.unique(values, return_inverse=True)
        m = len(cats)
        if m < 2:
            return None
        table = np.zeros((m, k))
        np.add.at(table, (codes, labels), 1)
        total = table.sum(axis=0)
        n = total.sum()
        best = None
        for size in range(1, m // 2 + 1):
            for subset in combinations(range(m), size):
                left = table[list(subset)].sum(axis=0)
                right = total - left
                n_left, n_right = left.sum(), right.sum()
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue
                score = (n_left * impurity(left, self.criterion)[0]
                         + n_right * impurity(right, self.criterion)[0]) / n
                if best is None or score < best[0]:
                    in_left = set(subset)
                    best = (score, None,
                            frozenset(cats[i] for i in range(m) if i in in_left),
                            frozenset(cats[i] for i in range(m) if i not in in_left))
        return best

    # -- using the tree -----------------------------------------------------
    def _is_leaf(self, node, depth):
        """A node is a leaf if it has no test, or if we stop reading at `depth`."""
        return node.attribute is None or (depth is not None and node.depth >= depth)

    def _reach(self, row, depth=None):
        node = self.root
        while not self._is_leaf(node, depth):
            node = node.left if node.goes_left(row[node.attribute]) else node.right
        return node

    def predict_proba(self, X, depth=None):
        """Share of each class among the training cars in the leaf each car reaches.
        `depth` reads the tree only down to that depth (same as growing it with
        max_depth=depth, since TDIDT builds the tree top-down)."""
        counts = np.array([self._reach(row, depth).counts
                           for row in X.to_dict('records')], dtype=float)
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X, depth=None):
        best = self.predict_proba(X, depth).argmax(axis=1)
        return pd.Series(np.array(self.classes)[best], index=X.index)

    def nodes(self, depth=None):
        """Every node down to `depth`, top-down."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if not self._is_leaf(node, depth):
                stack.extend([node.right, node.left])

    def n_leaves(self, depth=None):
        return sum(self._is_leaf(node, depth) for node in self.nodes(depth))

    def importance(self, depth=None):
        """Share of the tree's total impurity drop that comes from each attribute
        (each split's drop is weighted by the number of cars it splits)."""
        total = dict.fromkeys(self.attributes, 0.0)
        for node in self.nodes(depth):
            if not self._is_leaf(node, depth):
                total[node.attribute] += node.n * node.gain
        total = pd.Series(total)
        return (total / total.sum()).sort_values(ascending=False)

    def rules(self, names=None, depth=None):
        """Every leaf as an IF ... THEN rule: a list of dicts, most cars first."""
        names = names or {}
        rules = []

        def walk(node, bounds, allowed):
            if self._is_leaf(node, depth):
                rules.append({'conditions': describe(bounds, allowed),
                              'predict': self.classes[int(node.counts.argmax())],
                              'n': node.n,
                              'shares': node.counts / node.n})
                return
            a = node.attribute
            if node.threshold is not None:
                low, high = bounds.get(a, (-np.inf, np.inf))
                walk(node.left, {**bounds, a: (low, min(high, node.threshold))}, allowed)
                walk(node.right, {**bounds, a: (max(low, node.threshold), high)}, allowed)
            else:
                current = allowed.get(a)
                left = node.left_values if current is None else node.left_values & current
                right = node.right_values if current is None else node.right_values & current
                walk(node.left, bounds, {**allowed, a: left})
                walk(node.right, bounds, {**allowed, a: right})

        def describe(bounds, allowed):
            parts = []
            for a, (low, high) in bounds.items():
                name = names.get(a, a)
                if low > -np.inf and high < np.inf:
                    parts.append(f'{number(low)} < {name} <= {number(high)}')
                elif high < np.inf:
                    parts.append(f'{name} <= {number(high)}')
                else:
                    parts.append(f'{name} > {number(low)}')
            for a, values in allowed.items():
                name = names.get(a, a)
                values = sorted(values)
                if len(values) == 1:
                    parts.append(f'{name} = {values[0]}')
                else:
                    parts.append(f'{name} in {{{", ".join(values)}}}')
            return parts

        walk(self.root, {}, {})
        return sorted(rules, key=lambda r: -r['n'])


def number(x):
    """Readable split-point: 152,450 for mileages, 1.45 for engine sizes."""
    return f'{x:,.0f}' if abs(x) >= 100 else f'{x:g}'


# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------
class NaiveBayes:
    """P(C | X) is proportional to P(C) * product over attributes of P(A = x | C).

    Every attribute must be categorical, so numeric ones are put into bands
    before calling fit(). `alpha` = 1 adds one to every count (Laplace
    smoothing), so a value never seen with a class in training does not make
    that class impossible.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, classes=None):
        self.classes = list(classes) if classes is not None else sorted(y.unique())
        class_counts = y.value_counts().reindex(self.classes, fill_value=0)
        self.prior = (class_counts / len(y)).to_numpy()
        self.likelihood, self.unseen = {}, {}
        for a in X.columns:
            table = pd.crosstab(X[a], y).reindex(columns=self.classes, fill_value=0)
            m = len(table)
            self.likelihood[a] = (table + self.alpha) / (class_counts + self.alpha * m)
            self.unseen[a] = (self.alpha / (class_counts + self.alpha * m)).to_numpy()
        return self

    def predict_proba(self, X):
        log_p = np.tile(np.log(self.prior), (len(X), 1))
        for a, table in self.likelihood.items():
            p = table.reindex(X[a].to_numpy()).to_numpy()
            p = np.where(np.isnan(p), self.unseen[a], p)
            log_p += np.log(p)
        p = np.exp(log_p - log_p.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        best = self.predict_proba(X).argmax(axis=1)
        return pd.Series(np.array(self.classes)[best], index=X.index)
