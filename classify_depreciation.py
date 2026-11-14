"""
classify_depreciation.py - predict whether a VW Golf will be worth more or
less than other Golfs of the same age, now or some years from now.

Reads  data/carData_clean.csv  (made by clean_data.py)
Writes results/                (numbers, rules and figures for the report)

Steps:
    1. Prepare       - a class label and attributes for every car
    2. Split         - 80% training cars, 20% test cars (never seen in training)
    3. Choose depth  - 5-fold cross-validation on the training cars only
    4. Train + test  - decision tree (Gini), decision tree (information gain),
                       Naive Bayes, and a baseline that always guesses 'Normal'
    5. Repeat        - 30 different random splits, to check the result is stable
    6. Explain       - attribute importance, rules, example cars, misses
    7. Value kept    - median price of each age band as a share of a nearly new Golf
    8. Forecast      - move a car forward in time and classify the future car

The classifiers themselves are in tdidt.py; the charts are in figures.py.

Usage:
    python classify_depreciation.py
"""

import os

import numpy as np
import pandas as pd

import figures
from tdidt import DecisionTree, NaiveBayes, number

CURRENT_YEAR = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH = os.path.join(HERE, 'data', 'carData_clean.csv')
LIFT_PATH = os.path.join(HERE, 'data', 'attribute_contribution.csv')
OUT_DIR = os.path.join(HERE, 'results')
FIG_DIR = os.path.join(OUT_DIR, 'figures')

CLASSES = ['Faster', 'Normal', 'Slower']
BAND = 0.15            # >15% below similar cars of its age = Faster, >15% above = Slower
TEST_SHARE = 0.20
SEED = 2026
DEPTHS = range(1, 13)  # tree depths tried
MIN_LEAF = 10          # no leaf may hold fewer than 10 training cars
FOLDS = 5
REPEATS = 30
AGE_BANDS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 99)]

# Attributes that describe the ad rather than the car. Nobody knows them years
# ahead, so the forecast tree is trained without them.
AD_ATTRIBUTES = ['seller', 'warranty', 'nct', 'county']

# Performance Golfs cost far more new, so they are compared with each other
PERFORMANCE_TRIMS = {'GTI', 'GTI Performance', 'GTI Clubsport', 'GTI TCR',
                     'GTD', 'GTE', 'R', 'Edition-R'}
# For trim = Unknown, the ad title (in the URL) often names the version
PERFORMANCE_IN_TITLE = r'(?:^|-)(?:gti|gtd|gte|r32|clubsport|tcr)(?:-|$)|golf-r(?:-(?!line)|$)'

TRIM_LEVEL = {
    'Trendline': 'Base', 'BlueMotion': 'Base', 'Life': 'Base',
    'Comfortline': 'Mid', 'Comfort': 'Mid', 'Match': 'Mid', 'Lounge': 'Mid',
    'Allstar': 'Mid', 'Edition-40': 'Mid',
    'Highline': 'High', 'Style': 'High', 'R-Line': 'High', 'Edition 75': 'High',
    '50 YEARS': 'High', 'Launch Edition': 'High',
}

NAMES = {
    'mileage_km': 'Mileage (km)',
    'km_per_year': 'Km per year',
    'age': 'Age (years)',
    'previous_owners': 'Previous owners',
    'engine_size': 'Engine size (L)',
    'trim_level': 'Trim level',
    'fuel': 'Fuel',
    'transmission': 'Transmission',
    'body': 'Body type',
    'seller': 'Seller type',
    'warranty': 'Warranty',
    'nct': 'NCT',
    'county': 'County',
}

# Workshop attribute (data/attribute_contribution.csv) -> attribute used here
WORKSHOP_ATTRIBUTE = {
    'Seller type': 'seller', 'Year': 'age', 'Mileage >50% apart (vs <10%)': 'mileage_km',
    'Warranty': 'warranty', 'County': 'county', 'NCT (>=6 months apart)': 'nct',
    'Transmission': 'transmission', 'Trim': 'trim_level',
    'Previous owners (>=2 apart)': 'previous_owners',
    'Body type': 'body', 'Engine size': 'engine_size',
}

# Made-up cars to show what the model predicts
EXAMPLES = [
    dict(car='2017 1.6 TDI Trendline, manual, 210,000 km, 3 owners, private, NCT expired',
         year=2017, mileage_km=210000, previous_owners=3, engine_size=1.6, trim_level='Base',
         fuel='Diesel', transmission='Manual', body='Hatchback', seller='Private',
         warranty='No', nct='Expired', county='Outside Dublin'),
    dict(car='2015 1.6 TDI Highline, manual, 160,000 km, 2 owners, dealer with warranty',
         year=2015, mileage_km=160000, previous_owners=2, engine_size=1.6, trim_level='High',
         fuel='Diesel', transmission='Manual', body='Hatchback', seller='Dealer',
         warranty='Yes', nct='Over 6 months', county='Outside Dublin'),
    dict(car='2021 1.5 TSI Life, automatic, 45,000 km, 1 owner, dealer with warranty',
         year=2021, mileage_km=45000, previous_owners=1, engine_size=1.5, trim_level='Base',
         fuel='Petrol', transmission='Automatic', body='Hatchback', seller='Dealer',
         warranty='Yes', nct='Over 6 months', county='Dublin'),
    dict(car='2019 GTI 2.0, automatic, 60,000 km, 1 owner, dealer with warranty',
         year=2019, mileage_km=60000, previous_owners=1, engine_size=2.0, trim_level='Performance',
         fuel='Petrol', transmission='Automatic', body='Hatchback', seller='Dealer',
         warranty='Yes', nct='Over 6 months', county='Dublin'),
]

output = []


def say(*lines):
    """Print and keep for results.txt."""
    for line in lines:
        print(line)
        output.append(str(line))


def heading(title):
    say('', '=' * 78, title, '=' * 78)


# ---------------------------------------------------------------------------
# 1. Prepare: class label and attributes
# ---------------------------------------------------------------------------
def year_price(d):
    """Typical price of a Golf of each year, per group (Regular or Performance):
    the median price per year, smoothed with a 3-year rolling geometric mean so
    years with few cars don't jump around. Returns {group: Series indexed by year}."""
    curves = {}
    for name, g in d.groupby('group'):
        median = g.groupby('year').price_eur.median().sort_index()
        curves[name] = np.exp(np.log(median).rolling(3, center=True, min_periods=1).mean())
    return curves


def trim_premium(d, curves):
    """How much each trim level usually sells above or below the typical price
    for its year: the median of price / typical price over the cars of that
    trim level. A Highline cost more new than a Trendline, so it is compared
    with what Highlines usually fetch, not with all Golfs."""
    base = pd.Series(np.nan, index=d.index)
    for name, g in d.groupby('group'):
        base.loc[g.index] = g.year.map(curves[name]).to_numpy()
    return (d.price_eur / base).groupby(d.trim_level).median()


def expected_price(d, curves=None, premium=None):
    """Typical price of a Golf of the same year, group and trim level."""
    curves = year_price(d) if curves is None else curves
    expected = pd.Series(np.nan, index=d.index)
    for name, g in d.groupby('group'):
        expected.loc[g.index] = g.year.map(curves[name]).to_numpy()
    if premium is not None:
        expected = expected * d.trim_level.map(premium)
    return expected


def nct_band(months):
    """NCT months remaining -> band; 'N/A' (car too new) and 'Unknown' kept apart."""
    numeric = pd.to_numeric(months, errors='coerce')
    band = pd.Series('Unknown', index=months.index)
    band[months == 'N/A'] = 'Not due yet'
    band[numeric <= 0] = 'Expired'
    band[(numeric > 0) & (numeric <= 6)] = '1-6 months'
    band[numeric > 6] = 'Over 6 months'
    return band


def attributes(d):
    """Attributes the classifiers may use. Price is not one of them: the label
    is built from it. Import status is left out: 'Unknown' (the global constant
    from cleaning) turned out to mark dealer ads, not a property of the car. Age stays in, although the label already compares cars of
    the same year, because it raised cross-validated accuracy from 67.3% to 69.3%
    (prices of older cars spread out more, so they are less often 'Normal')."""
    X = pd.DataFrame(index=d.index)
    X['mileage_km'] = d.mileage_km.astype(float)
    X['km_per_year'] = d.mileage_km / (d.age + 0.5)   # a car from this year is ~6 months old
    X['age'] = d.age.astype(float)
    X['previous_owners'] = d.previous_owners.astype(float)
    X['engine_size'] = d.engine_size_litres.astype(float).round(1)   # 0.999 -> 1.0
    X['trim_level'] = d.trim_level
    X['fuel'] = d.fuel_type.replace({'Plug-in Hybrid': 'Hybrid'})
    X['transmission'] = d.transmission
    X['body'] = d.body_type.where(d.body_type.isin(['Hatchback', 'Estate']), 'Other')
    X['seller'] = d.seller_type
    X['warranty'] = np.where(d.warranty.astype(str) == 'True', 'Yes', 'No')
    X['nct'] = nct_band(d.nct_months_remaining.astype(str))
    X['county'] = np.where(d.county == 'Dublin', 'Dublin', 'Outside Dublin')
    return X


def bands(X):
    """Naive Bayes needs categorical attributes, so numeric ones become bands."""
    B = X.copy()
    B['mileage_km'] = pd.cut(X.mileage_km, [-1, 50e3, 100e3, 150e3, 200e3, np.inf],
                             labels=['<50k', '50-100k', '100-150k', '150-200k', '200k+']).astype(str)
    B['km_per_year'] = pd.cut(X.km_per_year, [-1, 10e3, 15e3, 20e3, 25e3, np.inf],
                              labels=['<10k', '10-15k', '15-20k', '20-25k', '25k+']).astype(str)
    B['age'] = pd.cut(X.age, [-1, 3, 7, 11, np.inf],
                      labels=['0-3', '4-7', '8-11', '12+']).astype(str)
    B['previous_owners'] = pd.cut(X.previous_owners, [-1, 1, 3, np.inf],
                                  labels=['0-1', '2-3', '4+']).astype(str)
    B['engine_size'] = X.engine_size.map(lambda v: f'{v:.1f} L')
    return B


def prepare():
    d = pd.read_csv(CLEAN_PATH, encoding='utf-8-sig', keep_default_na=False)
    n_clean = len(d)
    d = d[d.year <= CURRENT_YEAR].copy()       # 2027 cars are pre-orders, not used cars
    d['age'] = CURRENT_YEAR - d.year

    title = d.url.str.extract(r'cars-for-sale/([^/]+)/', expand=False).str.lower()
    from_title = (d.trim == 'Unknown') & title.str.contains(PERFORMANCE_IN_TITLE)
    performance = d.trim.isin(PERFORMANCE_TRIMS) | from_title | (d.engine_size_litres >= 2.8)
    d['group'] = np.where(performance, 'Performance', 'Regular')
    d['trim_level'] = np.where(performance, 'Performance', d.trim.map(TRIM_LEVEL).fillna('Unknown'))

    curves = year_price(d)
    premium = trim_premium(d, curves)
    d['expected_price'] = expected_price(d, curves, premium)
    d['year_price'] = expected_price(d, curves)                  # before the trim adjustment
    d['adjusted_price'] = d.price_eur / d.trim_level.map(premium)
    d['ratio'] = d.price_eur / d.expected_price
    d['label'] = pd.cut(d.ratio, [0, 1 - BAND, 1 + BAND, np.inf], labels=CLASSES).astype(str)

    heading('1. DATA')
    say(f'Cars in carData_clean.csv: {n_clean}',
        f'Dropped {n_clean - len(d)} pre-order cars from {CURRENT_YEAR + 1} -> {len(d)} cars used',
        f'Performance Golfs (GTI, GTD, GTE, R, R32): {performance.sum()} '
        f'({from_title.sum()} found from the ad title because trim was Unknown)',
        '',
        'Trim adjustment (median of price / typical price for its year):',
        '  ' + ', '.join(f'{t} x{v:.2f}' for t, v in premium.items()),
        '',
        f'Class label: price / typical price of Golfs from the same year, group and trim level',
        f'  Faster = ratio <= {1 - BAND:.2f}   (lost more value than similar cars of its age)',
        f'  Normal = {1 - BAND:.2f} < ratio <= {1 + BAND:.2f}',
        f'  Slower = ratio > {1 + BAND:.2f}    (held its value better)',
        '')
    counts = d.label.value_counts().reindex(CLASSES)
    for c in CLASSES:
        say(f'  {c:<7} {counts[c]:>5}  ({counts[c] / len(d):.0%})')
    say('', 'Label by group (row %):')
    say(pd.crosstab(d.group, d.label, normalize='index').reindex(columns=CLASSES)
        .map(lambda v: f'{v:.0%}').to_string())

    # Why the group matters: with one typical price for all Golfs, most
    # performance cars would be 'Slower' only because they cost more new
    d_one = d.assign(group='All')
    one_label = pd.cut(d.price_eur / expected_price(d_one), [0, 1 - BAND, 1 + BAND, np.inf],
                       labels=CLASSES).astype(str)
    share = (one_label[d.group == 'Performance'] == 'Slower').mean()
    say('', f'Without the Regular/Performance split, {share:.0%} of performance Golfs would be '
            f"'Slower' just because they cost more new.")

    X = attributes(d)
    say('', f'Attributes ({X.shape[1]}): ' + ', '.join(NAMES[a] for a in X.columns))
    return d, X, curves, premium


# ---------------------------------------------------------------------------
# 2-3. Splitting and choosing the depth
# ---------------------------------------------------------------------------
def stratified_split(y, test_share, seed):
    """Random train/test split that keeps each class's share the same in both."""
    rng = np.random.default_rng(seed)
    test = []
    for c in CLASSES:
        idx = rng.permutation(y.index[y == c].to_numpy())
        test.extend(idx[:round(len(idx) * test_share)])
    is_test = y.index.isin(test)
    return y.index[~is_test], y.index[is_test]


def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    fold = pd.Series(0, index=y.index)
    for c in CLASSES:
        idx = rng.permutation(y.index[y == c].to_numpy())
        fold.loc[idx] = np.arange(len(idx)) % k
    return fold


def accuracy(actual, predicted):
    return float((actual.to_numpy() == predicted.to_numpy()).mean())


def depth_search(X, y, criterion):
    """Mean training and cross-validated accuracy for each depth. Each fold's
    tree is grown once to the largest depth and read at every smaller depth."""
    folds = stratified_folds(y, FOLDS, SEED)
    rows = []
    for f in range(FOLDS):
        fit_idx, val_idx = folds.index[folds != f], folds.index[folds == f]
        tree = DecisionTree(criterion, max_depth=max(DEPTHS), min_leaf=MIN_LEAF)
        tree.fit(X.loc[fit_idx], y.loc[fit_idx], CLASSES)
        for depth in DEPTHS:
            rows.append({'depth': depth,
                         'train_accuracy': accuracy(y.loc[fit_idx], tree.predict(X.loc[fit_idx], depth)),
                         'cv_accuracy': accuracy(y.loc[val_idx], tree.predict(X.loc[val_idx], depth)),
                         'leaves': tree.n_leaves(depth)})
    result = pd.DataFrame(rows).groupby('depth').agg(
        train_accuracy=('train_accuracy', 'mean'), cv_accuracy=('cv_accuracy', 'mean'),
        cv_sd=('cv_accuracy', 'std'), leaves=('leaves', 'mean')).reset_index()
    best = int(result.loc[result.cv_accuracy.idxmax(), 'depth'])   # ties -> shallower
    return result, best


# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------
def confusion(actual, predicted):
    return (pd.crosstab(actual.rename('Actual'), predicted.rename('Predicted'))
            .reindex(index=CLASSES, columns=CLASSES, fill_value=0))


def class_scores(cm):
    """Precision: of the cars predicted as a class, how many were that class.
    Recall: of the cars in a class, how many were found.
    F-measure: 2 x precision x recall / (precision + recall), their harmonic mean."""
    hits = np.diag(cm.to_numpy())
    precision = hits / np.maximum(cm.sum(axis=0).to_numpy(), 1)
    recall = hits / np.maximum(cm.sum(axis=1).to_numpy(), 1)
    f = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return pd.DataFrame({'precision': precision, 'recall': recall, 'f_measure': f,
                         'cars': cm.sum(axis=1).to_numpy()}, index=CLASSES)


def roc_curve(is_class, score):
    """ROC curve for one class against the rest. The test cars are ranked by the
    model's probability for the class; the threshold is lowered one distinct
    probability at a time, giving a (false positive rate, true positive rate)
    point each time. Returns fpr, tpr and the area under the curve."""
    is_class = np.asarray(is_class, dtype=bool)
    score = np.asarray(score, dtype=float)
    fpr, tpr = [0.0], [0.0]
    for t in np.unique(score)[::-1]:
        chosen = score >= t
        tpr.append((chosen & is_class).sum() / is_class.sum())
        fpr.append((chosen & ~is_class).sum() / (~is_class).sum())
    fpr, tpr = np.array(fpr), np.array(tpr)
    area = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, area


def fit_models(X, B, y, train, depth_gini, depth_entropy):
    majority = y.loc[train].value_counts().idxmax()
    return {
        'Decision tree (Gini)': DecisionTree('gini', depth_gini, MIN_LEAF).fit(X.loc[train], y.loc[train], CLASSES),
        'Decision tree (information gain)': DecisionTree('entropy', depth_entropy, MIN_LEAF).fit(X.loc[train], y.loc[train], CLASSES),
        'Naive Bayes': NaiveBayes().fit(B.loc[train], y.loc[train], CLASSES),
        f"Baseline (always '{majority}')": majority,
    }


def predict(model, X, B, idx):
    if isinstance(model, str):
        return pd.Series(model, index=idx)
    if isinstance(model, NaiveBayes):
        return model.predict(B.loc[idx])
    return model.predict(X.loc[idx])


# ---------------------------------------------------------------------------
# 7. Value kept by age
# ---------------------------------------------------------------------------
def value_kept(d):
    """Median price of each age band, as a share of the median price of Golfs
    0-1 years old in the same group. The data is one snapshot, so older cars
    stand in for the same car later."""
    rows = []
    # Performance Golfs are left out: only a handful are 0-1 years old, so
    # there is no reliable 'nearly new' price to compare the older ones with
    for name, g in [('All Golfs', d), ('Regular', d[d.group == 'Regular'])]:
        new = g[g.age <= AGE_BANDS[0][1]].price_eur.median()
        for low, high in AGE_BANDS:
            band = g[(g.age >= low) & (g.age <= high)]
            if len(band) < 5:
                continue
            rows.append({'group': name, 'age_band': f'{low}-{high}' if high < 99 else f'{low}+',
                         'mid_age': (low + min(high, low + 3)) / 2, 'cars': len(band),
                         'median_price': band.price_eur.median(),
                         'value_kept': band.price_eur.median() / new})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Forecast: move a car forward in time, then classify the future car
# ---------------------------------------------------------------------------
def car_attributes(X):
    """The attributes that belong to the car itself and can be known years ahead."""
    return X.drop(columns=AD_ATTRIBUTES)


def age_car(car, years):
    """The same car `years` from now: older, with more kilometres added at the
    rate it has been driven so far. Engine, trim, fuel, transmission, body and
    owners are kept as they are."""
    age = CURRENT_YEAR - car['year']
    rate = car['mileage_km'] / (age + 0.5)
    future = dict(car)
    future['age'] = float(age + years)
    future['mileage_km'] = float(car['mileage_km'] + rate * years)
    future['km_per_year'] = future['mileage_km'] / (future['age'] + 0.5)
    return future


def forecast(tree, car, years, curves, premium):
    """Class probabilities for the car `years` from now, and the typical price
    (in today's euro) of Golfs of that age, group and trim level."""
    future = age_car(car, years)
    row = pd.DataFrame([future])[tree.attributes]
    row = row.astype({a: float for a in tree.numeric})
    proba = tree.predict_proba(row)[0]
    group = 'Performance' if car['trim_level'] == 'Performance' else 'Regular'
    curve = curves[group]
    year = CURRENT_YEAR - int(future['age'])
    nearest = min(curve.index, key=lambda y: abs(y - year))     # oldest year we have, if older
    typical = float(curve[nearest] * premium[car['trim_level']])
    return {'age': future['age'], 'mileage_km': future['mileage_km'],
            'predicted': CLASSES[int(proba.argmax())], 'proba': proba, 'typical_price': typical,
            'beyond_data': year < curve.index.min()}


def price_range(result):
    """What the predicted class means in euro."""
    t = result['typical_price']
    return {'Faster': f'under EUR {t * (1 - BAND):,.0f}',
            'Normal': f'EUR {t * (1 - BAND):,.0f} to {t * (1 + BAND):,.0f}',
            'Slower': f'over EUR {t * (1 + BAND):,.0f}'}[result['predicted']]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    d, X, curves, premium = prepare()
    B = bands(X)
    y = d.label

    # 2. Split
    train, test = stratified_split(y, TEST_SHARE, SEED)
    heading('2. SPLIT')
    say(f'Training cars: {len(train)}   Test cars: {len(test)} (kept unseen until the end)')

    # 3. Choose the depth
    heading('3. CHOOSING THE TREE DEPTH (5-fold cross-validation on the training cars)')
    search_gini, depth_gini = depth_search(X.loc[train], y.loc[train], 'gini')
    search_entropy, depth_entropy = depth_search(X.loc[train], y.loc[train], 'entropy')
    shown = search_gini.copy()
    for col in ['train_accuracy', 'cv_accuracy', 'cv_sd']:
        shown[col] = shown[col].map(lambda v: f'{v:.1%}')
    shown['leaves'] = shown.leaves.round(1)
    say(shown.to_string(index=False))
    say('', f'Chosen depth (Gini): {depth_gini}   (information gain: {depth_entropy})',
        'Deeper trees keep fitting the training cars better, but stop improving on '
        'cars they have not seen: that gap is overfitting.')
    search_gini.to_csv(os.path.join(OUT_DIR, 'depth_search.csv'), index=False)

    # 4. Train on all training cars, test once
    heading(f'4. TEST RESULTS ({TEST_SHARE:.0%} of cars never used in training)')
    models = fit_models(X, B, y, train, depth_gini, depth_entropy)
    tree = models['Decision tree (Gini)']
    predictions = {name: predict(m, X, B, test) for name, m in models.items()}
    for name, p in predictions.items():
        say(f'  {name:<34} {accuracy(y.loc[test], p):.1%}')
    say(f'  (decision tree accuracy on its own training cars: '
        f'{accuracy(y.loc[train], tree.predict(X.loc[train])):.1%})')

    cm = confusion(y.loc[test], predictions['Decision tree (Gini)'])
    say('', 'Confusion matrix, decision tree (Gini) - rows: actual, columns: predicted', cm.to_string())
    scores = class_scores(cm)
    say('', scores.map(lambda v: f'{v:.0%}' if isinstance(v, float) else v).to_string())
    cm_nb = confusion(y.loc[test], predictions['Naive Bayes'])
    scores_nb = class_scores(cm_nb)
    say('', 'Confusion matrix, Naive Bayes', cm_nb.to_string())
    say('', scores_nb.map(lambda v: f'{v:.0%}' if isinstance(v, float) else v).to_string())
    say('', 'Average F-measure over the three classes:',
        f'  Decision tree (Gini) {scores.f_measure.mean():.1%}   Naive Bayes {scores_nb.f_measure.mean():.1%}')

    # ROC curves: each class against the other two
    proba = {'Decision tree (Gini)': tree.predict_proba(X.loc[test]),
             'Naive Bayes': models['Naive Bayes'].predict_proba(B.loc[test])}
    roc = {name: {c: roc_curve(y.loc[test] == c, p[:, i]) for i, c in enumerate(CLASSES)}
           for name, p in proba.items()}
    say('', 'Area under the ROC curve (one class against the rest; 0.5 = guessing, 1 = perfect):')
    for name, curves_ in roc.items():
        say(f'  {name:<22} ' + '   '.join(f'{c} {curves_[c][2]:.2f}' for c in CLASSES))
    pd.DataFrame([{'model': name, 'class': c, 'auc': round(v[2], 3)}
                  for name, r in roc.items() for c, v in r.items()]
                 ).to_csv(os.path.join(OUT_DIR, 'roc_auc.csv'), index=False)

    # 5. Repeat with other random splits
    heading(f'5. STABILITY: {REPEATS} DIFFERENT RANDOM SPLITS (depths fixed as above)')
    rows = []
    for seed in range(REPEATS):
        tr, te = stratified_split(y, TEST_SHARE, seed)
        for name, m in fit_models(X, B, y, tr, depth_gini, depth_entropy).items():
            name = 'Baseline (always majority class)' if isinstance(m, str) else name
            rows.append({'model': name, 'seed': seed, 'accuracy': accuracy(y.loc[te], predict(m, X, B, te))})
    repeats = pd.DataFrame(rows)
    summary = (repeats.groupby('model', sort=False).accuracy
               .agg(['mean', 'std', 'min', 'max']).reset_index())
    say(summary.to_string(index=False, formatters={c: '{:.1%}'.format for c in ['mean', 'std', 'min', 'max']}))
    beats = (repeats.pivot(index='seed', columns='model', values='accuracy'))
    tree_wins = (beats['Decision tree (Gini)'] > beats['Naive Bayes']).sum()
    say('', f'Decision tree (Gini) beat Naive Bayes on {tree_wins} of {REPEATS} splits.')
    summary.to_csv(os.path.join(OUT_DIR, 'model_comparison.csv'), index=False)

    # Remove one attribute at a time: how much accuracy does each really add?
    heading(f'5b. ACCURACY LOST WHEN ONE ATTRIBUTE IS REMOVED (same {REPEATS} splits, Gini tree)')
    full = repeats[repeats.model == 'Decision tree (Gini)'].set_index('seed').accuracy
    rows = []
    for a in X.columns:
        Xa = X.drop(columns=a)
        acc = []
        for seed in range(REPEATS):
            tr, te = stratified_split(y, TEST_SHARE, seed)
            t = DecisionTree('gini', depth_gini, MIN_LEAF).fit(Xa.loc[tr], y.loc[tr], CLASSES)
            acc.append(accuracy(y.loc[te], t.predict(Xa.loc[te])))
        rows.append({'attribute_removed': NAMES[a], 'accuracy': np.mean(acc),
                     'change': np.mean(acc) - full.mean()})
    removal = pd.DataFrame(rows).sort_values('change')
    say(f'All attributes: {full.mean():.1%}',
        removal.to_string(index=False, formatters={'accuracy': '{:.1%}'.format,
                                                   'change': '{:+.1%}'.format}),
        'A small change means the other attributes carry the same information.')
    removal.to_csv(os.path.join(OUT_DIR, 'attribute_removal.csv'), index=False)

    # 6. Explain the model
    heading('6. WHAT THE TREE LEARNED')
    importance = tree.importance()
    say('Attribute importance (share of the total Gini drop):')
    for a, v in importance.items():
        say(f'  {NAMES[a]:<17} {v:6.1%}')
    importance.rename(index=NAMES).rename('importance').to_csv(
        os.path.join(OUT_DIR, 'feature_importance.csv'), index_label='attribute')

    lift = pd.read_csv(LIFT_PATH)
    lift = lift[lift.attribute_differs.isin(WORKSHOP_ATTRIBUTE)]
    compare = pd.DataFrame({
        'attribute': lift.attribute_differs.map(WORKSHOP_ATTRIBUTE).map(NAMES),
        'workshop_lift': lift.lift.to_numpy(),
        'tree_importance': lift.attribute_differs.map(WORKSHOP_ATTRIBUTE).map(importance).to_numpy(),
    }).sort_values('workshop_lift', ascending=False)
    say('', 'Workshop lift (similar-car pairs, price gap) vs tree importance:',
        compare.to_string(index=False, formatters={'tree_importance': '{:.1%}'.format}),
        '(Lift looked at one attribute at a time. The tree credits an attribute only for what',
        ' it adds after the others, so attributes that overlap share or lose credit.)')

    # Why import status is not an attribute: 'Unknown' marks dealer ads
    imported = d.is_import.astype(str).map({'True': 'Yes', 'False': 'No'}).fillna('Unknown')
    unknown = imported == 'Unknown'
    say('', "Import status was left out. 'Unknown' (the global constant from cleaning):",
        f'  {unknown.sum()} cars; {(X.seller[unknown] == "Dealer").mean():.0%} dealer ads '
        f'(all cars: {(X.seller == "Dealer").mean():.0%})',
        '  class mix: ' + ', '.join(f'{c} {v:.0%}' for c, v in
                                    y[unknown].value_counts(normalize=True).reindex(CLASSES).items())
        + '   (known: ' + ', '.join(f'{c} {v:.0%}' for c, v in
                                    y[~unknown].value_counts(normalize=True).reindex(CLASSES).items()) + ')',
        '  The value says how the ad was filled in, not what the car is, so a tree',
        '  that used it would be learning the ad form (Han et al., p.88).')

    rules = tree.rules(NAMES)
    lines = [f'Decision tree (Gini), depth {depth_gini}, {len(rules)} leaves, '
             f'trained on {len(train)} cars. Most cars first.', '']
    for r in rules:
        shares = ', '.join(f'{c} {s:.0%}' for c, s in zip(CLASSES, r['shares']))
        lines.append(f"IF {' AND '.join(r['conditions'])}")
        lines.append(f"   THEN {r['predict']}   [{r['n']} cars: {shares}]")
        lines.append('')
    with open(os.path.join(OUT_DIR, 'tree_rules.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    say('', f'Largest leaves (all {len(rules)} rules in results/tree_rules.txt):')
    for r in rules[:6]:
        shares = ', '.join(f'{c} {s:.0%}' for c, s in zip(CLASSES, r['shares']))
        say(f"  IF {' AND '.join(r['conditions'])}", f"     THEN {r['predict']}  [{r['n']} cars: {shares}]")
    say('', 'Purest Faster and Slower leaves (at least 20 cars):')
    for c in ['Faster', 'Slower']:
        i = CLASSES.index(c)
        best = max((r for r in rules if r['n'] >= 20), key=lambda r: r['shares'][i])
        say(f"  IF {' AND '.join(best['conditions'])}",
            f"     THEN {best['predict']}  [{best['n']} cars, {best['shares'][i]:.0%} {c}]")

    # Example cars
    examples = pd.DataFrame(EXAMPLES)
    examples['age'] = CURRENT_YEAR - examples.year
    examples['km_per_year'] = examples.mileage_km / (examples.age + 0.5)
    ex_X = examples[X.columns].astype({c: float for c in X.columns if X[c].dtype.kind == 'f'})
    proba = tree.predict_proba(ex_X)
    say('', 'Example cars:')
    for car, p in zip(examples.car, proba):
        say(f'  {car}', f"     -> {CLASSES[int(p.argmax())]}  ({', '.join(f'{c} {v:.0%}' for c, v in zip(CLASSES, p))})")

    # Biggest misses on the test set
    test_out = d.loc[test, ['ad_id', 'url', 'year', 'trim', 'mileage_km', 'seller_type',
                            'price_eur', 'expected_price', 'ratio', 'label']].copy()
    test_out['predicted_tree'] = predictions['Decision tree (Gini)']
    test_out['predicted_naive_bayes'] = predictions['Naive Bayes']
    test_out['expected_price'] = test_out.expected_price.round(0).astype(int)
    test_out['ratio'] = test_out.ratio.round(3)
    test_out.to_csv(os.path.join(OUT_DIR, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
    opposite = test_out[((test_out.label == 'Faster') & (test_out.predicted_tree == 'Slower'))
                        | ((test_out.label == 'Slower') & (test_out.predicted_tree == 'Faster'))]
    say('', f'Test cars put in the opposite class (Faster <-> Slower): {len(opposite)} of {len(test)}')
    for _, r in opposite.head(8).iterrows():
        say(f"  {r.ad_id}  {r.year} {r.trim:<12} {r.mileage_km:>7,} km  {r.seller_type:<7} "
            f"EUR {r.price_eur:>6,} vs typical {r.expected_price:>6,}  actual {r.label}, predicted {r.predicted_tree}")

    # Where the tree is right and wrong, by age and by class
    by_age = (test_out.assign(age_band=B.loc[test, 'age'], correct=test_out.label == test_out.predicted_tree)
              .groupby('age_band', sort=False)
              .agg(cars=('correct', 'size'), accuracy=('correct', 'mean'),
                   faster_or_slower=('label', lambda s: (s != 'Normal').mean()))
              .reindex(['0-3', '4-7', '8-11', '12+']))
    say('', 'Test accuracy by age (years), with the share of cars that are not Normal:',
        by_age.to_string(formatters={'accuracy': '{:.0%}'.format, 'faster_or_slower': '{:.0%}'.format}))

    # Class shares by attribute value (supports the rules)
    shares = []
    for a in ['seller', 'transmission', 'trim_level', 'warranty', 'fuel', 'nct',
              'county', 'body', 'mileage_km', 'km_per_year', 'age', 'previous_owners', 'engine_size']:
        t = pd.crosstab(B[a], y).reindex(columns=CLASSES, fill_value=0)
        for value, row in t.iterrows():
            shares.append({'attribute': NAMES[a], 'value': value, 'cars': int(row.sum()),
                           **{c: row[c] / row.sum() for c in CLASSES}})
    shares = pd.DataFrame(shares)
    shares.to_csv(os.path.join(OUT_DIR, 'class_by_attribute.csv'), index=False)

    # 7. Value kept by age
    heading('7. VALUE KEPT BY AGE (median price of each age band / median price at 0-1 years)')
    kept = value_kept(d)
    say(kept.drop(columns='mid_age').to_string(index=False, formatters={
        'median_price': '{:,.0f}'.format, 'value_kept': '{:.0%}'.format}))
    kept.to_csv(os.path.join(OUT_DIR, 'value_kept_by_age.csv'), index=False)

    # 8. Forecast
    heading('8. FORECAST: WILL THIS GOLF BE WORTH MORE OR LESS THAN OTHERS OF ITS AGE?')
    Xc = car_attributes(X)
    _, depth_car = depth_search(Xc.loc[train], y.loc[train], 'gini')
    car_tree = DecisionTree('gini', depth_car, MIN_LEAF).fit(Xc.loc[train], y.loc[train], CLASSES)
    acc_car = accuracy(y.loc[test], car_tree.predict(Xc.loc[test]))
    rep_acc = []
    for seed in range(REPEATS):
        tr, te = stratified_split(y, TEST_SHARE, seed)
        t = DecisionTree('gini', depth_car, MIN_LEAF).fit(Xc.loc[tr], y.loc[tr], CLASSES)
        rep_acc.append(accuracy(y.loc[te], t.predict(Xc.loc[te])))
    say('The forecast tree uses only what is known about the car years ahead:',
        '  ' + ', '.join(NAMES[a] for a in Xc.columns),
        f'  (left out, because they describe the ad: {", ".join(NAMES[a] for a in AD_ATTRIBUTES)})',
        f'Depth {depth_car}; test accuracy {acc_car:.1%}; '
        f'mean over {REPEATS} splits {np.mean(rep_acc):.1%} (all attributes: {full.mean():.1%})',
        '',
        'The car is moved forward: age + N years, mileage + N x its km per year so far.',
        'The tree, trained on Golfs that are that age today, then classifies the future car.',
        'Typical price = todays price of Golfs of that age, group and trim, in todays euro.')
    rows = []
    for ex in EXAMPLES:
        say('', f"  {ex['car']}")
        for years in [0, 3, 5]:
            r = forecast(car_tree, ex, years, curves, premium)
            when = 'today      ' if years == 0 else f'in {years} years'
            note = '  (older than most of our data)' if r['beyond_data'] else ''
            say(f"     {when}: age {r['age']:.0f}, {r['mileage_km']:>7,.0f} km -> {r['predicted']:<6} "
                f"({', '.join(f'{c} {v:.0%}' for c, v in zip(CLASSES, r['proba']))}); "
                f"typical EUR {r['typical_price']:,.0f}, so {price_range(r)}{note}")
            rows.append({'car': ex['car'], 'years_ahead': years, 'age': r['age'],
                         'mileage_km': round(r['mileage_km']), 'predicted': r['predicted'],
                         **{c: round(v, 2) for c, v in zip(CLASSES, r['proba'])},
                         'typical_price': round(r['typical_price'])})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'forecast_examples.csv'), index=False)

    # Figures
    figures.label_scatter(d, BAND, os.path.join(FIG_DIR, '1_class_label.png'))
    figures.depth_curve(search_gini, depth_gini, accuracy(y.loc[train], pd.Series(y.loc[train].value_counts().idxmax(), index=train)),
                        os.path.join(FIG_DIR, '2_depth_vs_accuracy.png'))
    figures.confusion_matrix(cm, os.path.join(FIG_DIR, '3_confusion_matrix.png'))
    figures.importance(importance.rename(index=NAMES), os.path.join(FIG_DIR, '4_attribute_importance.png'))
    figures.tree_diagram(tree, NAMES, os.path.join(FIG_DIR, '5_decision_tree.png'), levels=3)
    figures.model_comparison(summary, os.path.join(FIG_DIR, '6_model_comparison.png'))
    figures.class_shares(shares, ['Seller type', 'Transmission', 'Trim level', 'Warranty',
                                  'Km per year', 'Mileage (km)'],
                         os.path.join(FIG_DIR, '7_class_by_attribute.png'))
    figures.value_kept(kept, os.path.join(FIG_DIR, '8_value_kept_by_age.png'))
    figures.roc_curves(roc, os.path.join(FIG_DIR, '9_roc_curves.png'))

    with open(os.path.join(OUT_DIR, 'results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(output) + '\n')
    print(f'\nSaved results and figures to {OUT_DIR}')


if __name__ == '__main__':
    main()
