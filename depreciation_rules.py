"""
Association-rule mining for car depreciation.

Each listing is compared against the typical (median) price for its model year.
Listings are then turned into "transactions" of categorical items (age band,
mileage band, fuel, ...) and we mine itemsets that predict:

    * LowPrice  - bottom third of prices
    * MoreDep   - priced >15% below the expected price for its year
    * LessDep   - priced >15% above the expected price for its year
"""

import os

import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'carData.csv')


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
d = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
d = d[d.price_available & d.price_eur.notna() & (d.year <= CURRENT_YEAR)].copy()
d['age'] = CURRENT_YEAR - d.year


# ---------------------------------------------------------------------------
# Target labels
# ---------------------------------------------------------------------------
# Expected price for a given year: median price per year, smoothed with a
# 3-year rolling geometric mean so sparse years don't jump around.
median_by_year = d.groupby('year').price_eur.median().sort_index()
median_by_year = np.exp(np.log(median_by_year).rolling(3, center=True, min_periods=1).mean())

d['expected'] = d.year.map(median_by_year)
d['ratio'] = d.price_eur / d.expected

d['dep'] = pd.cut(d.ratio, [0, 0.85, 1.15, 99],
                  labels=['MoreDep', 'Normal', 'LessDep'])
d['value'] = pd.qcut(d.price_eur, 3,
                     labels=['LowPrice', 'MidPrice', 'HighPrice'])


# ---------------------------------------------------------------------------
# Build categorical items for each listing
# ---------------------------------------------------------------------------
trim = d.trim.fillna('').str.upper()
is_performance = trim.str.contains(r'^(?:GTI|R$|GTD|GTE|EDITION-R|R-LINE|GTI )') | trim.isin(['R'])

items = pd.DataFrame(index=d.index)
items['Age'] = pd.cut(d.age, [-1, 3, 7, 11, 99],
                      labels=['0-3y', '4-7y', '8-11y', '12y+']).astype(str)
items['Km'] = pd.cut(d.mileage_km, [-1, 50e3, 100e3, 150e3, 200e3, 1e7],
                     labels=['<50k', '50-100k', '100-150k', '150-200k', '200k+']).astype(str)
items['Fuel'] = d.fuel_type.replace({'Plug-in Hybrid': 'Hybrid'})
items['Trans'] = d.transmission.fillna('?')
items['Body'] = np.where(d.body_type == 'Estate', 'Estate',
                np.where(d.body_type == 'Hatchback', 'Hatch', 'Other'))
items['Owners'] = pd.cut(d.previous_owners, [-1, 1, 3, 99],
                         labels=['0-1', '2-3', '4+']).astype(str)
items['Import'] = d.is_import.map({True: 'Yes', False: 'No'}).fillna('?')
items['Seller'] = d.seller_type.fillna('?')
items['Warranty'] = d.warranty.map({True: 'Yes', False: 'No'})
items['NCT'] = pd.cut(d.nct_months_remaining, [-99, 0, 6, 99],
                      labels=['expired', '<=6m', '>6m']).astype(str)
items['Trim'] = np.where(is_performance, 'Performance',
                np.where(trim == '', '?', 'Standard'))
items['Dublin'] = np.where(d.county == 'Dublin', 'Yes', 'No')

# One transaction per listing, e.g. {'Age=4-7y', 'Fuel=Diesel', ...}; unknowns dropped
transactions = [
    {f'{col}={val}' for col, val in row.items() if val not in ('?', 'nan')}
    for _, row in items.iterrows()
]


# ---------------------------------------------------------------------------
# Apriori-style rule mining
# ---------------------------------------------------------------------------
def mine(target_series, target, minsup=0.03, maxlen=3):
    """Find itemsets (up to `maxlen` items) whose rows are enriched for `target`.

    Returns the base rate of the target and a DataFrame of rules with
    support, confidence and lift.
    """
    n_rows = len(transactions)
    is_target = (target_series == target).values
    base_rate = is_target.mean()

    all_items = sorted(set().union(*transactions))
    candidates = [frozenset([item]) for item in all_items]
    rules = []

    for k in range(1, maxlen + 1):
        frequent = []
        for itemset in candidates:
            matches = np.array([itemset <= t for t in transactions])
            n_match = matches.sum()
            if n_match / n_rows < minsup:
                continue
            frequent.append(itemset)

            n_hit = (matches & is_target).sum()
            if n_hit / n_rows >= minsup / 2:
                conf = n_hit / n_match
                rules.append((
                    ' & '.join(sorted(itemset)),
                    n_match,
                    round(n_hit / n_rows, 3),
                    round(conf, 2),
                    round(conf / base_rate, 2),
                ))

        # Next level: join frequent sets, never combining two values of the same attribute
        candidates = list({
            a | b
            for a in frequent for b in frequent
            if len(a | b) == k + 1
            and len({x.split('=')[0] for x in a | b}) == k + 1
        })

    rules = pd.DataFrame(rules, columns=['antecedent', 'n', 'support', 'conf', 'lift'])
    return base_rate, rules


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 90)

TARGETS = [
    (d.value, 'LowPrice'),
    (d.dep, 'MoreDep'),
    (d.dep, 'LessDep'),
]

for target_series, target in TARGETS:
    base_rate, rules = mine(target_series, target)

    print()
    print('=' * 80)
    print(f'  => {target}   (base rate {base_rate:.2f})')
    print('=' * 80)

    print('\nTop rules (lift >= 1.4, n >= 25):')
    strong = rules[(rules.lift >= 1.4) & (rules.n >= 25)]
    print(strong.sort_values(['lift', 'support'], ascending=False).head(25).to_string(index=False))

    print('\nSingle items:')
    singles = rules[~rules.antecedent.str.contains('&')]
    print(singles.sort_values('lift', ascending=False).to_string(index=False))
