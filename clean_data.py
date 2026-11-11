"""
clean_data.py - clean carData.csv using the methods chosen in the project.

Reads  data/carData.csv        (raw scraper output, never modified)
Writes data/carData_clean.csv  (cleaned dataset)
       data/cleaning_log.csv   (one row per removed tuple or changed value)

Steps, in order:
    1. Fill in manually  - values recovered by reading the ad's title/description
                           (wrong prices like EUR 1 / 1234, mileages typed in
                           thousands or miles, missing engine size/transmission)
    2. Ignore the tuple  - no price / POA / placeholder price with no real price
                           in the ad, mileage, engine size or transmission missing
                           and not in the ad, and ads that are not a VW Golf
    3. Global constant   - trim = Unknown, is_import = Unknown,
                           NCT = N/A (car too new for its first NCT) or Unknown
    4. Median by year    - previous_owners filled with the median for that year
    5. Most probable     - seller_type = Dealer if the ad offers a warranty,
                           otherwise Private

Manual fills run first so a tuple whose price or mileage was recovered is kept.

Usage:
    python clean_data.py
"""

import os

import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
FIRST_NCT_AGE = 4      # Irish cars get their first NCT at 4 years old
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RAW_PATH = os.path.join(DATA_DIR, 'carData.csv')
CLEAN_PATH = os.path.join(DATA_DIR, 'carData_clean.csv')
LOG_PATH = os.path.join(DATA_DIR, 'cleaning_log.csv')

MILES_TO_KM = 1.609344


# ---------------------------------------------------------------------------
# 1. Fill in manually: values read from each ad's title and description
# ---------------------------------------------------------------------------
# {ad_id: {column: (new value, where it was found)}}
MANUAL_FIXES = {
    # Price was a placeholder; the real price is in the title/description
    43084650: {'price_eur': (6950, 'title and description: "Price EUR 6950"')},
    43002198: {'price_eur': (6500, 'title and description: "EUR 6500"')},
    42948849: {'price_eur': (2450, 'title and description: "EUR 2450"')},
    43089196: {'price_eur': (24500, 'description: "Price is EUR 24500"')},
    43088254: {'price_eur': (1199, 'title and description: "EUR 1,199 ONO"')},

    # Mileage missing from the spec list but given in the ad
    43062218: {'mileage_km': (26085, 'mileage field now filled in on the ad: 26,085 km')},
    43079155: {'mileage_km': (164450, 'mileage field now filled in on the ad: 164,450 km')},
    43033339: {'mileage_km': (188000, 'description: "188kmh" (188k km)')},
    42868060: {'mileage_km': (round(220000 * MILES_TO_KM), 'description: "220miles" (220k miles)')},
    43084676: {'mileage_km': (76000, 'description: "76K Kms total mileage"'),
               'transmission': ('Automatic', 'title and description: "Automatic Transmission"')},
    42788971: {'mileage_km': (160000, 'description: "160,000kms"')},
    42997097: {'mileage_km': (141289, 'description: "141,289 km"'),
               'transmission': ('Manual', 'description: "Manual"')},
    43092546: {'mileage_km': (round(197000 * MILES_TO_KM), 'description: "197,000 miles"')},

    # Fake / mistyped mileage (thousands typed as units, or a wrong number)
    43018623: {'mileage_km': (259000, 'listed 259 km; description: "161k miles" = 259k km')},
    43065639: {'mileage_km': (204000, 'listed 204 km; description: "204,000 km"')},
    43058095: {'mileage_km': (193000, 'listed 193 km; description: "120 miles" (120k miles = 193k km)')},
    43077895: {'mileage_km': (140000, 'listed 140 km; description: "140 kms" on a 2018 car')},
    43077085: {'mileage_km': (238000, 'listed 238 km; description: "238KM mileage" on a 2012 car')},
    42920636: {'mileage_km': (312000, 'listed 312 km; description: "312k"')},
    42960767: {'mileage_km': (370000, 'listed 370 km; description: "230k miles" = 370k km')},
    42991343: {'mileage_km': (275000, 'listed 275 km; description: "275km" on a 2017 car')},
    42981125: {'mileage_km': (320000, 'listed 320 km; description: "320km on clock", serviced every 15,000k')},
    43089017: {'mileage_km': (187000, 'listed 187 km; description gives services at "133km", "169km" (thousands)')},
    43074940: {'mileage_km': (61153, 'listed 24,500 km (= the price); title and description: "61,153km"')},

    # Missing engine size / transmission found in the ad
    40548574: {'engine_size_litres': (2.0, 'title: "19 VW GOLF 2.0 TDI DSG"')},
    42875516: {'transmission': ('Automatic', 'transmission field now filled in on the ad; description: "DSG"')},
    42806844: {'transmission': ('Manual', 'description: "6 speed manual"')},
    42094895: {'transmission': ('Manual', 'description mentions a short shifter; checked by the group: manual')},
}


# ---------------------------------------------------------------------------
# 2. Ignore the tuple: rows the price or mileage cannot be recovered for
# ---------------------------------------------------------------------------
# {ad_id: reason}. Rows with no price at all (price_available False) are
# removed automatically on top of these.
DROP = {
    # Placeholder price, no real price anywhere in the ad
    41949759: 'price EUR 123,456 is a placeholder; description says POA',
    42955133: 'price EUR 1,234 is a placeholder; no price in the description',
    43089557: 'price EUR 1 is a placeholder; car is for breaking, no price given',
    43025613: 'price EUR 12,345,678 is a placeholder; no price in the description',

    # Not a VW Golf
    42987464: 'ad is for a VW Passat, not a Golf',

    # Mileage missing (or fake) and not given in the description
    42971764: 'mileage missing and not in the description',
    42708691: 'mileage missing and not in the description',
    32749557: 'mileage missing and not in the description',
    43086990: 'mileage missing and not in the description',
    43048715: 'mileage missing and not in the description',
    42726026: 'mileage missing and not in the description',
    43076494: 'mileage missing and not in the description',
    43076493: 'mileage missing and not in the description',
    42091193: 'mileage 1,234 km is a placeholder; not in the description',
    43007309: 'mileage 200 km is fake on a 2014 car; not in the description',
    42985482: 'mileage 200 km is fake on a 2011 car; not in the description',
    43090330: 'mileage 224 km is fake on a 2009 car; not in the description',
    41762506: 'mileage 175 km is fake on a 2016 car; not in the description',

    # Engine size or transmission missing and not recoverable from the ad
    42980316: 'engine size missing; ad only says "1.tsi" (could be 1.4 or 1.5)',
    42629967: 'engine size missing; eTSI could be 1.0 or 1.5',
    43075863: 'transmission missing and not in the description',
}


# ---------------------------------------------------------------------------
# 3. Global constant: trim values that are not real trims
# ---------------------------------------------------------------------------
INVALID_TRIMS = {
    'all',       # DoneDeal's "all trims" placeholder
    'e-GOLF',    # the electric e-Golf; every such row is a petrol TSI car
}

# Same trim written in different cases -> one spelling
TRIM_SPELLING = {
    'HIGHLINE': 'Highline',
    'TRENDLINE': 'Trendline',
    'COMFORTLINE': 'Comfortline',
    'BLUEMOTION': 'BlueMotion',
    'MATCH': 'Match',
    'EDITION-R': 'Edition-R',
    'GTI CLUBSPORT': 'GTI Clubsport',
}


log = []


def record(ad_id, column, old, new, method, reason):
    log.append({'ad_id': ad_id, 'column': column, 'old_value': old,
                'new_value': new, 'method': method, 'reason': reason})


def set_value(d, ad_id, column, new, method, reason):
    old = d.at[ad_id, column]
    if pd.isna(old) and pd.isna(new) or old == new:
        return
    d.at[ad_id, column] = new
    record(ad_id, column, old, new, method, reason)


def fill_column(d, mask, column, values, method, reason):
    """Fill d[column] where mask is True, logging every change. `values` is a
    constant or a Series aligned with d."""
    for ad_id in d.index[mask]:
        new = values[ad_id] if isinstance(values, pd.Series) else values
        set_value(d, ad_id, column, new, method, reason)


def main():
    d = pd.read_csv(RAW_PATH, encoding='utf-8-sig', dtype={'is_import': object})
    d = d.set_index('ad_id', drop=False)
    for col in ['trim', 'is_import', 'nct_months_remaining', 'seller_type', 'transmission']:
        d[col] = d[col].astype(object)
    n_raw, n_cols = d.shape

    # 1. Fill in manually
    for ad_id, fixes in MANUAL_FIXES.items():
        for column, (value, source) in fixes.items():
            set_value(d, ad_id, column, value, 'fill in manually', source)
            if column == 'price_eur':
                set_value(d, ad_id, 'price_available', True, 'fill in manually', source)

    # 2. Ignore the tuple
    drops = dict(DROP)
    for ad_id in d.index[~d.price_available | d.price_eur.isna()]:
        drops.setdefault(ad_id, 'no price given (POA / price on request)')
    for ad_id in d.index[d.mileage_km.isna()]:
        drops.setdefault(ad_id, 'mileage missing and not in the description')
    for ad_id, reason in drops.items():
        record(ad_id, '(whole tuple)', '', '', 'ignore the tuple', reason)
    d = d.drop(index=list(drops))

    # 3. Global constant
    trim = d.trim.astype(str).str.strip()
    fill_column(d, d.trim.notna() & trim.isin(TRIM_SPELLING), 'trim', trim.map(TRIM_SPELLING),
                'standardise spelling', 'same trim written in a different case')
    fill_column(d, d.trim.isna() | trim.isin(INVALID_TRIMS), 'trim', 'Unknown',
                'global constant', 'trim missing or not a real trim')
    fill_column(d, d.is_import.isna(), 'is_import', 'Unknown',
                'global constant', 'import status not listed')
    too_new = d.year > CURRENT_YEAR - FIRST_NCT_AGE
    fill_column(d, d.nct_months_remaining.isna() & too_new, 'nct_months_remaining', 'N/A',
                'global constant', f'car under {FIRST_NCT_AGE} years old: first NCT not due yet')
    fill_column(d, d.nct_months_remaining.isna() & ~too_new, 'nct_months_remaining', 'Unknown',
                'global constant', 'NCT due but not listed')

    # 4. Median by year (rounded half up; years with no known owners take the
    #    median of the nearest year that has one)
    known = d.groupby('year').previous_owners.median().dropna()
    median = pd.Series({y: known[min(known.index, key=lambda k: abs(k - y))]
                        for y in d.year.unique()})
    median = np.floor(median + 0.5).astype(int)
    fill_column(d, d.previous_owners.isna(), 'previous_owners', d.year.map(median),
                'median by year', 'previous owners not listed')
    d['previous_owners'] = d.previous_owners.astype(int)

    # 5. Most probable value
    fill_column(d, d.seller_type.isna() & d.warranty, 'seller_type', 'Dealer',
                'most probable value', 'seller type not listed; ad offers a warranty')
    fill_column(d, d.seller_type.isna() & ~d.warranty, 'seller_type', 'Private',
                'most probable value', 'seller type not listed; no warranty')

    d['mileage_km'] = d.mileage_km.astype(int)
    d['price_eur'] = d.price_eur.astype(int)
    d.to_csv(CLEAN_PATH, index=False, encoding='utf-8-sig')
    log_df = pd.DataFrame(log)
    log_df.to_csv(LOG_PATH, index=False, encoding='utf-8-sig')

    # Summary
    removed = log_df[log_df.method == 'ignore the tuple']
    changed = log_df[log_df.method != 'ignore the tuple']
    # Changes made to rows that were then removed don't count
    changed = changed[changed.ad_id.isin(d.index)]
    print(f'Tuples:  {n_raw} raw -> {len(d)} clean '
          f'({len(removed)} removed, {len(removed) / n_raw:.1%})')
    print(f'Values changed: {len(changed)} of {len(d) * n_cols:,} '
          f'({len(d)} rows x {n_cols} columns)\n')
    print(changed.groupby(['method', 'column']).size().to_string())
    print('\nRemoved because:')
    print(removed.reason.value_counts().to_string())

    left = d.isna().sum()
    if left.any():
        print('\nStill missing (needs a manual check):')
        print(left[left > 0].to_string())
        print(d.loc[d.isna().any(axis=1), ['url']].to_string())


if __name__ == '__main__':
    main()
