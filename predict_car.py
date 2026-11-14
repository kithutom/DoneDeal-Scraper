"""
predict_car.py - will this Golf be worth more or less than other Golfs of its age?

The car is moved forward N years (older, with more kilometres added at the rate
it has been driven so far). A decision tree trained on Golfs that are that age
today then says whether cars like it sell for less than (Faster), about the
same as (Normal) or more than (Slower) the typical Golf of that age.

Usage:
    python predict_car.py --year 2021 --mileage 45000 --engine 1.5 --trim Base
                          --fuel Petrol --transmission Automatic --years 5
"""

import argparse
import contextlib
import io

import classify_depreciation as cd
from tdidt import DecisionTree


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--year', type=int, required=True, help='year of the car, e.g. 2021')
    p.add_argument('--mileage', type=float, required=True, help='kilometres today')
    p.add_argument('--engine', type=float, required=True, help='engine size in litres, e.g. 1.5')
    p.add_argument('--trim', default='Unknown', choices=['Base', 'Mid', 'High', 'Performance', 'Unknown'])
    p.add_argument('--fuel', default='Petrol', choices=['Petrol', 'Diesel', 'Hybrid'])
    p.add_argument('--transmission', default='Manual', choices=['Manual', 'Automatic'])
    p.add_argument('--body', default='Hatchback', choices=['Hatchback', 'Estate', 'Other'])
    p.add_argument('--owners', type=int, default=1, help='previous owners')
    p.add_argument('--years', type=int, default=5, help='how many years ahead to look')
    args = p.parse_args()

    # Same data, labels, split and depth search as classify_depreciation.py
    with contextlib.redirect_stdout(io.StringIO()):
        d, X, curves, premium = cd.prepare()
        y = d.label
        train, _ = cd.stratified_split(y, cd.TEST_SHARE, cd.SEED)
        Xc = cd.car_attributes(X)
        _, depth = cd.depth_search(Xc.loc[train], y.loc[train], 'gini')
    tree = DecisionTree('gini', depth, cd.MIN_LEAF).fit(Xc.loc[train], y.loc[train], cd.CLASSES)

    car = dict(year=args.year, mileage_km=args.mileage, previous_owners=args.owners,
               engine_size=args.engine, trim_level=args.trim, fuel=args.fuel,
               transmission=args.transmission, body=args.body)
    meaning = {'Faster': 'worth LESS than', 'Normal': 'worth about the SAME as', 'Slower': 'worth MORE than'}
    for years in sorted({0, args.years}):
        r = cd.forecast(tree, car, years, curves, premium)
        when = 'Today' if years == 0 else f'In {years} years'
        print(f"\n{when}: {r['age']:.0f} years old, about {r['mileage_km']:,.0f} km")
        print(f"  Prediction: {r['predicted']} - {meaning[r['predicted']]} other Golfs of that age")
        print('  Confidence: ' + ', '.join(f'{c} {v:.0%}' for c, v in zip(cd.CLASSES, r['proba'])))
        print(f"  Typical Golf of that age and trim: EUR {r['typical_price']:,.0f} in today's money, "
              f"so this car: {cd.price_range(r)}")
        if r['beyond_data']:
            print('  Note: that is older than most of the Golfs in our data, so treat this as a rough guide.')


if __name__ == '__main__':
    main()
