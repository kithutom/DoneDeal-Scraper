# DoneDeal Used-Car Scraper

A single-file Python scraper that collects used-car ads from DoneDeal.ie and
appends them to a star-schema CSV dataset: one table of cars and one table of
price observations over time. The dataset is intended for training a model
that predicts used-car depreciation in Ireland.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Requires Python 3.10+ and two libraries: `requests` and `beautifulsoup4`.

## Usage

1. Create a text file (e.g. `urls.txt`) with one DoneDeal car ad URL per line.
   Blank lines and lines starting with `#` are ignored. See `urls_example.txt`.
2. Run:

```powershell
.venv\Scripts\python donedeal_scraper.py urls.txt
```

A single ad can also be scraped directly:

```powershell
.venv\Scripts\python donedeal_scraper.py https://www.donedeal.ie/cars-for-sale/<title>/<ad-id>
```

Or let the scraper find the ads itself: give a make and model and how many
ads to take from the top of DoneDeal's search results (default 30):

```powershell
.venv\Scripts\python donedeal_scraper.py --make Volkswagen --model Golf --count 50
```

It reads the search result pages (30 ads each, promoted "spotlight" ads left
out) and then scrapes each ad as usual. Out-of-scope ads (e.g. electric or
priced in GBP) are still skipped, so fewer than `--count` rows may be added.
Make and model are spelled as on donedeal.ie (`/cars/<Make>/<Model>`); a
misspelling finds no results.

Each ad prints one line (`OK`, `SKIPPED`, `DONE` or `FAILED` with a reason),
followed by a one-line summary. Re-running is safe: ads already recorded today
are not downloaded or written again. To track price changes, re-run the same
URL file on later days.

## Output (`data/`)

**`cars.csv`**: vehicle dimension, one row per unique ad (primary key `ad_id`)

| column | meaning |
|---|---|
| ad_id | numeric DoneDeal ad ID from the URL |
| url | ad URL without tracking parameters |
| make, model, trim | from DoneDeal's structured fields |
| year | year of manufacture |
| mileage_km | integer km |
| fuel_type | Petrol, Diesel, Hybrid, Plug-in Hybrid, Other |
| engine_size_litres | e.g. 1.6 |
| engine_power_hp | integer |
| transmission | Manual / Automatic |
| body_type | as listed by DoneDeal (Hatchback, Saloon, Estate, MPV, ...) |
| num_doors, previous_owners | integers |
| is_import | True if registered or previously registered outside Ireland |
| seller_type | Dealer / Private |
| county | e.g. Limerick |
| warranty | True / False |
| road_tax_eur | annual road tax |
| safety_rating | Euro NCAP stars (0-5) |

**`listings.csv`**: fact table, one row per ad per scrape date

| column | meaning |
|---|---|
| ad_id | foreign key to cars.csv |
| date_scraped | date of this observation |
| price_eur | asking price in euro (empty if not given) |
| price_available | False if the price is missing or on request |
| nct_months_remaining | months until NCT expiry (negative = expired) |

Missing values are left empty.

## How it works

`donedeal_scraper.py` is organised in five sections:

1. **Fetching**: checks robots.txt, waits a random 3-5 s between requests,
   and downloads the page with a browser User-Agent.
2. **Extracting**: reads the ad's data from the page's embedded
   `<script id="__NEXT_DATA__">` JSON (under `props.pageProps`).
3. **Cleaning**: converts text to numbers ("63,000 km" to 63000, "1.2 L" to 1.2,
   "150 hp" to 150), standardises fuel and seller type, calculates NCT months
   and import status.
4. **Storing**: appends to the CSVs. A car is added to `cars.csv` only the
   first time its `ad_id` is seen; `listings.csv` gets at most one row per ad
   per day.
5. **Main loop**: reads the URLs, processes each one, and prints the result.
   A failing ad is reported and skipped; it never stops the run.

Settings (output folder, delays, timeout) are constants at the top of the file.

## Scope, ethics and GDPR

* Only Republic of Ireland ads priced in EUR are kept; fully electric cars are
  skipped.
* robots.txt is respected, requests are spaced 3-5 s apart, and only public ad
  pages are read (never DoneDeal's internal API, which robots.txt disallows).
* No personal data is stored: fields are read by name, and seller names,
  phone numbers, emails and addresses are never extracted.
* If many ads fail with HTTP 429 (too many requests), increase `MIN_DELAY`
  and `MAX_DELAY` at the top of the script.
