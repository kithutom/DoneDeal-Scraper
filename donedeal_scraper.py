"""
donedeal_scraper.py - DoneDeal used-car ad scraper.

Downloads DoneDeal car ads and appends them to two CSVs (a star schema):
    data/cars.csv      one row per unique ad          (vehicle dimension)
    data/listings.csv  one row per ad per scrape date (fact table)

Usage:
    python donedeal_scraper.py urls.txt     a file of ad URLs, one per line
    python donedeal_scraper.py <ad-url>     a single ad
    python donedeal_scraper.py --make Volkswagen --model Golf --count 50
                                            the first 50 ads of a DoneDeal search
"""

import argparse
import csv
import html
import json
import random
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from urllib import robotparser
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

# ==========================================================================
# Settings
# ==========================================================================
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
CARS_CSV = OUTPUT_DIR / "cars.csv"
LISTINGS_CSV = OUTPUT_DIR / "listings.csv"

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")
MIN_DELAY, MAX_DELAY = 3.0, 5.0     # seconds between requests (randomised)
TIMEOUT = 20                        # seconds before a request gives up
RETRY_WAIT = 60                     # seconds to pause after HTTP 429 (x attempt)
MAX_RETRIES = 3                     # 429 retries per page before giving up

CARS_COLUMNS = [
    "ad_id", "url", "make", "model", "trim", "year", "mileage_km", "fuel_type",
    "engine_size_litres", "engine_power_hp", "transmission", "body_type",
    "num_doors", "previous_owners", "is_import", "seller_type", "county",
    "warranty", "road_tax_eur", "safety_rating",
]
LISTINGS_COLUMNS = ["ad_id", "date_scraped", "price_eur", "price_available",
                    "nct_months_remaining"]


# ==========================================================================
# 1. Fetching
# ==========================================================================
session = requests.Session()
session.headers["User-Agent"] = USER_AGENT
robots = robotparser.RobotFileParser()


def load_robots():
    """Read DoneDeal's robots.txt once; if it can't be read, allow everything."""
    try:
        robots.parse(session.get("https://www.donedeal.ie/robots.txt",
                                 timeout=TIMEOUT).text.splitlines())
    except requests.RequestException:
        robots.parse([])


def retry_wait(response, attempt):
    """Seconds to back off after a 429: the server's Retry-After if it gives
    a number, otherwise RETRY_WAIT, growing with each attempt."""
    retry_after = response.headers.get("Retry-After", "")
    return int(retry_after) if retry_after.isdigit() else RETRY_WAIT * attempt


def fetch(url):
    """Politely download one page. Raises an exception saying why if it fails.
    On HTTP 429 (too many requests) it pauses and retries the same page."""
    if not robots.can_fetch(USER_AGENT, url):
        raise RuntimeError("blocked by robots.txt")
    for attempt in range(1, MAX_RETRIES + 2):
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        response = session.get(url, timeout=TIMEOUT)
        if response.status_code == 429 and attempt <= MAX_RETRIES:
            wait = retry_wait(response, attempt)
            print(f"         rate limited (HTTP 429), waiting {wait}s "
                  f"before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(wait)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        return response.text


# ==========================================================================
# 2. Extracting
# ==========================================================================
def extract(page_html):
    """
    The ad's data sits in a JSON blob: <script id="__NEXT_DATA__">, under
    props.pageProps. Returns (ad, attrs), where attrs is the spec list
    ("displayAttributes") as a simple {name: value} dictionary.
    Only named fields are ever read, so seller personal data never is (GDPR).
    """
    tag = BeautifulSoup(page_html, "html.parser").find("script", id="__NEXT_DATA__")
    if tag is None or not tag.string:
        raise RuntimeError("no __NEXT_DATA__ JSON in page")
    ad = json.loads(tag.string)["props"]["pageProps"]
    if "id" not in ad:
        raise RuntimeError("page has no ad data (removed or expired?)")
    attrs = {a["name"]: a["value"] for a in ad.get("displayAttributes", [])}
    return ad, attrs


def ad_id_from_url(url):
    """The ad ID is the number at the end of the URL: .../golf-1-5-tsi/42782499"""
    match = re.search(r"/(\d+)/?(?:[?#]|$)", url)
    return match.group(1) if match else None


# ==========================================================================
# 3. Cleaning
# ==========================================================================
def text(value):
    """Tidy a raw value; empty strings and DoneDeal's '---' placeholder -> None."""
    if value is None:
        return None
    s = html.unescape(str(value)).replace("\xa0", " ").strip()
    return s if s.strip("-") else None


def number(value):
    """First number in a string: '63,000 km' -> 63000.0, '1.2 L' -> 1.2."""
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text(value) or "")
    return float(match.group().replace(",", "")) if match else None


def integer(value):
    """'150 hp' -> 150, '5/5' -> 5."""
    n = number(value)
    return int(n) if n is not None else None


def fuel_type(value):
    """DoneDeal's fuel labels -> Petrol, Diesel, Hybrid, Plug-in Hybrid, Electric, Other."""
    s = (text(value) or "").lower()
    if not s:
        return None
    if "plug-in" in s:
        return "Plug-in Hybrid"
    for label in ("Hybrid", "Electric", "Petrol", "Diesel"):
        if label.lower() in s:
            return label
    return "Other"


def nct_months_remaining(value, today):
    """'Aug 2027' scraped in Sep 2026 -> 11. Expired NCTs are negative."""
    try:
        expiry = datetime.strptime(text(value), "%b %Y")
    except (TypeError, ValueError):
        return None
    return (expiry.year - today.year) * 12 + (expiry.month - today.month)


def is_import(country, previous_country):
    """True if registered (or previously registered) outside Ireland; None if unknown.
    DoneDeal uses 'No' in the previous-country field to mean 'not imported'."""
    current = (text(country) or "").lower()
    previous = (text(previous_country) or "").lower()
    if previous not in ("", "no", "ireland") or current not in ("", "ireland"):
        return True
    return False if current == "ireland" else None


def build_rows(ad, attrs, url, today):
    """Turn one ad into a cars.csv row and a listings.csv row."""
    ad_id = str(ad["id"])
    car = {
        "ad_id": ad_id,
        "url": url.split("?")[0],
        "make": text(attrs.get("make")),
        "model": text(attrs.get("model")),
        "trim": text(attrs.get("trim")),
        "year": integer(attrs.get("year")),
        "mileage_km": integer(attrs.get("mileage")),
        "fuel_type": fuel_type(attrs.get("fuelType")),
        "engine_size_litres": number(attrs.get("engine")),
        "engine_power_hp": integer(attrs.get("enginePower")),
        "transmission": text(attrs.get("transmission")),
        "body_type": text(attrs.get("bodyType")),
        "num_doors": integer(attrs.get("numDoors")),
        "previous_owners": integer(attrs.get("owners")),
        "is_import": is_import(attrs.get("country"),
                               attrs.get("countryOfPreviousRegistration")),
        "seller_type": {"PRO": "Dealer", "PRIVATE": "Private"}.get(ad.get("sellerType")),
        "county": text(ad.get("county")),
        "warranty": any(m.get("field") == "warranty" for m in ad.get("meritsV2") or []),
        "road_tax_eur": integer(attrs.get("roadTax")),
        "safety_rating": integer(attrs.get("safetyRating")),
    }
    price = ad.get("priceEuro")
    if ad.get("priceOnRequest") or not isinstance(price, (int, float)) or price <= 0:
        price = None
    listing = {
        "ad_id": ad_id,
        "date_scraped": today.isoformat(),
        "price_eur": price,
        "price_available": price is not None,
        "nct_months_remaining": nct_months_remaining(attrs.get("NCTExpiry"), today),
    }
    return car, listing


def out_of_scope(ad, attrs):
    """Only Republic of Ireland (EUR) ads for non-electric cars are kept."""
    if ad.get("currency") not in (None, "EUR"):
        return f"priced in {ad['currency']}"
    if fuel_type(attrs.get("fuelType")) == "Electric":
        return "electric car"
    return None


# ==========================================================================
# 4. Storing
# ==========================================================================
def read_keys(path, key_columns):
    """The key values already in a CSV, e.g. {('42787197', '2026-09-24'), ...}."""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {tuple(row[c] for c in key_columns) for row in csv.DictReader(f)}


def append_row(path, columns, row):
    """Append one row, writing the header first if the file is new.
    utf-8-sig adds a BOM so Excel shows characters like '€' correctly."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ==========================================================================
# 5. Main loop
# ==========================================================================
def read_urls(source):
    """A single URL, or every non-blank, non-# line of a text file."""
    if source.startswith("http"):
        return [source]
    lines = Path(source).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def search_urls(make, model, count):
    """
    The ad URLs of the first `count` results of a DoneDeal search, e.g.
    donedeal.ie/cars/Volkswagen/Golf. Each results page holds 30 ads in the
    same __NEXT_DATA__ JSON as an ad page (props.pageProps.ads); '?start=30'
    gives the next page. Promoted 'spotlight' ads are left out because they
    repeat on every page.
    """
    base = f"https://www.donedeal.ie/cars/{quote(make)}/{quote(model)}"
    urls, start = [], 0
    while len(urls) < count:
        tag = BeautifulSoup(fetch(f"{base}?start={start}"), "html.parser").find(
            "script", id="__NEXT_DATA__")
        if tag is None or not tag.string:
            raise RuntimeError("no __NEXT_DATA__ JSON in search page")
        page = json.loads(tag.string)["props"]["pageProps"]
        ads = page.get("ads") or []
        if not ads:
            break                           # no (more) results
        for ad in ads:
            url = ad["friendlyUrl"].split("?")[0]
            if url not in urls:
                urls.append(url)
        paging = page.get("paging") or {}
        if paging.get("currentPage", 0) >= paging.get("totalPages", 0):
            break                           # that was the last page
        start = paging["nextFrom"]
    return urls[:count]


def main():
    parser = argparse.ArgumentParser(description="DoneDeal used-car ad scraper.",
                                     epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="an ad URL or a file of ad URLs")
    parser.add_argument("--make", help="search mode: car make, e.g. Volkswagen")
    parser.add_argument("--model", help="search mode: car model, e.g. Golf")
    parser.add_argument("--count", type=int, default=30,
                        help="search mode: how many ads to scrape (default 30)")
    args = parser.parse_args()
    if bool(args.source) == bool(args.make and args.model):
        parser.error("give either an ad URL / URL file, or --make and --model")

    load_robots()
    if args.source:
        urls = read_urls(args.source)
    else:
        urls = search_urls(args.make, args.model, args.count)
        print(f"Search found {len(urls)} {args.make} {args.model} ads.\n")
        if not urls:
            sys.exit("No results - check the make and model spelling on donedeal.ie.")

    today = date.today()
    known_cars = {key[0] for key in read_keys(CARS_CSV, ["ad_id"])}
    known_listings = read_keys(LISTINGS_CSV, ["ad_id", "date_scraped"])
    new_cars = new_listings = failures = 0

    for i, url in enumerate(urls, 1):
        prefix = f"[{i}/{len(urls)}]"
        # Already recorded today? Then don't download it again.
        if (ad_id_from_url(url), today.isoformat()) in known_listings:
            print(f"{prefix} DONE     {url}: already scraped today")
            continue
        try:
            ad, attrs = extract(fetch(url))
            reason = out_of_scope(ad, attrs)
            if reason:
                print(f"{prefix} SKIPPED  {url}: {reason}")
                continue
            car, listing = build_rows(ad, attrs, url, today)

            if car["ad_id"] not in known_cars:
                append_row(CARS_CSV, CARS_COLUMNS, car)
                known_cars.add(car["ad_id"])
                new_cars += 1
            key = (listing["ad_id"], listing["date_scraped"])
            if key not in known_listings:
                append_row(LISTINGS_CSV, LISTINGS_COLUMNS, listing)
                known_listings.add(key)
                new_listings += 1
            print(f"{prefix} OK       {car['ad_id']}: {car['make']} {car['model']}, "
                  f"€{listing['price_eur']}")
        except Exception as exc:     # one bad ad must never stop the whole run
            failures += 1
            print(f"{prefix} FAILED   {url}: {exc}")

    print(f"\nDone: {new_cars} new cars, {new_listings} listings appended, "
          f"{failures} failed.")


if __name__ == "__main__":
    main()
