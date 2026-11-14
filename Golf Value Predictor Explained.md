# Golf Value Predictor Explained

A guide to the CSC1104 Group 15 project. It predicts whether a used Volkswagen Golf will be worth more or less than other Golfs of its age, today or some years from now. No earlier knowledge of the project or of data mining is assumed.

**1,291**ads scraped from DoneDeal.ie

**1,246**cars used after cleaning

**72.7%**of unseen test cars classified correctly

**58.6%**is what plain guessing would score

1. [The project in one minute](#big-picture)
2. [Part 1: every keyword](#glossary)
3. [Part 1: every file and function](#functions)
4. [Part 2: how the system works, step by step](#how)
5. [What the results say](#results)
6. [What the project cannot tell you](#limits)

## The project in one minute

The module asks each group to build a dataset and use it to make a prediction with a classification method. This group collected adverts for used VW Golfs in Ireland and asked one question about each car: **is it worth less than, about the same as, or more than a typical Golf of the same age?**

Every car gets one of three labels:

- Faster: priced more than 15% below typical. It has lost value faster than its peers.
- Normal: within 15% of typical.
- Slower: priced more than 15% above typical. It has held its value better.

A program then learns, from 80% of the cars, which details of a car (mileage, age, engine, trim and so on) go with each label. It is tested on the other 20%, which it has never seen. Finally, the same program can take any Golf, move it forward a few years (older, more kilometres) and say which label cars like that carry today. That is the forecast.

**donedeal_scraper.py**\
Downloads each advert and saves one row per car.

↓ data/carData.csv (1,291 rows)

**clean_data.py**\
Fixes wrong values, fills gaps, removes rows that cannot be used.

↓ data/carData_clean.csv (1,250 rows)

**classify_depreciation.py** with **tdidt.py**\
Labels each car, trains and tests the decision tree and Naive Bayes, and forecasts.

↓ results/ (numbers, rules, tables)

**figures.py** and **predict_car.py**\
Draw the charts, and forecast one car you type in.

Part 1

## Every keyword

Terms are grouped by where they appear. Each one is explained in plain words, then tied to how this project uses it.

### Data words

Web scraping

Having a program read web pages and copy out the facts, instead of a person doing it by hand. Here the pages are car adverts on DoneDeal.ie.

Dataset

The table of collected facts. Each row is one car advert and each column is one fact about it.

Tuple

The textbook word for one row of the table. One tuple is one car.

Attribute

The textbook word for one column, such as mileage or fuel type. Attributes are what the model is allowed to look at.

Categorical attribute

An attribute whose values are names with no order, such as fuel (Petrol, Diesel, Hybrid).

Numeric (continuous) attribute

An attribute whose values are numbers you can sort, such as mileage or engine size.

Class label

The answer the model must predict for each car. Here it is Faster, Normal or Slower.

CSV

A plain text file that stores a table, with commas between the columns. It opens in Excel.

Primary key

The column that identifies each row uniquely. Here it is `ad_id`, the number at the end of the advert's web address.

Flat table

One single table holding everything. The project's data is a flat table, with one row per advert.

NCT

The National Car Test, Ireland's roadworthiness test. A car first needs it at 4 years old. The data records how many months are left on it.

POA

"Price on application". The seller has not given a price, so the row cannot be used.

Trim

The equipment level of the car, such as Trendline (basic) or Highline (well equipped). Better trims cost more when new.

### Data cleaning words

Raw scraped data has gaps and mistakes. The textbook (Han, Kamber and Pei, section 3.2) lists standard ways to deal with a missing value. The project uses five of them.

Missing value

A cell with nothing in it, because the seller left that field empty.

Noisy data / outlier

A value that is present but wrong or wildly unusual, such as a price of €1 or a mileage of 1,234 km typed only to fill a required box.

Ignore the tuple

Delete the whole row. Used when the price or mileage is missing and cannot be recovered. 41 rows (3.2%) were removed this way.

Fill in manually

A person reads the advert and types in the right value. Used 30 times, for example when the mileage box said 204 km but the description said 204,000 km.

Global constant

Replace every missing value in a column with the same word, such as "Unknown". Used for trim, import status and NCT.

Median by year

Fill a gap with the middle value among cars of the same year. Used for previous owners, because older cars tend to have had more owners.

Most probable value

Fill a gap with the value that is most likely given the rest of the row. Used for 7 missing seller types: if the advert offers a warranty, the seller is taken to be a dealer, otherwise private.

Median

The middle value when a list is sorted. Half the values are below it and half above. It is not thrown off by a few extreme values, which is why the project uses it instead of the average.

Cleaning log

A record of every row removed and every value changed, with the reason. It is saved as `data/cleaning_log.csv` so the cleaning can be checked.

### Classification words

Classification

Teaching a program to put things into named groups by showing it examples whose group is already known.

Model / classifier

The thing that has been taught. You give it a car's attributes and it gives back a class.

Training set

The cars the model learns from. Here it is 80% of the data, 997 cars.

Test set

Cars kept hidden during learning and used once at the end to see how well the model does on cars it has never seen. Here it is 20%, 249 cars.

Holdout method

The name for splitting the data once into a training set and a test set.

Stratified split

A split that keeps the same share of each class in both sets. If 24% of all cars are Faster, about 24% of the training cars and of the test cars are Faster too.

Seed

A number that fixes the "random" choices, so that running the program again gives the same split and the same results. The project uses 2026.

Baseline

The score a lazy method would get. Here the lazy method always answers "Normal", which is right 58.6% of the time. A model is only useful if it beats this.

Overfitting

When a model memorises its training cars, quirks included, and then does worse on new cars.

### Decision tree words

Decision tree

A model shaped like a flowchart of yes/no questions. You start at the top, answer each question about the car, and end at a box that gives the prediction.

TDIDT

Top-Down Induction of Decision Trees. The method from the course notes for building a tree: pick the best question for all the cars, split them, then repeat inside each group.

Node

One box in the tree. A node holds a group of training cars.

Root

The first node at the top, which holds all the training cars.

Leaf

A node at the bottom with no further question. It predicts the most common class among the training cars that ended up in it.

Split

A question that divides a node's cars into two groups.

Binary split

A split with exactly two answers. Every split in this project is binary.

Split-point

For a numeric attribute, the number in the question, as in "mileage ≤ 164,950 km?". It is taken halfway between two neighbouring values in the data.

Splitting subset

For a categorical attribute, the set of values in the question, as in "trim level in {High, Mid, Unknown}?".

Impurity

How mixed a group of cars is. A group that is all one class has impurity 0. The more evenly the classes are mixed, the higher it is.

Gini index

One way to measure impurity: 1 minus the sum of each class's share squared. The best split is the one whose two groups have the lowest combined Gini index. This is the project's main measure.

Entropy / Info(D)

A second way to measure impurity, from information theory: minus the sum of each share times its base-2 logarithm.

Information gain

How much the entropy drops because of a split. The project builds a second tree with this measure to compare it with Gini.

Depth

How many questions deep the tree goes. A deeper tree fits the training cars more closely.

Maximum depth

A limit on depth, used to stop overfitting. The project's limit, 5, was chosen by cross-validation.

Minimum leaf size

No leaf may hold fewer than 10 training cars, so no rule is based on a handful of cars.

Adequacy condition

From the notes: no two cars with identical attributes may have different classes. Real car data breaks this rule, so a leaf can stay mixed and predicts its majority class.

Rule

The path from the root to a leaf written as a sentence: IF these answers THEN this class.

Attribute importance

How much of the tree's total drop in impurity comes from each attribute. It shows which attributes the tree leans on.

### Naive Bayes words

Naive Bayes

A second kind of classifier, based on probability. For each class it multiplies how common the class is by how common each of the car's attribute values is within that class. The class with the largest result wins.

Prior probability

How common a class is before looking at the car. About 59% of training cars are Normal, so the prior for Normal is 0.59.

Likelihood

How common an attribute value is inside one class. For example, the share of Faster cars that have over 200,000 km.

Posterior probability

The model's final probability for a class after looking at the car.

"Naive"

The method assumes each attribute is independent of the others within a class. That is not really true (age and mileage go together), but the method often works anyway.

Bands (discretisation)

Naive Bayes needs categories, so numbers are grouped into bands first. Mileage becomes "under 50k", "50-100k" and so on.

Laplace smoothing

Adding 1 to every count. Without it, a value never seen with a class would give a probability of zero and wipe out that class entirely.

### Evaluation words

Accuracy

The share of test cars the model classifies correctly.

Confusion matrix

A table of actual class against predicted class. The diagonal holds the correct answers and every other cell is a kind of mistake.

Precision

Of the cars the model called a class, the share that really were that class.

Recall

Of the cars that really were a class, the share the model found.

F-measure

One number that combines precision and recall: 2 × precision × recall ÷ (precision + recall). Useful when the classes are uneven in size, as they are here.

k-fold cross-validation

Split the training cars into k equal parts. Train on k − 1 parts and check on the remaining part, k times, so each part is the check once. The project uses 5 folds to choose the tree depth without touching the test set.

Random subsampling

Repeating the holdout split many times with different random splits and averaging. The project does this 30 times to check the result is not a fluke of one split.

Standard deviation

A measure of how much a set of numbers varies around its average.

ROC curve

A chart for one class against the rest. It shows the trade-off between catching more of that class (true positive rate) and raising more false alarms (false positive rate).

True positive rate

The same as recall: of the cars that are the class, the share caught.

False positive rate

Of the cars that are not the class, the share wrongly flagged as it.

Area under the curve (AUC)

One number summarising an ROC curve. 0.5 is no better than guessing and 1.0 is perfect.

### Words specific to this project

Typical price

The median asking price of Golfs from the same year, smoothed over neighbouring years, then adjusted for trim level.

Group

Regular or Performance. Performance Golfs (GTI, GTD, GTE, R) cost far more new, so they are only compared with each other.

Trim level

The many trim names grouped into Base, Mid, High, Performance or Unknown.

Trim adjustment

A multiplier for each trim level, so a Highline is compared with what Highlines usually sell for. Base ×0.93, Mid ×0.99, High ×1.06.

Ratio

The car's price divided by its typical price. 0.80 means 20% cheaper than typical.

Band (±15%)

The width of "Normal". A ratio of 0.85 or less is Faster, above 1.15 is Slower, and anything between is Normal.

Km per year

Mileage divided by age plus half a year. It shows how hard the car has been driven.

Value kept

The median price of an age band as a share of the median price of Golfs 0-1 years old.

Forecast

Moving a car forward N years and classifying the future car.

Snapshot

The data was collected over a few days (30 September to 5 October 2026). It shows the market at one moment, not the same cars over time.

## Every file and function

The code is in six Python files. Each table lists the functions in the order they run.

### `donedeal_scraper.py`: collect the adverts

| Function | What it does |
| --- | --- |
| `load_robots()` | Reads DoneDeal's robots.txt, the file where a website says which pages programs may read. |
| `fetch(url)` | Downloads one page politely. It waits a random 3 to 5 seconds first, checks robots.txt allows the page, and identifies itself as a browser. If the site says "too many requests" it pauses and retries; if the site refuses outright it stops the run. |
| `retry_wait()` | Works out how long to pause after a "too many requests" answer. |
| `extract(page_html)` | Finds the advert's data inside the page. DoneDeal pages carry it as a block of structured text (JSON) in a tag called `__NEXT_DATA__`. |
| `ad_id_from_url(url)` | Takes the advert's ID number from the end of its web address. |
| `text()`, `number()`, `integer()` | Tidy raw values. "63,000 km" becomes 63000, "1.2 L" becomes 1.2, and empty placeholders become "missing". |
| `fuel_type()` | Maps DoneDeal's fuel labels onto Petrol, Diesel, Hybrid, Plug-in Hybrid, Electric or Other. |
| `nct_months_remaining()` | Turns an NCT expiry such as "Aug 2027" into months left. Expired tests give a negative number. |
| `is_import()` | True if the car is or was registered outside Ireland, and "missing" if the advert does not say. |
| `build_row()` | Turns one advert into one row with the 20 columns of `carData.csv`. |
| `out_of_scope()` | Rejects adverts not priced in euro and fully electric cars. |
| `read_keys()`, `read_ids()` | Read which adverts are already saved or already rejected, so no advert is downloaded twice. |
| `append_row()`, `append_id()` | Add one row to the CSV, or one rejected ID to `skipped_ads.txt`. |
| `read_urls()` | Reads the list of advert addresses to scrape, from a file or a single address. |
| `search_urls()` | Reads DoneDeal's search result pages for a make and model and collects the addresses of adverts not yet scraped. |
| `main()` | Runs the whole loop: for each address, fetch, extract, check scope, build the row and save it. One failed advert never stops the run. |

### `clean_data.py`: clean the table

| Name | What it does |
| --- | --- |
| `MANUAL_FIXES` | A hand-written list of corrections, each with the advert ID, the right value and where in the advert it was found. |
| `DROP` | A hand-written list of rows to remove, each with the reason. |
| `INVALID_TRIMS`, `TRIM_SPELLING` | Trim values that are not real trims, and one agreed spelling for trims typed in different cases. |
| `record()` | Writes one line into the cleaning log. |
| `set_value()` | Changes one cell and logs the old value, new value, method and reason. |
| `fill_column()` | Fills every chosen cell in a column, using `set_value()` for each one. |
| `main()` | Applies the five cleaning methods in order, saves `carData_clean.csv` and the log, and prints a summary. |

### `tdidt.py`: the two classifiers, written from scratch

| Name | What it does |
| --- | --- |
| `impurity(counts, criterion)` | Given how many cars of each class are in a group, returns its Gini index or its entropy. |
| `Node` | One box of the tree. It stores the class counts of the cars that reach it, its depth, its question (if any) and its two children. |
| `Node.goes_left(value)` | Answers the node's question for one car: yes sends the car left, no sends it right. |
| `DecisionTree` | The tree. Created with a measure (`'gini'` or `'entropy'`), a maximum depth and a minimum leaf size. |
| `fit(X, y)` | Trains the tree. `X` is the table of attributes and `y` is the list of class labels. |
| `_grow()` | Builds one node, then calls itself for the two children. It stops when the node is pure, at the depth limit, or too small to split. |
| `_best_split()` | Tries every attribute at a node and keeps the split with the lowest impurity. |
| `_numeric_split()` | For a numeric attribute, tests every halfway point between neighbouring values. |
| `_categorical_split()` | For a categorical attribute, tests every way of dividing its values into two sets. |
| `predict_proba(X)` | For each car, walks down the tree and returns the share of each class in the leaf it reaches. |
| `predict(X)` | Returns the class with the largest share. |
| `nodes()`, `n_leaves()` | List the nodes and count the leaves. |
| `importance()` | Returns each attribute's share of the tree's total drop in impurity. |
| `rules()` | Writes every leaf as an IF … THEN rule. |
| `number()` | Formats a split-point for reading, such as 152,450 or 1.45. |
| `NaiveBayes` | The second classifier. `fit()` counts how often each attribute value appears in each class. `predict_proba()` multiplies the prior by the likelihoods. `predict()` picks the largest. |

Both classifiers were checked against the worked example in the course notes and textbook (the 14-row "buys computer" table). The code gives the same answers: a Gini index of 0.459 for the whole table, 0.443 for the best split on income, 0.357 for the best split on age, and an 80% probability of "yes" from Naive Bayes for the example customer.

### `classify_depreciation.py`: the experiment

| Name | What it does |
| --- | --- |
| Settings at the top | `BAND = 0.15` (width of Normal), `TEST_SHARE = 0.20`, `SEED = 2026`, `DEPTHS` 1 to 12, `MIN_LEAF = 10`, `FOLDS = 5`, `REPEATS = 30`. |
| `year_price(d)` | The median price for each year in each group, smoothed over 3 neighbouring years. |
| `trim_premium()` | The trim adjustment: the median of price ÷ year price for each trim level. |
| `expected_price()` | The typical price of each car: year price × trim adjustment. |
| `nct_band()` | Groups NCT months into Expired, 1-6 months, Over 6 months, Not due yet, or Unknown. |
| `attributes(d)` | Builds the 13 attributes the classifiers may use. |
| `bands(X)` | Groups the numeric attributes into bands for Naive Bayes. |
| `prepare()` | Loads the clean data, finds Performance Golfs, works out typical price, ratio and class label for every car. |
| `stratified_split()` | Makes the 80/20 split, keeping class shares equal. |
| `stratified_folds()` | Divides the training cars into 5 folds for cross-validation. |
| `accuracy()` | Share of predictions that match the actual class. |
| `depth_search()` | Runs the 5-fold cross-validation for every depth from 1 to 12 and returns the best depth. |
| `confusion()` | Builds the confusion matrix. |
| `class_scores()` | Precision, recall and F-measure for each class. |
| `roc_curve()` | The points of an ROC curve for one class against the rest, and the area under it. |
| `fit_models()` | Trains the four things compared: Gini tree, information gain tree, Naive Bayes and the baseline. |
| `predict()` | Gets predictions from whichever of the four is passed in. |
| `value_kept(d)` | Median price of each age band as a share of the 0-1 year price. |
| `car_attributes()` | Keeps only attributes of the car itself, dropping those that describe the advert. |
| `age_car(car, years)` | Returns the same car N years on: older, with kilometres added at its own rate. |
| `forecast()` | Ages the car, classifies the future car, and looks up the typical price for that age. |
| `price_range()` | Turns the predicted class into a price range in euro. |
| `say()`, `heading()` | Print a line and keep it for `results.txt`. |
| `main()` | Runs every step in order and saves all results and figures. |

### `figures.py` and `predict_car.py`

| Function | What it does |
| --- | --- |
| `label_scatter()` | Figure 1: every car against the typical price line, coloured by class. |
| `depth_curve()` | Figure 2: accuracy at each tree depth, on seen and unseen cars. |
| `confusion_matrix()` | Figure 3: the confusion matrix as a shaded grid. |
| `importance()` | Figure 4: attribute importance as bars. |
| `tree_diagram()` | Figure 5: the top three levels of the tree. |
| `model_comparison()` | Figure 6: accuracy of each model over the 30 splits. |
| `class_shares()` | Figure 7: the class mix for each value of an attribute. |
| `value_kept()` | Figure 8: value kept by age. |
| `roc_curves()` | Figure 9: ROC curves for the tree and Naive Bayes. |
| `predict_car.py` | A small command-line tool. You type a car's details and a number of years, and it prints the forecast. |

Part 2

## How the system works, step by step

### Step 1. Collect the adverts

The scraper is given a make and model (Volkswagen Golf). It reads DoneDeal's search pages to find adverts, then opens each advert and copies 20 facts into one row: ID, web address, make, model, trim, year, mileage, fuel, engine size, transmission, body type, previous owners, import status, seller type, county, warranty, date scraped, price, whether a price was given, and NCT months left.

It follows three rules of good behaviour. It obeys the site's robots.txt. It waits 3 to 5 seconds between pages so it does not strain the site. It stores no personal data: seller names, phone numbers and addresses are never read. Only adverts priced in euro are kept, and fully electric cars are left out.

The result is 1,291 rows, collected between 30 September and 5 October 2026.

### Step 2. Clean the data

Sellers type adverts by hand, so the raw table has gaps and errors. The cleaning runs in a fixed order.

1. **Fill in manually (30 values).** Someone read the advert and recovered the real value. For example, a mileage box saying 259 km on a car whose description said "161k miles" became 259,000 km. These fixes run first, so a row that gets its price or mileage back is not thrown away in the next step.
2. **Ignore the tuple (41 rows, 3.2%).** Rows with no usable price, mileage, engine size or transmission were removed, along with one advert that turned out to be a Passat. Price is what the label is built from, so a row without it is useless.
3. **Global constant (723 values).** Missing trim and import status became "Unknown". Missing NCT became "N/A" for cars under 4 years old, where no test is due yet, and "Unknown" for older cars. Keeping those two apart matters: one means nothing is missing, the other means something is.
4. **Median by year (108 values).** Missing previous owners were filled with the median for cars of the same year.
5. **Most probable value (7 values).** Missing seller type became Dealer if the advert offered a warranty, otherwise Private.

Trim names typed in different cases, such as HIGHLINE and Highline, were also given one spelling (234 values). Every change is written to the cleaning log. The output is 1,250 clean rows. Four more cars dated 2027 are dropped later because they are pre-orders, not used cars, leaving 1,246.

### Step 3. Give every car its class label

The label is not in the adverts. The project has to work it out from the price.

1. **Sort cars into two groups.** Performance Golfs (GTI, GTD, GTE, R and R32; 159 cars) are kept apart from Regular Golfs (1,087 cars). If they were mixed, 75% of performance cars would be called Slower only because they cost more when new.
2. **Find the typical price for each year.** Within each group, take the median price of each year, then smooth it across the neighbouring years so a year with few cars does not jump around.
3. **Adjust for trim level.** A Highline cost more new than a Trendline. So each trim level gets a multiplier: Base ×0.93, Mid ×0.99, High ×1.06, Performance ×1.02, Unknown ×1.01.
4. **Compare.** Ratio = the car's price ÷ its typical price.
5. **Label.** Ratio of 0.85 or less is Faster. Above 1.15 is Slower. In between is Normal.

Why does this measure depreciation? Depreciation is the value a car has lost since it was new. Golfs of the same year, group and trim started at roughly the same price. So a car that is now 20% cheaper than its peers has lost more value than they have.

| Class | Cars | Share |
| --- | --- | --- |
| Faster | 293 | 24% |
| Normal | 732 | 59% |
| Slower | 221 | 18% |

![Scatter chart of every car's price by year, with the typical price line and a shaded band of plus or minus 15 percent. Cars below the band are red, inside grey, above blue.](fig/1_class_label.png)

Each dot is one car. The black line is the typical price for its year. Red dots sit more than 15% below it, blue dots more than 15% above.

### Step 4. Choose what the model may look at

The model gets 13 attributes: mileage, km per year, age, previous owners, engine size, trim level, fuel, transmission, body type, seller type, warranty, NCT band and county (Dublin or outside Dublin).

Two things are deliberately kept out.

- **Price.** The label is built from price. Showing it to the model would be giving away the answer.
- **Import status.** During cleaning, missing import status became "Unknown". All 266 of those cars turned out to be dealer adverts, and 34% of them were Slower against 13% of the rest. So "Unknown" really meant "a dealer left this box empty". A model using it would be learning how adverts are filled in, not anything about the car. The textbook warns about exactly this risk with global constants (p.88).

### Step 5. Split into training and test cars

The 1,246 cars are split at random: 997 for training (80%) and 249 for testing (20%). The split is stratified, so each set has the same mix of Faster, Normal and Slower. The test cars are locked away until the very end. Any choice made along the way, such as the tree depth, uses the training cars only.

### Step 6. Grow the decision tree

The tree is built from the top down, following TDIDT.

1. Start with all 997 training cars in the root.
2. For every attribute, try every possible yes/no question. For mileage, that means every halfway point between two neighbouring mileages. For fuel, every way of dividing {Petrol, Diesel, Hybrid} into two sets.
3. For each question, measure how mixed the two resulting groups are, using the Gini index.
4. Keep the question that gives the least mixed groups.
5. Repeat inside each of the two groups.
6. Stop when a group is all one class, when the depth limit is reached, when a group is too small to give two leaves of 10 cars, or when no question helps.

**Worked example: the first split.** The root holds 234 Faster, 586 Normal and 177 Slower cars.

```
Gini(root) = 1 − (234/997)² − (586/997)² − (177/997)²
           = 1 − 0.055 − 0.345 − 0.032 = 0.568

Best question found: "Mileage ≤ 164,950 km?"
  Yes: 736 cars (73 Faster, 510 Normal, 153 Slower)   Gini = 0.467
  No : 261 cars (161 Faster, 76 Normal, 24 Slower)    Gini = 0.526

Gini after the split = (736 × 0.467 + 261 × 0.526) / 997 = 0.482
Drop in impurity     = 0.568 − 0.482 = 0.086
```

No other question on any attribute gave a bigger drop, so this became the root. It makes sense: of the cars over 164,950 km, 62% are Faster. The tree then asks about age on the "yes" side and km per year on the "no" side, and so on down.

![Diagram of the top three levels of the decision tree, starting with the mileage question.](fig/5_decision_tree.png)

The top three levels of the tree. Read from the top and answer each question. The small bar in each box shows the class mix of the training cars there.

### Step 7. Choose how deep the tree should go

A deeper tree always fits its training cars better, but past a point it is only memorising them. To find that point without touching the test cars, the project uses 5-fold cross-validation on the training cars. It splits them into 5 parts, trains on 4, checks on the fifth, and rotates so each part is the check once. It does this for every depth from 1 to 12.

| Depth | Accuracy on cars it trained on | Accuracy on held-out cars |
| --- | --- | --- |
| 1 | 67.4% | 66.6% |
| 3 | 72.3% | 69.4% |
| **5** | 74.8% | **69.8%** |
| 7 | 76.5% | 67.5% |
| 12 | 77.2% | 67.4% |

Accuracy on held-out cars peaks at depth 5 and then falls while training accuracy keeps rising. That widening gap is overfitting. So the final tree is limited to depth 5, which gives 24 leaves, or 24 rules.

![Line chart of accuracy by tree depth. The training line keeps rising; the held-out line peaks at depth 5.](fig/2_depth_vs_accuracy.png)

Training accuracy keeps climbing with depth. Accuracy on held-out cars does not.

### Step 8. Train Naive Bayes as a comparison

The numeric attributes are first grouped into bands. Then Naive Bayes counts. For a new car it works out, for each class:

```
score(class) = P(class) × P(mileage band | class) × P(age band | class) × … × P(county | class)
```

The class with the highest score is the prediction. Every count has 1 added to it (Laplace smoothing), so an unseen combination never produces a zero that would cancel the whole product.

A second decision tree is also trained using information gain in place of the Gini index. Both trees use binary splits, which the notes allow, so the two measures are compared on equal terms.

### Step 9. Test on the unseen cars

Each model now classifies the 249 test cars, and the answers are compared with the real labels.

| Model | Test accuracy | Average over 30 splits |
| --- | --- | --- |
| Decision tree (Gini) | 72.7% | 69.5% |
| Decision tree (information gain) | 71.5% | 69.9% |
| Naive Bayes | 66.7% | 64.9% |
| Baseline (always "Normal") | 58.6% | 58.6% |

One split could be lucky, so the whole thing is repeated on 30 different random splits. The Gini tree beat Naive Bayes on 26 of the 30.

**The confusion matrix** for the Gini tree shows where the mistakes are. Rows are the real class and columns are what the tree said.

| Actual | Said Faster | Said Normal | Said Slower | Precision | Recall | F-measure |
| --- | --- | --- | --- | --- | --- | --- |
| Faster (59) | **38** | 21 | 0 | 69% | 64% | 67% |
| Normal (146) | 16 | **128** | 2 | 72% | 88% | 79% |
| Slower (44) | 1 | 28 | **15** | 88% | 34% | 49% |

How to read one row: 44 test cars were really Slower. The tree found 15 of them (recall 34%), called 28 Normal and 1 Faster. When the tree does say Slower it is usually right (precision 88%), but it says it rarely.

![Three ROC charts, one for each class against the rest, each with a line for the decision tree and a line for Naive Bayes.](fig/9_roc_curves.png)

ROC curves. The closer a line is to the top-left corner, the better. Area under the curve: tree 0.84, 0.77, 0.84 and Naive Bayes 0.87, 0.75, 0.86 for Faster, Normal and Slower.

### Step 10. Forecast a car's future

This is the prediction the project set out to make. You give a car's details and a number of years.

1. **Age the car.** Age goes up by N years. Mileage goes up by N × the car's own km per year so far. Engine, trim, fuel, transmission, body and owners stay the same.
2. **Classify the future car.** A tree trained only on attributes of the car itself (not seller, warranty, NCT or county, which nobody can know years ahead) says which class cars like that fall into today.
3. **Turn the class into euro.** Look up the typical price of Golfs of that future age, group and trim today. Faster means under 85% of it, Slower means over 115%.

**Example.** A 2017 1.6 TDI Trendline with 210,000 km has done about 22,100 km a year.

| When | Age | Mileage | Prediction | Typical Golf that age |
| --- | --- | --- | --- | --- |
| Today | 9 | 210,000 km | Normal (63%) | €13,400 |
| In 3 years | 12 | 276,000 km | Faster (98%) | €9,900, so this car under €8,400 |
| In 5 years | 14 | 321,000 km | Faster (85%) | €6,000, so this car under €5,100 |

The tree's reasoning: 12-year-old Golfs with over 224,000 km and heavy yearly use are almost all priced well below typical today, so this one will be too.

Leaving out the four advert attributes cost nothing. The car-only tree scores the same 72.7% on the test cars and 69.5% over 30 splits.

## What the results say

- **The tree beats guessing by about 14 points** on the test cars (72.7% against 58.6%), and by about 11 points averaged over 30 splits.
- **Mileage is what matters most.** It accounts for 53% of the tree's work, followed by age (23%), km per year (14%) and engine size (4%). Seller type, fuel, transmission, NCT and previous owners were not used at all.
- **The tree is good at spotting Faster cars and poor at spotting Slower ones.** High mileage is a clear sign of a cheap car. Nothing in the adverts marks out a car that holds its value as clearly.
- **The decision tree is the better model on accuracy and F-measure** (65% against 60% averaged over the classes). Naive Bayes is slightly ahead on the ROC curves for Faster and Slower, meaning its probabilities rank cars a little better even though its final answers are right less often.
- **Accuracy depends on age.** It is 86 to 89% for cars up to 7 years old and 62 to 70% for older ones, whose prices are far more spread out.
- **A Regular Golf keeps about 78% of its value at 2-3 years, 50% at 6-7, 32% at 10-11 and 13% at 14-15.**

![Bar chart of attribute importance. Mileage is the longest bar, then age, km per year and engine size.](fig/4_attribute_importance.png)

How much of the tree's total drop in impurity comes from each attribute.

![Line chart of the share of value a Regular Golf keeps at each age band, falling from 100 percent to 7 percent.](fig/8_value_kept_by_age.png)

Median price of each age band as a share of the median price at 0-1 years.

## What the project cannot tell you

- **The data is one snapshot.** The cars were not followed over time. The forecast assumes that in N years the market will treat cars the way it treats them today. That cannot be checked with this data.
- **Prices are asking prices.** They are what sellers hoped for, not what buyers paid.
- **Forecasts past about 15 years old are unreliable.** There are few cars that old in the data, and one example flips from Faster to Slower at 16.
- **Only Golfs.** Nothing here says the same rules hold for other cars.
- **The model misses most Slower cars.** If it says Slower you can mostly believe it, but a Normal answer does not rule Slower out.