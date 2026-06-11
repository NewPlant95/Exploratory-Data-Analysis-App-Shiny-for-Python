# Exploratory-Data-Analysis-App-Shiny-for-Python
A web app within the shiny for python framework that lets students, early career data professionals as well as non-technical users explore and play with their datasets.

Within this Shiny app you can inspect, clean, reshape, plot, and analyze CSV data in the browser. When no file is uploaded, the app loads a bundled demo dataset based upon the work of Ignaz Semmelweis and the importance of handwashing within hosptials. The demo date features columns: `date`, `births`, `deaths`, `pct_deaths`, `rolling_mean`, and `washing_hands`.

## What It Does

- Upload one or more CSV files and switch the active table from the `Data Source` panel.
- Preview the active data and control which columns are shown.
- Apply common cleaning, reshaping, and transformation operations.
- Build charts with Plotly, Seaborn, or Matplotlib.
- Tune titles, axis labels, font sizes, ticks, gridlines, and line styling.
- Use twin y axes for scatter and line charts, with separate styles and colors for each series.
- Run built-in statistical views including descriptive summaries, correlation, regression, logistic regression, t-test, ANOVA, heteroscedasticity checks, PCA, and clustering-related previews.
- Save edited data, save the current preview as a new active dataset, and export plots or statistical output.

## Run It

From the project root:

```bash
./.venv/bin/shiny run --reload app.py
```

If you are not using the bundled virtual environment, install the dependencies first:

```bash
pip install shiny numpy pandas matplotlib seaborn plotly
```

Then run:

```bash
shiny run --reload app.py
```

The app currently accepts CSV uploads only.

## Main Workflow

1. Upload one or more CSV files using **Data Source**.
2. Pick the active file from **Active CSV file**.
3. Use **Data Preview** to inspect the current table.
4. Use **Column Tools** to clean, reshape, join, or calculate new columns.
5. Use the preview controls to show specific columns or all columns.
6. Move to **Visualisation Controls** to build and format charts.
7. Use **Statistical Controls** to run the analysis views.
8. Save the edited CSV, save the current preview as a new data source, or export a plot/result.

## Data Source

The `Data Source` panel supports:

- Uploading multiple CSV files in one session
- Switching the active dataset without re-uploading
- Falling back to the bundled demo dataset when no CSV is loaded

## Data Preview

The preview panel supports:

- Showing selected columns only
- Showing all columns
- Removing a column from the preview and then restoring the full set later

## Column Tools

The `Column Tools` panel currently supports:

- Rename columns
- Convert column types
- Find and replace text
- Drop rows based on missing values or a selected value
- Find duplicate rows
- Drop duplicates from one column or all columns
- Pivot wide
- Melt to long format
- Join uploaded tables using a primary/foreign key mapping
- Z-score normalisation for one numeric column
- Round numeric values to a chosen number of decimal places
- Create a calculated column from a formula using `A` and `B`
- Save the current preview as a new active dataset
- Download the current preview as a CSV
- Undo the last column edit

Note: The app keeps column edits in history so you can undo the last change.

### Calculated Columns

The formula tool lets you build a new column from two selected columns.

Example:

```
(A*B)/B
```

Where `A` and `B` are used as placeholders for the two chosen columns. The result is added as a new column in the preview.

### Pivot Wide

Use this when you want to summarize and spread one field into multiple columns, similar to an Excel PivotTable.

Typical setup:

- **Row fields**: the grouping columns that define the subset
- **Column field**: values that become new columns
- **Values field**: the metric to aggregate
- **Aggregation**: `sum`, `mean`, `count`, and similar options

Example:

- Row fields: `date`, `washing_hands`
- Column field: `metric`
- Values field: `value`
- Aggregation: `sum` or `mean`

If the same row/column combination appears more than once, the app aggregates those rows using the selected function.

### Melt

Use this when several columns represent the same kind of measurement and you want to stack them into a long table.

Typical setup:

- **Identifier fields**: columns to keep fixed
- **Fields to unpivot**: columns to turn into rows
- **Variable column name**: output label column
- **Value column name**: output value column

Example:

- Identifier fields: `date`, `washing_hands`
- Fields to unpivot: `births`, `deaths`, `pct_deaths`, `rolling_mean`
- Variable column name: `metric`
- Value column name: `value`

Use `Melt` when you want to create subsets like:

- one row per date and metric
- one row per `washing_hands` state and metric
- a long table that is easier to plot or filter by metric name

### Join Tables

Use this when you have multiple uploaded CSV files and want to combine them using a key from the current table and a key from another uploaded table.

Typical setup:

- **Foreign table**: the uploaded CSV you want to join in
- **Active table**: the currently selected uploaded CSV that the rest of the app uses
- **Join type**: `left`, `inner`, `right`, or `outer`
- **Primary key column**: the key in the current table
- **Foreign key column**: the matching key in the uploaded table

This is useful for:

- adding lookup values from another file
- combining fact and dimension tables
- enriching the active dataset without manual copying

## Plotting

The visualization panel lets you choose:

- Plot type: scatter, line, bar, histogram, pie, box, heatmap
- Rendering engine: Plotly, Seaborn, or Matplotlib
- X/Y columns and grouping
- Optional twin y axis for scatter and line charts
- Log scaling for axes
- Plot title and axis titles
- Title and axis font sizes
- Tick rotation and tick density
- Grid axis, opacity, and style
- Line styling for line charts, including dash pattern, width, and markers
- Per-axis styling for twin-axis line charts

The plot formatting menu is hidden by default and can be opened with the toggle in the visualization panel.

The app supports plot exports where the export file types depend on the plot engine:

- Plotly plots export as interactive HTML by default.
- Plotly image export to PNG, SVG, or PDF works if `kaleido` is installed.
- Matplotlib plots export as PNG, SVG, or PDF.

## Statistics

The statistics panel supports:

- Descriptive summaries
- Correlation
- Simple linear regression
- Multiple regression
- Logistic regression
- t-test
- ANOVA
- Heteroscedasticity preview
- PCA with a data-driven suggested component count based on the selected columns
- Optional K-means cluster colouring for supported plots with a suggested K value

The PCA suggestion is based on the elbow of the cumulative explained-variance curve for the selected columns.

The statistics plot also supports axis log switches where appropriate.

## Output Files

- **Save new dataframe** turns the current preview into a new active dataset and adds it to the `Data Source` list.
- **Download new dataframe** downloads the current preview as a CSV.
- **Save current plot** appears on both the main plot panel and the statistics plot panel.
- **Save results CSV** exports the current statistics output.

## Notes

- If no CSV is uploaded, the app loads the bundled monthly dataset automatically.
- The app keeps column edits in history so you can undo the last change.
