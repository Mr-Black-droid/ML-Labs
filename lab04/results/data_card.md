# Data Card: Adult Census Income Dataset

## Dataset Description
- **Source**: UCI Machine Learning Repository via OpenML (`fetch_openml(name='adult', version=2, as_frame=True)`)[cite: 5, 9].
- **Size**: 48,842 rows and 15 columns[cite: 9].
- **Target Variable**: `class` (`<=50K` or `>50K` annual income).
- **Features**: Includes demographic and employment attributes such as `age`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, and `native-country`.

## Known Biases
- Data is extracted from the 1994 US Census Bureau database, reflecting historical socioeconomic and demographic distributions, wage gaps, and occupational trends of that era.
- May contain sampling biases favoring specific geographic or demographic representation present in 1990s tax and census filings.

## Privacy Concerns
- Contains demographic markers and employment data. While extracted from public census datasets, individual record re-identification risks exist when cross-referenced with external public registries.