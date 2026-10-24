# Corporate dashboard

The redesigned `app.py` is presentation-only. Training code in `src/` and data files are unchanged. Previous attractive design is backed up as `app_previous_design.py`; original dashboard as `app_classic.py`.

From the project root:

```powershell
python -m pip install -r requirements.txt
python -m src.train --data data/raw/demo_reviews.csv
python -m streamlit run app.py
```

To train with real public reviews:

```powershell
python prepare_real_data.py
python -m src.train --data data/raw/amazon_reviews.csv
python -m streamlit run app.py
```

Charts only show category or time dimensions if those fields exist.
