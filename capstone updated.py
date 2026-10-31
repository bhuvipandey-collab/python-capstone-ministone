
import os
import warnings
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_DIR = r'C:\Users\prash\Desktop'
DATA_PATH = os.path.join(DATA_DIR, 'city_day.csv')

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

pollutants = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]

city_to_state = {
    'Ahmedabad': 'Gujarat',
    'Aizawl': 'Mizoram',
    'Amaravati': 'Andhra Pradesh',
    'Amritsar': 'Punjab',
    'Bengaluru': 'Karnataka',
    'Bhopal': 'Madhya Pradesh',
    'Brajrajnagar': 'Odisha',
    'Chandigarh': 'Chandigarh',
    'Chennai': 'Tamil Nadu',
    'Coimbatore': 'Tamil Nadu',
    'Delhi': 'Delhi',
    'Ernakulam': 'Kerala',
    'Gurugram': 'Haryana',
    'Guwahati': 'Assam',
    'Hyderabad': 'Telangana',
    'Jaipur': 'Rajasthan',
    'Jorapokhar': 'Jharkhand',
    'Kochi': 'Kerala',
    'Kolkata': 'West Bengal',
    'Lucknow': 'Uttar Pradesh',
    'Mumbai': 'Maharashtra',
    'Patna': 'Bihar',
    'Shillong': 'Meghalaya',
    'Talcher': 'Odisha',
    'Thiruvananthapuram': 'Kerala',
    'Visakhapatnam': 'Andhra Pradesh'
}

df['State'] = df['City'].map(city_to_state)

# Remove rows where all pollutant values are missing
df.dropna(subset=pollutants, how='all', inplace=True)

# Keep original readings for warnings and charts
before_impute = df.copy()

# --------------------------------------------------
# 2. HANDLE MISSING VALUES
# --------------------------------------------------

for col in pollutants:
    df[col] = df.groupby('City')[col].transform(
        lambda x: x.fillna(x.median())
    )
    df[col] = df[col].fillna(df[col].median())

df['AQI'] = (
    df.groupby('City')['AQI']
    .transform(lambda x: x.ffill().bfill())
    .fillna(df['AQI'].median())
)

def fill_bucket(series):
    mode = series.mode()
    if not mode.empty:
        return series.fillna(mode.iloc[0])
    return series

df['AQI_Bucket'] = (
    df.groupby('City')['AQI_Bucket']
    .transform(fill_bucket)
)

if df['AQI_Bucket'].notna().any():
    df['AQI_Bucket'] = df['AQI_Bucket'].fillna(
        df['AQI_Bucket'].mode().iloc[0]
    )

# --------------------------------------------------
# 3. MAIN STATE-WISE TABLE
# --------------------------------------------------

state_table = (
    df.groupby('State')[pollutants + ['AQI']]
    .mean()
    .sort_values('AQI', ascending=False)
)

print('\n--- STATE-WISE AVERAGE AQI & POLLUTANT LEVELS ---')
print(state_table.round(2).to_string())

# --------------------------------------------------
# 4. CHART 1: STATE-WISE AVERAGE AQI
# --------------------------------------------------

plt.figure(figsize=(12, 7))

sns.barplot(
    x=state_table.index,
    y=state_table['AQI'],
    hue=state_table.index,
    palette='Reds_r',
    legend=False
)

plt.title('State-wise Average AQI', fontsize=16)
plt.xlabel('State')
plt.ylabel('Average AQI')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig(
    os.path.join(DATA_DIR, 'chart_state_average_aqi.png'),
    dpi=150
)

# --------------------------------------------------
# 5. CHART 2: AQI CATEGORY DISTRIBUTION BY STATE
# --------------------------------------------------

aqi_labels = [
    'Good', 'Satisfactory', 'Moderate',
    'Poor', 'Very Poor', 'Severe'
]

plt.figure(figsize=(15, 8))

sns.countplot(
    data=df,
    x='State',
    hue='AQI_Bucket',
    hue_order=aqi_labels,
    palette='viridis'
)

plt.title('Distribution of AQI Categories by State', fontsize=16)
plt.xlabel('State')
plt.ylabel('Number of Days')
plt.xticks(rotation=45, ha='right')
plt.legend(title='AQI Category')
plt.tight_layout()

plt.savefig(
    os.path.join(DATA_DIR, 'chart_aqi_buckets_by_state.png'),
    dpi=150
)

# --------------------------------------------------
# 6. CHART 3: PM2.5 VS PM10 FOR TOP 5 STATES
# --------------------------------------------------

top5_states = state_table.head(5).index

top5_data = df[df['State'].isin(top5_states)]

plt.figure(figsize=(12, 6))

sns.scatterplot(
    data=top5_data,
    x='PM2.5',
    y='PM10',
    hue='State',
    alpha=0.6
)

plt.title('PM2.5 vs PM10 for Top 5 Most Polluted States', fontsize=16)
plt.xlabel('PM2.5')
plt.ylabel('PM10')
plt.tight_layout()

plt.savefig(
    os.path.join(DATA_DIR, 'chart_pm25_vs_pm10_top5.png'),
    dpi=150
)

# --------------------------------------------------
# 7. ASK USER FOR STATE AND MONTH
# --------------------------------------------------

available_states = sorted(df['State'].dropna().unique())
state_lookup = {s.lower(): s for s in available_states}

print('\nAvailable states:')
print(', '.join(available_states))

entered_state = input('\nEnter your state: ').strip()
selected_state = state_lookup.get(entered_state.lower())

month_input = input('Enter month number (1-12): ').strip()

month = None

if month_input.isdigit() and 1 <= int(month_input) <= 12:
    month = int(month_input)

if selected_state is None:
    print('Invalid state. Please enter a state from the list.')

elif month is None:
    print('Invalid month. Please enter a number from 1 to 12.')

else:
    # --------------------------------------------------
    # 8. STATE AND MONTH DATA USING REAL READINGS
    # --------------------------------------------------

    state_data = before_impute[
        (before_impute['State'] == selected_state) &
        (before_impute['Date'].dt.month == month)
    ].copy()

    real_aqi = state_data.dropna(subset=['AQI'])

    print(
        f'\n--- WARNINGS FOR {selected_state.upper()} - '
        f'{date(2000, month, 1).strftime("%B").upper()} ---'
    )

    warnings_found = 0

    if real_aqi.empty:
        print('No actual AQI readings available for this state and month.')

    else:
        avg_aqi = real_aqi['AQI'].mean()
        poor_days = (real_aqi['AQI'] > 200).mean() * 100

        # AQI warning
        if avg_aqi <= 50:
            label = 'Good'
        elif avg_aqi <= 100:
            label = 'Satisfactory'
        elif avg_aqi <= 200:
            label = 'Moderate'
        elif avg_aqi <= 300:
            label = 'Poor'
        elif avg_aqi <= 400:
            label = 'Very Poor'
        else:
            label = 'Severe'

        print(f'Average AQI: {avg_aqi:.0f} ({label})')

        if avg_aqi > 200:
            print(
                '[WARNING] Poor air quality. '
                'Reduce prolonged outdoor activity.'
            )
            warnings_found += 1

        if poor_days >= 50:
            print(
                f'[WARNING] {poor_days:.0f}% of recorded days '
                'had AQI above 200.'
            )
            warnings_found += 1

        # Pollutant warnings based on recorded values only
        standards = {
            'PM2.5': 60,
            'PM10': 100,
            'NO2': 80,
            'SO2': 80,
            'CO': 2.0,
            'O3': 100,
            'NH3': 400
        }

        for pollutant, limit in standards.items():
            values = state_data[pollutant].dropna()

            if not values.empty and values.mean() > limit:
                print(
                    f'[WARNING] {pollutant} average is '
                    f'{values.mean():.2f}, exceeding the '
                    f'reference limit of {limit}.'
                )
                warnings_found += 1

        if warnings_found == 0:
            print('No warnings for the selected state and month.')

    # --------------------------------------------------
    # 9. CHART 4: SELECTED STATE MONTHLY AIR QUALITY
    # --------------------------------------------------

    state_history = before_impute[
        before_impute['State'] == selected_state
    ].dropna(subset=['AQI']).copy()

    if state_history.empty:
        print('No historical AQI data available for the chart.')

    else:
        # Group actual AQI readings by calendar month
        monthly_data = state_history.groupby(
            state_history['Date'].dt.month
        )['AQI'].agg(
            median_aqi='median',
            days='count',
            poor_days=lambda x: (x > 200).mean() * 100
        )

        monthly_data = monthly_data.reindex(range(1, 13))

        fig, ax1 = plt.subplots(figsize=(11, 5))

        months = range(1, 13)

        ax1.bar(
            months,
            monthly_data['poor_days'],
            alpha=0.65
        )

        ax1.set_ylabel('% of Recorded Days with AQI > 200')
        ax1.set_ylim(0, 100)
        ax1.set_xticks(list(months))
        ax1.set_xticklabels([
            date(2000, m, 1).strftime('%b')
            for m in months
        ])

        # Highlight the month selected by the user
        ax1.axvspan(
            month - 0.5,
            month + 0.5,
            color='grey',
            alpha=0.2
        )

        ax2 = ax1.twinx()

        ax2.plot(
            list(months),
            monthly_data['median_aqi'],
            marker='o'
        )

        ax2.set_ylabel('Median AQI')

        ax1.set_title(
            f'{selected_state}: Monthly Air Quality '
            f'(Selected Month: {date(2000, month, 1).strftime("%B")})'
        )

        fig.tight_layout()

        chart_path = os.path.join(
            DATA_DIR,
            f'{selected_state}_monthly_chart.png'
        )

        fig.savefig(chart_path, dpi=150)
        print(f'\nSelected-state chart saved: {chart_path}')

# --------------------------------------------------
# 10. SAVE CLEANED DATA
# --------------------------------------------------

df.to_csv(
    os.path.join(DATA_DIR, 'city_day_cleaned.csv'),
    index=False
)

print('\nAll 3 general charts and the selected-state chart are saved.')
print('Cleaned dataset saved as city_day_cleaned.csv.')

plt.show()
