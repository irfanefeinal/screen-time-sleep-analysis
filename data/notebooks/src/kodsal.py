import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# 1) DATA LOAD
# =========================
df = pd.read_csv("data.csv", sep=";")

# kolon isimlerini temizle
df.columns = df.columns.str.strip().str.lower()

print("\nColumns:", df.columns.tolist())

# kolonları otomatik bul
date_col = [col for col in df.columns if "date" in col][0]
screen_col = [col for col in df.columns if "screen" in col][0]
sleep_col = [col for col in df.columns if "sleep" in col][0]

# =========================
# 2) CLEANING
# =========================

# date düzeltme
df["date"] = pd.to_datetime(
    df[date_col].astype(str).str.strip() + ".2026",
    format="%d.%b.%Y",
    errors="coerce"
)

# screen time → dakika
def convert_screen(x):
    x = str(x).strip()
    h = int(x.split("h")[0])
    m = int(x.split("h")[1].replace("m", "").strip())
    return h * 60 + m

df["screen_minutes"] = df[screen_col].apply(convert_screen)

# sleep → saat
df["sleep_hours"] = (
    df[sleep_col]
    .astype(str)
    .str.replace("h", "", regex=False)
    .str.strip()
    .astype(float)
)

# temizle
df = df.dropna()
df = df.sort_values("date").reset_index(drop=True)

print("\n=== CLEAN DATA ===")
print(df.head())

# =========================
# 3) FEATURE ENGINEERING
# =========================
df["weekday"] = df["date"].dt.weekday
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

# =========================
# 4) DESCRIPTIVE STATS
# =========================
print("\n=== STATS ===")
print(df[["screen_minutes", "sleep_hours"]].describe())

# =========================
# 5) CORRELATION
# =========================
corr = df["screen_minutes"].corr(df["sleep_hours"])
print("\nCorrelation:", corr)

# =========================
# 6) SCATTER
# =========================
plt.figure()
plt.scatter(df["screen_minutes"], df["sleep_hours"])
plt.xlabel("Screen Time (minutes)")
plt.ylabel("Sleep Hours")
plt.title("Screen Time vs Sleep")
plt.show()

# =========================
# 7) TIME SERIES
# =========================
plt.figure()
plt.plot(df["date"], df["screen_minutes"], label="Screen Time")
plt.plot(df["date"], df["sleep_hours"], label="Sleep")
plt.legend()
plt.xticks(rotation=45)
plt.title("Time Series")
plt.show()

# =========================
# 8) HYPOTHESIS TEST
# =========================
mean_screen = df["screen_minutes"].mean()

high = df[df["screen_minutes"] > mean_screen]["sleep_hours"]
low = df[df["screen_minutes"] <= mean_screen]["sleep_hours"]

t_stat, p_val = ttest_ind(high, low, equal_var=False)

print("\n=== T-TEST ===")
print("p-value:", p_val)

# =========================
# 9) MACHINE LEARNING
# =========================
X = df[["screen_minutes", "is_weekend"]]
y = df["sleep_hours"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== ML RESULTS ===")
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# =========================
# 10) REGRESSION PLOT
# =========================
simple_X = df[["screen_minutes"]]
simple_y = df["sleep_hours"]

simple_model = LinearRegression()
simple_model.fit(simple_X, simple_y)

plt.figure()
plt.scatter(simple_X, simple_y)
plt.plot(simple_X, simple_model.predict(simple_X))
plt.xlabel("Screen Time")
plt.ylabel("Sleep")
plt.title("Regression Line")
plt.show()

# =========================
# FINAL PRINT
# =========================
print("\n=== DONE ===")
print("EDA + Hypothesis + ML tamamlandı.")