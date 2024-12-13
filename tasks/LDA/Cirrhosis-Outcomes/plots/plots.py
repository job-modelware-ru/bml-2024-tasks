import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = pd.read_csv("data.txt")
deep_pink_hex = "#d71868"
light_pink_hex = "#ffbdd8"
def create_pink_palette(num_colors, deep_pink_hex="#d71868", light_pink_hex="#ffbdd8"):
    if num_colors <= 0:
        return []

    pink_colors = []
    for i in range(num_colors):
        r = int((1 - i / (num_colors - 1)) * int(deep_pink_hex[1:3], 16) + i / (num_colors - 1) * int(light_pink_hex[1:3], 16))
        g = int((1 - i / (num_colors - 1)) * int(deep_pink_hex[3:5], 16) + i / (num_colors - 1) * int(light_pink_hex[3:5], 16))
        b = int((1 - i / (num_colors - 1)) * int(deep_pink_hex[5:7], 16) + i / (num_colors - 1) * int(light_pink_hex[5:7], 16))
        pink_colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return pink_colors
data['Age_Years'] = data['Age'] / 365.25
# Гистограммы числовых переменных
numeric_cols = ['N_Days', 'Age_Years', 'Bilirubin', 'Stage']
for col in numeric_cols:
    plt.figure(figsize=(8, 6))

    sns.histplot(data[col], kde=True, color=deep_pink_hex)
    plt.title(f'Histogram of {col}')
    plt.show()

# Диаграммы разброса
n_drugs = len(data['Drug'].unique())
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Bilirubin', y='Stage', hue='Drug', data=data, palette=create_pink_palette(n_drugs))
plt.title('Bilirubin vs. Stage by Drug')
plt.show()

# Столбчатые диаграммы
cols = ['Sex', 'Drug', 'Status']
col: str
for col in cols:
    n = len(data[col].unique())
    plt.figure(figsize=(8, 6))
    sns.countplot(x=data[col], data=data, palette=create_pink_palette(n))
    plt.title(f'Count of Patients by {col}')
    plt.show()





