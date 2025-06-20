# postpartum_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Streamlit Page Configuration
st.set_page_config(page_title="Postpartum Depression Dashboard", layout="wide")

st.title("🧠 Postpartum Depression Dashboard")
st.markdown("""
This dashboard explores associations between **postpartum anxiety** and different clinical and demographic variables using data from a Kenyan hospital.
""")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Data/post natal data.csv")
    df["Anxiety_binary"] = df["Feeling anxious"].apply(lambda x: 1 if x == "Yes" else 0)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Irritable towards baby & partner"].fillna(df["Irritable towards baby & partner"].mode()[0], inplace=True)
    df["Problems concentrating or making decision"].fillna(df["Problems concentrating or making decision"].mode()[0], inplace=True)
    df["Feeling of guilt"].fillna(df["Feeling of guilt"].mode()[0], inplace=True)
    return df

post_natal = load_data()

# Sidebar filter
st.sidebar.header("🔎 Filters")
age_group = st.sidebar.multiselect("Select Age Groups", options=post_natal['Age'].unique(), default=post_natal['Age'].unique())

filtered_data = post_natal[post_natal['Age'].isin(age_group)]

# Age Distribution Plot
st.subheader("📊 Age Distribution of Participants")

age_counts = filtered_data['Age'].value_counts()
age_percent = (age_counts / age_counts.sum()) * 100
age_percent = age_percent.sort_values(ascending=True)

fig1, ax = plt.subplots(figsize=(8, 4))
fig1.patch.set_facecolor('#FAF5EE')
ax.set_facecolor('#FAF5EE')
bars = ax.barh(age_percent.index, age_percent, color='#278783')
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', ha='left', fontsize=10)
ax.set_title('Age Distribution of Participants', pad=20)
ax.set_xticks([])
ax.spines[['top', 'right', 'bottom']].set_visible(False)
st.pyplot(fig1)

# Chi-square heatmap section
st.subheader("🔥 Association Strength (Chi-square Test) with Postpartum Anxiety")

predictors = ['Age', 'Feeling sad or Tearful', 'Irritable towards baby & partner',
              'Trouble sleeping at night', 'Problems concentrating or making decision',
              'Overeating or loss of appetite', 'Feeling of guilt',
              'Problems of bonding with baby', 'Suicide attempt']

chi2_p_values = {}
for col in predictors:
    contingency = pd.crosstab(filtered_data[col], filtered_data['Anxiety_binary'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    chi2_p_values[col] = p

pval_df = pd.DataFrame.from_dict(chi2_p_values, orient='index', columns=['p-value'])
pval_df['-log10(p-value)'] = -np.log10(pval_df['p-value'])
pval_df_sorted = pval_df.sort_values('-log10(p-value)', ascending=False)

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(pval_df_sorted[['-log10(p-value)']], annot=True, cmap='YlGnBu', cbar_kws={'label': '-log10(p-value)'}, ax=ax2)
ax2.set_title('Chi-square Association Strength with Postpartum Anxiety')
ax2.tick_params(axis='y', labelrotation=0)
st.pyplot(fig2)

# Show raw data toggle
with st.expander("🗃️ View Raw Data"):
    st.dataframe(filtered_data)

# Footer
st.markdown("""---""")
st.markdown("Developed for exploratory insights on postpartum depression in Kenya using clinical screening data.")

