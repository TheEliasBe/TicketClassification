import pandas as pd
import numpy as np


df_2022 = pd.read_csv("../data/01_raw/2022/contents.csv")
df_2023 = pd.read_csv("../data/01_raw/2023/contents.csv")
df_2021 = pd.read_csv("../data/01_raw/2021/contents.csv")
df_text = pd.concat([df_2021, df_2022, df_2023])


df_2021 = pd.read_csv("../data/01_raw/2021/tickets.csv")
df_2022 = pd.read_csv("../data/01_raw/2022/tickets.csv")
df_2023 = pd.read_csv("../data/01_raw/2023/tickets.csv")
df_ticket = pd.concat([df_2021, df_2022, df_2023])


df = pd.merge(df_text, df_ticket, on="ID", how="inner")

df[df["Abteilung Label"].str.contains("Basis")]

df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
print("Total since 2021", len(df))
print("Ticket", len(df.dropna(subset=["Ticket Label"])))
print("Abteilung", len(df.dropna(subset=["Abteilung Label"])))
print("Produkt", len(df.dropna(subset=["Produkt Label"])))