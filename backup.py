# %%
import pandas
dataframe = pandas.read_csv("dataset.csv")
dataframe['date'] = pandas.to_datetime(dataframe['date'], format='%Y-%m-%d %H:%M:%S')

# %%
dataframe.info()

# %%
dataframe.head()

# %%
dataframe.describe()

# %%
print(list(dict.fromkeys(dataframe.columns)))

# %%
# column_list = list(dict.fromkeys(dataframe.columns))
# dir_structure = "./graphs/feature_vs_date"
# DPI = 100
# for i in range(1 , 2):
# # for i in range(1 , len(column_list)):
#     try:
#         print(column_list[i] + "_vs_" + column_list[0])
#         import matplotlib.pyplot as graph
#         from matplotlib.pyplot import figure
#         figure(figsize=(32, 9), dpi=DPI)
#         graph.title(column_list[i] + " vs " + column_list[0])
#         graph.xlabel(column_list[0])
#         graph.ylabel(column_list[i])
#         graph.plot(dataframe[column_list[0]], dataframe[column_list[i]], linewidth=1)
#         graph.savefig(dir_structure + "/" + column_list[i] + "_vs_" + column_list[0] + ".png", dpi = DPI)
#         graph.show()
#     except FileNotFoundError:
#         from pathlib import Path
#         Path(dir_structure).mkdir(parents=True, exist_ok=True)
#         graph.savefig(dir_structure + "/" + column_list[i] + "_vs_" + column_list[0] + ".png", dpi = DPI)
#         graph.show()

# %%
# column_list = list(dict.fromkeys(dataframe.columns))
# dir_structure = "./graphs/feature_vs_week1"
# filtered_week1_dataset = dataframe.loc[ ("2016-01-11" < dataframe["date"]) & (dataframe["date"] <= "2016-01-18")]
# DPI = 100
# # for i in range(1 , 2):
# for i in range(1 , len(column_list)):
#     try:
#         print(column_list[i] + "_vs_" + column_list[0])
#         import matplotlib.pyplot as graph
#         from matplotlib.pyplot import figure
#         figure(figsize=(32, 9), dpi=DPI)
#         graph.title(column_list[i] + " vs " + column_list[0] + "(week 1)")
#         graph.xlabel(column_list[0])
#         graph.ylabel(column_list[i])
#         graph.plot(filtered_week1_dataset[column_list[0]], filtered_week1_dataset[column_list[i]], linewidth=1)
#         graph.savefig(dir_structure + "/" + column_list[i] + "_vs_" + column_list[0] + "_week1.png", dpi = DPI)
#         graph.show()
#     except FileNotFoundError:
#         from pathlib import Path
#         Path(dir_structure).mkdir(parents=True, exist_ok=True)
#         graph.savefig(dir_structure + "/" + column_list[i] + "_vs_" + column_list[0] + "_week1.png", dpi = DPI)
#         graph.show()

# %%
# column_list = list(dict.fromkeys(dataframe.columns))
# dir_structure = "./graphs/frequency"
# DPI = 100
# for i in range(1 , 2):
# # for i in range(1 , len(column_list)):
#     try:
#         print(column_list[i] + "_vs_" + column_list[0])
#         import matplotlib.pyplot as graph
#         from matplotlib.pyplot import figure
#         figure(figsize=(32, 9), dpi=DPI)
#         feature_frequency_dataframe = dataframe[column_list[i]].value_counts()
#         feature_frequency_dataframe.hist()
#     except FileNotFoundError:
#         from pathlib import Path
#         Path(dir_structure).mkdir(parents=True, exist_ok=True)

# %%
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
time_dataframe = dataframe
time_dataframe['date'] = time_dataframe['date'].dt.strftime("%H:%M:%S")
x = time_dataframe[['date', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']]
y = time_dataframe['Appliances']
linear_regression = LinearRegression()
linear_regression.fit(x, y)
print('Intercept: \n', linear_regression.intercept_)
print('Coefficients: \n', linear_regression.coef_)

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
print_model = model.summary()
print(print_model)

# %%



