# %%
import pandas
dataframe = pandas.read_csv("dataset.csv")
dataframe['date'] =  pandas.to_datetime(dataframe['date'], format='%Y-%m-%d %H:%M:%S')

# %%
dataframe.info()

# %%
dataframe.head()

# %%
column_list = list(dict.fromkeys(dataframe.columns))
dir_structure = "./graphs/feature_vs_date"
DPI = 100
# for i in range(1 , 2):
for i in range(1 , len(column_list)):
    try:
        print(column_list[i] + "_vs_" + column_list[0])
        import matplotlib.pyplot as graph
        from matplotlib.pyplot import figure
        figure(figsize=(32, 9), dpi=DPI)
        graph.title(column_list[i] + " vs " + column_list[0])
        graph.xlabel(column_list[0])
        graph.ylabel(column_list[i])
        graph.plot(dataframe[column_list[0]], dataframe[column_list[i]], linewidth=1)
        graph.savefig(dir_structure + "/" + column_list[i] + "_vs_" + column_list[0] + ".png", dpi = DPI)
        graph.show()
    except FileNotFoundError:
        from pathlib import Path
        Path(dir_structure).mkdir(parents=True, exist_ok=True)
        graph.savefig(dir_structure + "/" + column_list[i] + "_vs_" + column_list[0] + ".png", dpi = DPI)
        graph.show()

# %%
column_list = list(dict.fromkeys(dataframe.columns))
dir_structure = "./graphs/feature_vs_week1"
filtered_week1_dataset = dataframe.loc[ ("2016-01-11" < dataframe["date"]) & (dataframe["date"] <= "2016-01-18")]
DPI = 100
# for i in range(1 , 2):
for i in range(1 , len(column_list)):
    try:
        print(column_list[i] + "_vs_" + column_list[0])
        import matplotlib.pyplot as graph
        from matplotlib.pyplot import figure
        figure(figsize=(32, 9), dpi=DPI)
        graph.title(column_list[i] + " vs " + column_list[0])
        graph.xlabel(column_list[0])
        graph.ylabel(column_list[i])
        graph.plot(filtered_week1_dataset[column_list[0]], filtered_week1_dataset[column_list[i]], linewidth=1)
        graph.savefig(dir_structure + "/" + column_list[i] + "_vs_" + column_list[0] + ".png", dpi = DPI)
        graph.show()
    except FileNotFoundError:
        from pathlib import Path
        Path(dir_structure).mkdir(parents=True, exist_ok=True)
        graph.savefig(dir_structure + "/" + column_list[i] + "_vs_" + column_list[0] + ".png", dpi = DPI)
        graph.show()


