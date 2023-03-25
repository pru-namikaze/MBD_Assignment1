# %%
import datetime as datetime
import pandas as pandas
import numpy as numpy
dataframe = pandas.read_csv("dataset.csv")
dataframe["date"] = pandas.to_datetime(dataframe['date'], format='%Y-%m-%d %H:%M:%S')
dataframe["datetimestamp"] = dataframe["date"]
dataframe["NSM"] = (dataframe["date"] - pandas.to_datetime(dataframe['date'].dt.date, format='%Y-%m-%d')).dt.total_seconds()
dataframe["week_status"] = dataframe["date"].apply(lambda date: date.weekday() <= 4)
dataframe["day_of_week"] = dataframe["date"].dt.day_name()
dataframe.info()
from pathlib import Path  
output_dataset_path = Path("./dataframeResult/modifiedDataset.csv")
output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
dataframe.to_csv(Path(output_dataset_path))  
dataframe.head()

# %%
import numpy as numpy
dataframe.plot.line(
    x = "datetimestamp", 
    y = "Appliances", 
    figsize = (48,9), 
    title = "Appliances energy consumption vs datetimestamp", 
    ylabel = "Appliances energy consumption(in Wh)", 
    xlabel = "datetimestamp"
)

dataframe.where((dataframe["datetimestamp"] - dataframe["datetimestamp"].min()) < datetime.timedelta(days = 7)).plot.line(
    x = "datetimestamp", 
    y = "Appliances", 
    figsize = (48,9), 
    title = "Appliances energy consumption vs datetimestamp", 
    ylabel = "Appliances energy consumption(in Wh)", 
    xlabel = "datetimestamp"
)

# %%



