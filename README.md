# Olympic-medal-prediction-model-using-machine-learning
import pandas as pd
athletes = pd.read_csv("athlete_events.csv")
athletes.head()
athletes = athletes[athletes["Season"] == "Summer"]
def team_summary(data):
    return pd.Series({
        'team': data.iloc[0,:]["NOC"],
        'country': data.iloc[-1,:]["Team"],
        'year': data.iloc[0,:]["Year"],
        'events': len(data['Event'].unique()),
        'athletes': data.shape[0],
        'age': data["Age"].mean(),
        'height': data['Height'].mean(),
        'weight': data['Weight'].mean(),
        'medals': sum(~pd.isnull(data["Medal"]))
    })

team = athletes.groupby(["NOC", "Year"]).apply(team_summary)

team = team.reset_index(drop=True)
team = team.dropna()

team

def prev_medals(data):
    data = data.sort_values("year", ascending=True)
    data["prev_medals"] = data["medals"].shift(1)
    data["prev_3_medals"] = data.rolling(3, closed="left", min_periods=1).mean()["medals"]
    return data

team = team.groupby(["team"]).apply(prev_medals)
team = team.reset_index(drop=True)
team = team[team["year"] > 1960]
team = team.round(1)

team[team["team"] == "USA"]

team

team.to_csv("teams.csv", index=False)

import pandas as pd
teams = pd.read_csv("teams.csv")
teams

teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
teams.corr()["medals"]

import seaborn as sns
sns.lmplot(x='athletes',y='medals',data=teams,fit_reg=True, ci=None) 

sns.lmplot(x='age', y='medals', data=teams, fit_reg=True, ci=None) 

teams.plot.hist(y="medals")

teams[teams.isnull().any(axis=1)].head(20)

teams = teams.dropna()
teams.shape

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

# About 80% of the data
train.shape

# About 20% of the data
test.shape

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
predictors = ["athletes", "prev_medals"]

reg.fit(train[predictors], train["medals"])

predictions = reg.predict(test[predictors])
predictions.shape

test["predictions"] = predictions
test.loc[test["predictions"] < 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test["medals"], test["predictions"])
error

teams.describe()["medals"]

test["predictions"] = predictions
test[test["team"] == "USA"]

test[test["team"] == "IND"]

errors = (test["medals"] - predictions).abs()
error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team 
import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio.plot.hist()

error_ratio.sort_values()
