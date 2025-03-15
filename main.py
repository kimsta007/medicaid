import numpy as np
import scipy.stats as stats
import altair as alt
import pandas as pd

def cdf(s):
    return stats.norm.cdf(s, 0, 1)

class SurpriseGroup:
    def __init__(self, df, name, rate_key, population_key):
        self.df = df
        self.name = name
        self.rate_key = rate_key
        self.population_key = population_key

        self.rate_mean = self.df[self.rate_key].mean()
        
        self.std_dev = self.df[self.rate_key].std()

    def calculate(self):
        # calculate the z-score
        self.zScore = (self.df[self.rate_key] - self.rate_mean) / self.std_dev

        # calculate likelihood (pMs)
        test_statistic = (self.df[self.rate_key] - self.rate_mean) / (
            self.std_dev
            / np.sqrt(self.df[self.population_key] / self.df[self.population_key].sum())
        )

        pMs = 2 * (1 - cdf(abs(test_statistic)))
        self.pMs = np.maximum(pMs, 1e-10)

        # calculate kl divergence
        self.kl = self.pMs * np.log(self.pMs) / np.log(2)

        # calculate surprise
        surprise = np.where(
            self.df[self.rate_key] - self.rate_mean > 0,
            np.abs(self.kl),
            -1 * np.abs(self.kl),
        )
        self.surprise = np.where(self.df[self.rate_key] == 0, 0, surprise)

        # put z-score back into the dataframe
        self.df[self.name + "_zScore"] = self.zScore

        # put surprise back into the dataframe
        self.df[self.name + "_surprise"] = self.surprise


class Surprise:
    def __init__(
        self,
        df,
        groups=[],
        rate_keys=[]
    ):
        self.df = df
        self.groups = groups
        self.rate_keys = rate_keys

        self.surprise_groups = []
        for idx, group in enumerate(self.groups):
            self.surprise_groups.append(
                SurpriseGroup(
                    self.df, group, self.rate_keys[idx], 'pop'
                )
            )

    def calculate(self):
        for group in self.surprise_groups:
            group.calculate()

        self.surprise_keys = [
            group.name + "_surprise" for group in self.surprise_groups
        ]

    def funnel_plot(self, key:str, data: pd.DataFrame = None, axis: str = "zScore"):
        if data is None or data.empty:
            _df = self.df.copy()
        else: 
            _df = data.copy()

        key_rate = f'{key}_rate'
        key_pop = 'pop'
        key_zScore = f'{key}_zScore'
        key_surprise = f'{key}_surprise'
        key_axis = f'{key}_{axis}'

        rate_mean = _df[key_rate].mean()
        std_dev = _df[key_rate].std()
        totalPopulation = _df[key_pop].sum()
        max_population = _df[key_pop].max()
        min_population = 20000

        funnel_df = pd.DataFrame()
        funnel_df["population"] = np.linspace(min_population, max_population, 1000)
        funnel_df['n'] = funnel_df["population"] / totalPopulation
        funnel_df['ci'] = (1.96 * std_dev) / (funnel_df['n'] ** 0.5)
        funnel_df['lcl95'] = rate_mean - funnel_df['ci']
        funnel_df['ucl95'] = rate_mean + funnel_df['ci']


        g_df = funnel_df

        max_surprise = _df[key_surprise].max()
        min_surprise = _df[key_surprise].min()
        abs_max = max(abs(max_surprise), abs(min_surprise))

        # Shaded area between confidence intervals
        ci_band = alt.Chart(g_df).mark_area(opacity=0.3, color='gray').encode(
            x='population',
            y='lcl95',
            y2='ucl95'
        )

        # Lower confidence interval line
        ci_lower = alt.Chart(g_df).mark_line(color='black').encode(
            x='population',
            y='lcl95'
        )

        #Upper confidence interval line
        ci_upper = alt.Chart(g_df).mark_line(color='black').encode(
            x='population',
            y='ucl95'
        )

        chart = alt.Chart(_df).mark_circle(size=60).encode(
            x=key_pop,
            y=key_axis,
            color=alt.Color(key_surprise, scale=alt.Scale(scheme='redblue', domainMid=0, domain=[-abs_max, abs_max])),
            tooltip=['name', 'state', key_surprise, key_rate, key_pop]
        ).properties(
            width=800,
            height=400
        ).interactive()
        return (ci_band + ci_lower + ci_upper + chart)
