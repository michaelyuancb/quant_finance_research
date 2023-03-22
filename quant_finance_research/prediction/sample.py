import numpy as np


class SampleBase:

    def __init__(self, purge, time_col='time_id'):
        self.purge = purge
        self.time_col = time_col
        self.time_list = None

    def generate_time_idx(self, df):
        time_idx = df[self.time_col].values
        self.time_idx = np.sort(np.unique(time_idx))

    def purge_seq(self, df):
        df = df[df[self.time_col] < self.time_idx[-self.purge]]
        return df


class Sample_Horizon(SampleBase):

    def __init__(self, purge, horizon):
        super().__init__(purge)
        self.horizon = horizon

    def sample(self, df):
        self.generate_time_idx(df)
        if self.purge + self.horizon > self.time_idx.shape[0]:
            raise ValueError(f"The timestamp len={self.time_idx.shape[0]}, "
                             f"while the horizon+purge={self.purge + self.horizon}.")
        df = self.purge_seq(df)
        df = df[df[self.time_col] >= self.time_idx[-self.horizon-self.purge]]
        return df