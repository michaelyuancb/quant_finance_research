from quant_finance_research.prediction.sample import *
from quant_finance_research.utils import *


class DebugSample_Horizon:

    def __init__(self):
        self.df, self.df_column = get_example_df()
        self.dfval = self.df.copy()
        self.sampler = Sample_Horizon(purge=1, horizon=3)

    def debug_sample(self):
        print(self.df)
        df = self.sampler.sample(self.df)
        print(df)
        assert (self.dfval.values == self.df.values).all()


if __name__ == "__main__":
    DebugSample_Horizon().debug_sample()