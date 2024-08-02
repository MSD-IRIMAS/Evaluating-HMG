import pandas as pd
from typing import *

from hmgmetrics.quality import FID, DENSITY, AOG
from hmgmetrics.diversity import APD, ACPD, COVERAGE, WPD, MMS


class METRIC_CALCULATOR:
    def __init__(self, args: dict, output_dir: str = "./") -> None:
        self.str_to_metric = {
            "fid": FID,
            "density": DENSITY,
            "apd": APD,
            "acpd": ACPD,
            "coverage": COVERAGE,
            "mms": MMS,
            "aog": AOG,
            "wpd": WPD,
        }

        self.metric_to_str = {
            FID: "fid",
            DENSITY: "density",
            APD: "apd",
            ACPD: "acpd",
            COVERAGE: "coverage",
            MMS: "mms",
            AOG: "aog",
            WPD: "wpd",
        }

        self.output_dir = output_dir
        self.args = args

        self.used_metrics_names = list(self.args.keys())
        self.used_metrics = [
            self.str_to_metric[metric_name] for metric_name in self.used_metrics_names
        ]

        self.df_results = pd.DataFrame(columns=["On"] + self.used_metrics_names)

    def get_metrics_csv(
        self,
        xgenerated=None,
        ygenerated=None,
        xreal=None,
        yreal=None,
    ):
        # on real samples
        row_to_add = {"On": "real"}

        for METRIC in self.used_metrics:
            print(self.metric_to_str[METRIC])
            if "metric_params" in self.args[self.metric_to_str[METRIC]].keys():
                metric = METRIC(
                    classifier=self.args[self.metric_to_str[METRIC]].get(
                        "classifier", None
                    ),
                    batch_size=self.args[self.metric_to_str[METRIC]].get(
                        "batch_size", None
                    ),
                    **self.args[self.metric_to_str[METRIC]]["metric_params"]
                )
            else:
                metric = METRIC(
                    classifier=self.args[self.metric_to_str[METRIC]].get(
                        "classifier", None
                    ),
                    batch_size=self.args[self.metric_to_str[METRIC]].get(
                        "batch_size", None
                    ),
                )

            # if (
            #     self.metric_to_str[METRIC] != "aog"
            # ):
            metric_value = metric.calculate(
                xreal=xreal,
                yreal=yreal,
            )
            # else:
            #     metric_value = "N/A"

            row_to_add[self.metric_to_str[METRIC]] = metric_value

        self.df_results.loc[len(self.df_results)] = row_to_add

        # on generated samples
        row_to_add = {"On": "generated"}

        for METRIC in self.used_metrics:
            print(self.metric_to_str[METRIC])
            if "metric_params" in self.args[self.metric_to_str[METRIC]].keys():
                metric = METRIC(
                    classifier=self.args[self.metric_to_str[METRIC]].get(
                        "classifier", None
                    ),
                    batch_size=self.args[self.metric_to_str[METRIC]].get(
                        "batch_size", None
                    ),
                    **self.args[self.metric_to_str[METRIC]]["metric_params"]
                )
            else:
                metric = METRIC(
                    classifier=self.args[self.metric_to_str[METRIC]].get(
                        "classifier", None
                    ),
                    batch_size=self.args[self.metric_to_str[METRIC]].get(
                        "batch_size", None
                    ),
                )

            metric_value = metric.calculate(
                xgenerated=xgenerated,
                ygenerated=ygenerated,
                xreal=xreal,
                yreal=yreal,
            )

            row_to_add[self.metric_to_str[METRIC]] = metric_value

        self.df_results.loc[len(self.df_results)] = row_to_add

        self.df_results.to_csv(self.output_dir + "metrics.csv", index=False)

        return self.df_results
