import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import os

from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import pandas as pd

map_metric_names = {
    "fid": "FID",
    "wpd": "WPD",
    "mms": "MMS",
    "aog": "AOG",
    "coverage": "Coverage",
    "acpd": "ACPD",
    "apd": "APD",
    "density": "Density",
    # "fid-mean" : "FID",
    # "wpd-mean" : "WPD",
    # "mms-mean" : "MMS",
    # "aog-mean" : "AOG",
    # "coverage-mean" : "Coverage",
    # "acpd-mean" : "ACPD",
    # "apd-mean" : "APD",
    # "density-mean" : "Density",
}


def get_csv_files_with_titles(prefix, dir="./"):

    csv_files = []
    titles = []

    for filename in os.listdir(dir):
        if filename.endswith(".csv") and filename.startswith(prefix):
            csv_files.append(filename)
            titles.append(filename[:-4])

    return csv_files, titles


def _regisrer_radar_projection(
    numberOfMetrics: int = None,
    frame: str = "polygon",
):
    """
    based on the Matplotlib tutorial on radar charts
    with some modifications done by us:
    https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html
    """

    angles = np.linspace(0, 2 * np.pi, numberOfMetrics, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(numberOfMetrics)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "poylgone-chart"

        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(angles), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "polygon":
                return RegularPolygon(
                    (0.5, 0.5), numberOfMetrics, radius=0.5, edgecolor="k"
                )

        def draw(self, renderer):
            """Draw. If frame is polygon, make gridlines polygon-shaped"""
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = numberOfMetrics
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(numberOfMetrics),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)

    return angles


def _normalize_metrics(df_metrics: pd.DataFrame):

    models_column = df_metrics[list(df_metrics.columns)[0]]
    metrics_columns = df_metrics.drop(list(df_metrics.columns)[0], axis=1)

    normalized_metrics = metrics_columns.divide(metrics_columns.max())
    normalized_dataframe = pd.DataFrame(
        normalized_metrics, columns=metrics_columns.columns
    )

    return pd.concat([models_column, normalized_dataframe], axis=1)


def _transform_metrics(df_metrics, usedMetrics):

    df_copy = df_metrics.copy()

    model_names = list(df_metrics["Model"])
    model_names.remove("Real")

    for _metric in usedMetrics:

        df_metric_real = df_copy.loc[df_copy["Model"] == "Real"]
        _metric_real = df_metric_real[_metric].iloc[0]

        for model_name in model_names:
            df_metric_row = df_copy.loc[df_copy["Model"] == model_name]
            _metric_model = df_metric_row[_metric].iloc[0]

            if "fid" in _metric:
                if _metric_model > _metric_real:
                    df_metrics.loc[df_metrics["Model"] == model_name, _metric] = 1.0 - (
                        _metric_model - _metric_real
                    )
                else:
                    df_metrics.loc[df_metrics["Model"] == model_name, _metric] = 1.0 + (
                        _metric_real - _metric_model
                    )

            else:
                if _metric_model < _metric_real:
                    df_metrics.loc[df_metrics["Model"] == model_name, _metric] = 1.0 - (
                        _metric_real - _metric_model
                    )
                else:
                    df_metrics.loc[df_metrics["Model"] == model_name, _metric] = 1.0 + (
                        _metric_model - _metric_real
                    )

        df_metrics.loc[df_metrics["Model"] == "Real", _metric] = 1.0

    return df_metrics


# def _transform_density_metric(df_metrics):

#     df_density = df_metrics[["Model","density-mean"]].copy()

#     model_names = list(df_metrics["Model"])
#     model_names.remove("Real")

#     df_density_real = df_density.loc[df_density["Model"] == "Real"]
#     _density_real = df_density_real["density-mean"].iloc[0]

#     for model_name in model_names:
#         df_density_row = df_density.loc[df_density["Model"] == model_name]
#         _density = df_density_row["density-mean"].iloc[0]

#         if _density > _density_real:
#             df_metrics.loc[df_metrics["Model"] == model_name, "fid-mean"] = 1.0 - (_density - _density_real)
#         else:
#             df_metrics.loc[df_metrics["Model"] == model_name, "fid-mean"] = 1.0 + (_fid_real - _fid)

#     df_metrics.loc[df_metrics["Model"] == "Real", "fid-mean"] = 1.0

#     return df_metrics


def plot_metrics_on_polygone(
    df_metrics,
    usedMetrics: list = None,
    usedModels: list = None,
    frame: str = "polygon",
    title: str = None,
    figsize: Tuple[int, int] = (5, 5),
):

    df_metrics = _normalize_metrics(df_metrics=df_metrics)

    if usedMetrics is None:
        metrics_ = list(df_metrics.columns)
        metrics_.remove(list(df_metrics.columns)[0])

        metrics = [_metric for _metric in metrics_ if not "std" in _metric]

    else:
        metrics = usedMetrics

    numberOfMetrics = len(metrics)

    df_metrics = _transform_metrics(df_metrics=df_metrics, usedMetrics=metrics)
    # print(df_metrics[["Model","coverage-mean"]])

    if usedModels is None:
        models = list(df_metrics[list(df_metrics.columns)[0]])
    else:
        models = usedModels

    numberOfModels = len(models)

    angles = _regisrer_radar_projection(
        numberOfMetrics=numberOfMetrics,
        frame=frame,
    )

    fig, ax = plt.subplots(
        figsize=figsize, nrows=1, ncols=1, subplot_kw=dict(projection="poylgone-chart")
    )

    colors = plt.get_cmap("Dark2")(np.linspace(start=0.0, stop=1.0, num=numberOfModels))

    ax.set_rgrids(
        np.linspace(
            start=0.1,
            stop=df_metrics.select_dtypes(include=["number"]).max().max(),
            num=5,
        )
    )

    ax.set_title(
        title,
        weight="bold",
        size="medium",
        position=(0.5, 1.1),
        horizontalalignment="center",
        verticalalignment="center",
    )

    for i in range(numberOfModels):

        modelName = models[i]
        color = colors[i]
        metricData = df_metrics.loc[df_metrics["Model"] == modelName][metrics]

        _metricData = []

        for _metric in metrics:
            if not "std" in _metric:
                _metricData.append(float(metricData[_metric].iloc[0]))

        ax.plot(angles, _metricData, color=color, label=modelName)
        ax.fill(angles, _metricData, facecolor=color, alpha=0.25, label="_nolegend_")

    ax.set_varlabels([map_metric_names[_metric] for _metric in metrics])
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.15))

    fig.savefig(title + ".pdf")


if __name__ == "__main__":

    csv_file = "metrics_models.csv"
    title = "Metrics"

    plot_metrics_on_polygone(
        df_metrics=pd.read_csv(csv_file),
        title=title,
    )
