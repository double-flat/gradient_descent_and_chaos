from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BifurcationExperimentConfig:
    """一次元・一パラメータ力学系の分岐図を描画するための設定データクラス

    attributes:
        parameter_values: 分岐図を描画するためのパラメータの値の配列
        parameter_name: パラメータの名前（デフォルトは "eta"）
        x0: 初期値。float または初期値のリストを指定できる
        n_burn: 軌道の初期部分を破棄するための反復回数
        n_iter: 軌道のサンプリングを行う反復回数
        sample_every: 軌道のサンプリング間隔
        escape_radius: 発散判定の閾値。この値を超えるか非有限値になった軌道は打ち切る
        fixed_kwargs: map_fnに渡す固定のキーワード引数の辞書（デフォルトは空の辞書）
    """

    parameter_values: np.ndarray | None = None
    parameter_name: str = "eta"
    x0: float | list[float] | np.ndarray = 0.1
    n_burn: int = 2000
    n_iter: int = 5000
    sample_every: int = 1
    escape_radius: float = 1e6
    fixed_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.parameter_values is None:
            # デフォルトのパラメータ値の範囲を設定
            self.parameter_values = np.linspace(0.001, 0.8, 3000)
        else:
            self.parameter_values = np.asarray(self.parameter_values, dtype=float)

        self.x0 = np.atleast_1d(np.asarray(self.x0, dtype=float))

        if self.fixed_kwargs is None:
            self.fixed_kwargs = {}


def _build_map_caller(map_fn, parameter_name, fixed_kwargs=None):
    """ユーザー定義の関数を呼び出すためのラッパー関数を構築する。
    この関数は、ユーザー定義の関数がパラメータを位置引数として受け取るか、キーワード引数として受け取るかを判定し、適切な呼び出し方法を提供する。
    Args:
        map_fn: ユーザー定義の関数
        parameter_name: パラメータの名前
        fixed_kwargs: map_fnに渡す固定のキーワード引数の辞書（デフォルトはNone）
    Returns:
        ラッパー関数
    """

    fixed_kwargs = {} if fixed_kwargs is None else dict(fixed_kwargs)

    # 関数シグネチャの解析は高コストなので、反復ループの外側で一度だけ行う。
    # ここで map_fn(x, parameter_value) 形式にそろえた呼び出し関数を作る。
    signature = inspect.signature(map_fn)
    parameters = signature.parameters

    if parameter_name in parameters:
        parameter_info = parameters[parameter_name]

        # 典型例は logistic_map(x, a) や gd_map(x, eta)。
        # この形なら追加の dict を作らず、そのまま位置引数で呼べるので最速。
        positional_parameters = [
            parameter
            for parameter in parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        parameter_is_second_positional = (
            len(positional_parameters) >= 2
            and positional_parameters[1].name == parameter_name
            and parameter_info.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )

        if parameter_is_second_positional:
            if fixed_kwargs:
                return lambda x, parameter_value: map_fn(
                    x, parameter_value, **fixed_kwargs
                )
            return map_fn

        # パラメータ名は一致するが、第2位置引数ではない場合。
        # 例: map_fn(x, *, eta) のような keyword-only 引数をサポートする。
        if parameter_info.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            if fixed_kwargs:
                return lambda x, parameter_value: map_fn(
                    x, **{parameter_name: parameter_value, **fixed_kwargs}
                )
            return lambda x, parameter_value: map_fn(x, **{parameter_name: parameter_value})
        if parameter_info.kind is inspect.Parameter.POSITIONAL_ONLY:
            if fixed_kwargs:
                return lambda x, parameter_value: map_fn(
                    x, parameter_value, **fixed_kwargs
                )
            return map_fn

    # parameter_name がシグネチャに無い場合でも、2引数関数として呼べるなら許可する。
    # 例: lambda x, r: r * x * (1 - x) を parameter_name="a" で使うケース。
    positional_count = sum(
        1
        for parameter in parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values()):
        positional_count = max(positional_count, 2)

    if positional_count >= 2:
        if fixed_kwargs:
            return lambda x, parameter_value: map_fn(x, parameter_value, **fixed_kwargs)
        return map_fn

    raise TypeError(
        "The map function must accept the parameter as either a positional "
        "argument or a named keyword argument."
    )


def run_bifurcation_experiment(map_fn, config: BifurcationExperimentConfig):
    """分岐の実験を実行する
    Args:
        map_fn: ユーザー定義の関数
        config: BifurcationExperimentConfig オブジェクト
    Returns:
        parameter_samples: パラメータのサンプル値の配列
        orbit_samples: 軌道のサンプル値の配列
    """

    map_caller = _build_map_caller(
        map_fn,
        config.parameter_name,
        fixed_kwargs=config.fixed_kwargs,
    )
    samples_per_parameter = (config.n_iter + config.sample_every - 1) // config.sample_every
    n_initial_values = len(config.x0)
    total_samples = len(config.parameter_values) * n_initial_values * samples_per_parameter
    parameter_samples = np.empty(total_samples, dtype=float)
    orbit_samples = np.empty(total_samples, dtype=float)
    sample_index = 0

    for parameter_value in config.parameter_values:
        for initial_value in config.x0:
            x = initial_value

            for _ in range(config.n_burn):
                x = map_caller(x, parameter_value)
                if not np.isfinite(x) or abs(x) > config.escape_radius:
                    break
            else:
                for step in range(config.n_iter):
                    x = map_caller(x, parameter_value)
                    if not np.isfinite(x) or abs(x) > config.escape_radius:
                        break
                    if step % config.sample_every == 0:
                        parameter_samples[sample_index] = parameter_value
                        orbit_samples[sample_index] = x
                        sample_index += 1

    return parameter_samples[:sample_index], orbit_samples[:sample_index]


def plot_bifurcation_diagram(
    parameter_values,
    orbit_values,
    config: BifurcationExperimentConfig,
    title: str | None = None,
    save: bool = False,
    save_path: str | None = None,
):
    """分岐図を描画する
    Args:
        parameter_values: パラメータのサンプル値の配列
        orbit_values: 軌道のサンプル値の配列
        config: BifurcationExperimentConfig オブジェクト
        title: グラフのタイトル（デフォルトは None）
        save: グラフを保存するかどうか（デフォルトは False）
        save_path: グラフを保存するパス（デフォルトは None）
    Returns:
        fig: matplotlib.figure.Figure オブジェクト
        ax: matplotlib.axes.Axes オブジェクト
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(parameter_values, orbit_values, ",", color='k', alpha=0.3, markersize=0.5)
    # ax.scatter(parameter_values, orbit_values, s=0.3, color="black")
    # ax.plot(eta_values, x_values, ",", color='k', alpha=0.3, markersize=0.5)
    ax.set_xlabel(config.parameter_name)
    ax.set_ylabel("x")
    plt.grid(True, alpha=0.3)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Bifurcation diagram")
    fig.tight_layout()

    if save and save_path is not None:
        fig.savefig(save_path, dpi=300)

    plt.show()

    return fig, ax
