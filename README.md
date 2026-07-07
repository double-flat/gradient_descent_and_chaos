# gd-and-chaos

`gd-and-chaos` は、多項式 $f(x) = (x^2 - 1)^2$ に対する勾配降下法から現れる一次元写像

$$
F_\eta(x) = (1 + 4\eta)x - 4\eta x^3
$$

に対して離散力学系・カオス理論的考察をするための実験用レポジトリです．
とくに，勾配降下法において学習率を変化させていった時に分岐現象やカオスの発生が生じることを観察します．。

## 環境構築

このプロジェクトは `uv` を使って依存関係を管理しています。

まず、リポジトリを clone したあとに依存関係を同期します。

```bash
uv sync
```

`uv sync` は `pyproject.toml` と `uv.lock` に基づいて `.venv` を作成し、
このプロジェクトと依存パッケージを使える状態にします。

Jupyter Notebook を使う場合は、次のように起動できます。

```bash
uv run jupyter notebook
```

依存パッケージを追加したい場合は、`uv add` を使います。

```bash
uv add package-name
```

## `bifurcation_tools` の使い方

`bifurcation_tools` は、一次元写像のパラメータを動かしながら軌道をサンプリングし、
分岐図を作るための道具です。写像は、動かすパラメータを第 2 位置引数として
受け取る形でも、`parameter_name` と同じ名前のキーワード引数として受け取る形でも
使えます。

例えば、ロジスティック写像の分岐図は次のように作れます。

```python
import numpy as np
from gd_and_chaos.bifurcation_tools import (
    BifurcationExperimentConfig,
    plot_bifurcation_diagram,
    run_bifurcation_experiment,
)

def logistic_map(x, a):
    return a * x * (1 - x)

config = BifurcationExperimentConfig(
    parameter_values=np.linspace(2.5, 4.0, 10000),
    parameter_name="a",
    x0=0.2,
    n_burn=3000,
    n_iter=300,
)

parameter_samples, orbit_samples = run_bifurcation_experiment(logistic_map, config)
fig, ax = plot_bifurcation_diagram(
    parameter_samples,
    orbit_samples,
    config,
    title="Logistic map bifurcation diagram",
    save=True,
    save_path="out/bifurcation_diagram_logistic.png",
)
```

`x0` には単一の初期値だけでなく、複数の初期値も指定できます。複数の初期値を
指定した場合は、それぞれのパラメータ値に対して各初期値から軌道をサンプリングします。

主な設定項目は次の通りです。

- `parameter_values`: 掃引するパラメータ値の配列。
- `parameter_name`: x 軸ラベル、およびキーワード引数として写像関数を呼ぶときの名前。
- `x0`: 初期値。単一の値、またはリスト・配列を指定できます。
- `n_burn`: 過渡的な軌道として捨てる反復回数。
- `n_iter`: burn-in 後にサンプリングする反復回数。
- `sample_every`: 軌道を何ステップごとに記録するか。
- `escape_radius`: 発散または非有限値になった軌道を打ち切るための閾値。
- `fixed_kwargs`: `map_fn` に毎回渡す追加のキーワード引数。

複数のパラメータを持つ写像には、`fixed_kwargs` で固定パラメータを渡せます。

```python
import numpy as np
from gd_and_chaos.bifurcation_tools import (
    BifurcationExperimentConfig,
    run_bifurcation_experiment,
)

def shifted_logistic_map(x, a, shift):
    return a * x * (1 - x) + shift

config = BifurcationExperimentConfig(
    parameter_values=np.linspace(2.5, 4.0, 3000),
    parameter_name="a",
    x0=0.2,
    n_burn=1000,
    n_iter=200,
    fixed_kwargs={"shift": 0.01},
)

parameter_samples, orbit_samples = run_bifurcation_experiment(
    shifted_logistic_map,
    config,
)
```

## Lyapunov 指数の計算

Lyapunov 指数のスキャンも、同じく `gd_and_chaos` パッケージから利用できます。

```python
import numpy as np
from gd_and_chaos import LyapunovExperimentConfig, run_lyapunov_scan, plot_lyapunov_exponent

config = LyapunovExperimentConfig(
    etas=np.linspace(0.001, 0.8, 2000),
    x0=0.123456789,
    n_burn=1000,
    n_iter=2000,
)

etas, lambdas = run_lyapunov_scan(config)
plot_lyapunov_exponent(etas, lambdas)
```

## コマンドラインからの実行

```bash
uv run python main.py --eta-min 0.001 --eta-max 0.8 --eta-count 3000
```
