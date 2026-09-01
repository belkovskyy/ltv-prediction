"""Обучение LightGBM для прогноза GMV пользователя на 30 дней.

Метрика соревнования — RMSLE. Обучаемся в пространстве log1p(target) с обычным
MSE: RMSE в лог-пространстве и есть RMSLE. Предсказания возвращаем через expm1
и зануляем отрицательные.
"""
from __future__ import annotations

import argparse
import datetime as dt

import lightgbm as lgb
import numpy as np
import polars as pl

from data import make_supervised

# Границы данных: история до 2026-02-13 включительно, прогноз на 30 дней вперёд.
DATA_END = dt.date(2026, 2, 14)          # эксклюзивная граница истории
HORIZON = 30
# Валидация: последнее полностью размеченное 30-дневное окно.
VALID_CUTOFF = DATA_END - dt.timedelta(days=HORIZON)  # 2026-01-15
# Обучение: несколько более ранних срезов дают больше примеров отображения.
TRAIN_CUTOFFS = [
    VALID_CUTOFF - dt.timedelta(days=30 * k) for k in range(1, 5)
]

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=100,
    n_jobs=-1,
    verbose=-1,
)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def _xy(df: pl.DataFrame):
    feats = [c for c in df.columns if c not in ("user_id", "target")]
    x = df.select(feats).to_numpy()
    y = df.get_column("target").to_numpy() if "target" in df.columns else None
    return x, y, feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\Загрузки\train.parquet")
    ap.add_argument("--model-out", default="model.txt")
    args = ap.parse_args()

    print("Сборка обучающих срезов...")
    train_parts = [make_supervised(args.data, c, HORIZON) for c in TRAIN_CUTOFFS]
    train = pl.concat(train_parts)
    valid = make_supervised(args.data, VALID_CUTOFF, HORIZON)
    print(f"train: {train.height:,} строк, valid: {valid.height:,} строк")

    x_tr, y_tr, feats = _xy(train)
    x_va, y_va, _ = _xy(valid)

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        x_tr, np.log1p(y_tr),
        eval_set=[(x_va, np.log1p(y_va))],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred_va = np.expm1(model.predict(x_va))
    print(f"\nВалидация ({VALID_CUTOFF} +{HORIZON}д):")
    print(f"  RMSLE = {rmsle(y_va, pred_va):.5f}")
    print(f"  baseline (все нули) RMSLE = {rmsle(y_va, np.zeros_like(y_va)):.5f}")
    print(f"  baseline (среднее)  RMSLE = {rmsle(y_va, np.full_like(y_va, y_va.mean())):.5f}")

    imp = sorted(zip(feats, model.feature_importances_), key=lambda t: -t[1])[:12]
    print("\nТоп признаков:")
    for name, val in imp:
        print(f"  {name}: {val}")

    model.booster_.save_model(args.model_out)
    print(f"\nМодель сохранена → {args.model_out}")


if __name__ == "__main__":
    main()
