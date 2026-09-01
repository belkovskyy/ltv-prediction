"""Построение сабмишена: прогноз GMV на 30 дней вперёд для всех пользователей.

Признаки считаются по всей истории до конца данных, модель применяется как есть,
отрицательные прогнозы зануляются (по правилам соревнования).
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import polars as pl

from data import make_supervised
from train import DATA_END, HORIZON


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\Загрузки\train.parquet")
    ap.add_argument("--model", default="model.txt")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    booster = lgb.Booster(model_file=args.model)

    df = make_supervised(args.data, DATA_END, HORIZON, with_target=False)
    feats = [c for c in df.columns if c != "user_id"]
    pred = np.expm1(booster.predict(df.select(feats).to_numpy()))
    pred = np.clip(pred, 0, None)

    sub = df.select("user_id").with_columns(pl.Series("predict", pred))
    sub.write_csv(args.out)
    print(f"Сабмишен на {sub.height:,} пользователей → {args.out}")
    print(sub.head())


if __name__ == "__main__":
    main()
