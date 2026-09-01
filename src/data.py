"""Построение признаков и обучающей выборки из дневной активности.

Данные — разреженные дневные сводки по пользователю (одна строка на активный
день). Признаки считаются по окну истории до даты cutoff, целевая переменная —
сумма gmv за horizon дней начиная с cutoff.
"""
from __future__ import annotations

import datetime as dt
from typing import List

import polars as pl

# Числовые признаки, суммируемые по окну.
_SUM_COLS = [
    "search", "cat", "searches", "to_cart", "to_ord",
    "gmv", "gmv_search", "gmv_cat",
    "search_to_cart", "search_to_ord", "cat_to_cart", "cat_to_ord",
]
# Окна (в днях) для «свежих» агрегатов.
_WINDOWS = [7, 14, 30, 60, 90]


def all_user_ids(path: str) -> pl.DataFrame:
    """Все user_id из данных — сабмишен нужен для каждого."""
    return pl.scan_parquet(path).select("user_id").unique().collect()


def _window_sum(col: str, cutoff: dt.date, days: int) -> pl.Expr:
    thr = cutoff - dt.timedelta(days=days)
    return (
        pl.when(pl.col("event_date") >= thr)
        .then(pl.col(col))
        .otherwise(0)
        .sum()
        .alias(f"{col}_{days}d")
    )


def build_features(path: str, cutoff: dt.date, start: dt.date | None = None) -> pl.DataFrame:
    """Признаки по активности в [start, cutoff) для каждого пользователя."""
    lf = pl.scan_parquet(path).filter(pl.col("event_date") < cutoff)
    if start is not None:
        lf = lf.filter(pl.col("event_date") >= start)

    aggs: List[pl.Expr] = [pl.len().alias("n_active_days")]
    aggs += [pl.col(c).sum().alias(f"{c}_sum") for c in _SUM_COLS]
    aggs += [
        pl.col("gmv").max().alias("gmv_max"),
        pl.col("gmv").mean().alias("gmv_mean_day"),
        (pl.col("gmv") > 0).sum().alias("days_gmv_pos"),
        (pl.col("to_ord") > 0).sum().alias("days_ord_pos"),
        (pl.col("to_cart") > 0).sum().alias("days_cart_pos"),
        pl.col("event_date").max().alias("last_active"),
        pl.col("event_date").min().alias("first_active"),
    ]
    # Свежие агрегаты по нескольким окнам.
    for w in _WINDOWS:
        aggs += [_window_sum(c, cutoff, w) for c in ("gmv", "to_ord", "searches")]

    feats = lf.group_by("user_id").agg(aggs)

    feats = feats.with_columns(
        (pl.lit(cutoff) - pl.col("last_active")).dt.total_days().alias("recency_days"),
        (pl.lit(cutoff) - pl.col("first_active")).dt.total_days().alias("tenure_days"),
        (pl.col("gmv_sum") / pl.col("n_active_days")).alias("gmv_per_active_day"),
        (pl.col("to_ord_sum") / (pl.col("searches_sum") + 1)).alias("ord_per_search"),
        (pl.col("to_cart_sum") / (pl.col("searches_sum") + 1)).alias("cart_per_search"),
        (pl.col("days_gmv_pos") / pl.col("n_active_days")).alias("buy_day_rate"),
    )
    # Тренд: свежие 30 дней против предыдущих 30.
    feats = feats.with_columns(
        (pl.col("gmv_30d") - (pl.col("gmv_60d") - pl.col("gmv_30d"))).alias("gmv_trend_30"),
    )
    return feats.drop(["last_active", "first_active"]).collect()


def build_target(path: str, cutoff: dt.date, horizon: int = 30) -> pl.DataFrame:
    """Целевая переменная: сумма gmv в [cutoff, cutoff + horizon)."""
    end = cutoff + dt.timedelta(days=horizon)
    return (
        pl.scan_parquet(path)
        .filter((pl.col("event_date") >= cutoff) & (pl.col("event_date") < end))
        .group_by("user_id")
        .agg(pl.col("gmv").sum().alias("target"))
        .collect()
    )


def make_supervised(
    path: str, cutoff: dt.date, horizon: int = 30, with_target: bool = True
) -> pl.DataFrame:
    """Выборка: все пользователи, признаки до cutoff, (опц.) таргет после.

    Пользователи без активности в окне признаков получают нули — это дешевле,
    чем дозаполнять разреженные ряды по всем дням.
    """
    users = all_user_ids(path)
    feats = build_features(path, cutoff)
    df = users.join(feats, on="user_id", how="left")
    # Неактивные в окне пользователи: счётчики → 0, но «давность» и «стаж»
    # должны быть большими, а не нулём (0 означал бы «активен сегодня»).
    big = 10_000
    df = df.with_columns(
        pl.col("recency_days").fill_null(big),
        pl.col("tenure_days").fill_null(0),
    ).fill_null(0)
    if with_target:
        tgt = build_target(path, cutoff, horizon)
        df = df.join(tgt, on="user_id", how="left").with_columns(
            pl.col("target").fill_null(0.0)
        )
    return df
