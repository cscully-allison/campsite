import pandas as pd
import numpy as np
import warnings
import os
import sys
from pandas.api import types as ptypes

def convert_to_float(value):
    if pd.isna(value):
        return np.nan
    elif value.endswith('K'):
        return float(value[:-1]) * 1e3
    if value.endswith('M'):
        return float(value[:-1]) * 1e6
    elif value.endswith('B'):
        return float(value[:-1]) * 1e9
    return float(value)


def validate_and_clean_dataframe(in_cpy, supress_warnings=False):
    _warn_skips = (os.path.dirname('.'),)
    warn_supported_version = False

    original_cols = in_cpy.columns
    o_df = in_cpy.dropna(axis=1, how='all')

    report = {}


    #remove columns with only nans
    col_diff = original_cols.difference(o_df.columns)
    if(len(col_diff)>0):
        rmvd_cols = ', '.join(col_diff)
        report = {"na_columns": col_diff}
        if(not supress_warnings):
            if warn_supported_version:
                warnings.warn("The following columns were dropped because they contained entirely 'na' values which guidepost does not support:[{}]".format(rmvd_cols), skip_file_prefixes=_warn_skips)
            else:
                print("Warning: The following columns were dropped because they contained entirely 'na' values which guidepost does not support:[{}]".format(rmvd_cols))
        original_cols = o_df.columns

    # Report NaN presence but do NOT drop rows
    na_counts = o_df.isna().sum()
    cols_with_na = na_counts[na_counts > 0]
    if len(cols_with_na) > 0:
        report["na_column_counts"] = {col: int(count) for col, count in cols_with_na.items()}
        if not supress_warnings:
            na_summary = ', '.join(f"{col}({count})" for col, count in cols_with_na.items())
            if warn_supported_version:
                warnings.warn(
                    f"The following columns contain missing values (count per column): [{na_summary}].",
                    skip_file_prefixes=_warn_skips,
                )
            else:
                print(f"Note: The following columns contain missing values (count per column): [{na_summary}].")

    # Convert timedelta/duration columns to total seconds (float64)
    td_cols = [
        col for col in o_df.columns
        if pd.api.types.is_timedelta64_dtype(o_df[col])
        or str(o_df[col].dtype).startswith("duration")
    ]
    if td_cols:
        for col in td_cols:
            o_df[col] = o_df[col].dt.total_seconds().astype("float64")
        report["timedelta_converted"] = td_cols
        converted_cols = ', '.join(td_cols)
        if not supress_warnings:
            if warn_supported_version:
                warnings.warn(
                    f"The following timedelta/duration columns were converted to total seconds (float): [{converted_cols}].",
                    skip_file_prefixes=_warn_skips,
                )
            else:
                print(f"Note: The following timedelta/duration columns were converted to total seconds (float): [{converted_cols}].")

    #drop arrays/complex datatypes
    col_diff = []
    for col in o_df.columns:
        dtype_str = str(o_df[col].dtype)
        is_arrow_list = "list<" in dtype_str
        is_numpy_array = len(o_df) > 0 and type(o_df[col].iloc[0]) == type(np.ndarray([]))
        if is_arrow_list or is_numpy_array:
            col_diff.append(col)
            o_df = o_df.drop(col, axis=1)

    if(len(col_diff)>0):
        rmvd_cols = ', '.join(col_diff)
        report = {"array_columns": col_diff}

        if(not supress_warnings):
            if warn_supported_version:
                warnings.warn("The following columns were dropped because they contained array values in cells which guidepost does not support:[{}]".format(rmvd_cols), skip_file_prefixes=_warn_skips)
            else:
                print("Warning: The following columns were dropped because they contained array values in cells which guidepost does not support:[{}]".format(rmvd_cols))
        original_cols = o_df.columns


    #add synthetic index
    if(o_df.shape[0]>250_000):
        if(not supress_warnings):
            if warn_supported_version:
                warnings.warn("Your dataframe is very large. You may experience performance issues. Consider subsampling or reducing the data down to below 200,000 rows to enhance performance.", skip_file_prefixes=_warn_skips)
            else:
                print("Warning: Your dataframe is very large. You may experience performance issues. Consider subsampling or reducing the data down to below 200,000 rows to enhance performance.")

    return o_df, report

def _safe_float(val):
    """Convert a value to float, returning None for NA/NaT/NaN/Inf types."""
    if pd.isna(val):
        return None
    result = float(val)
    if not np.isfinite(result):
        return None
    return result

def extract_summary_statistics(o_df):
        summary = {}
        type_counts = {"continuous": 0, "ordinal": 0, "categorical": 0}

        for col in o_df.columns:
            s = o_df[col]
            n_rows = len(s)
            n_missing = int(s.isna().sum())
            pct_missing = float(n_missing) / n_rows if n_rows > 0 else 0.0
            n_unique = int(s.nunique(dropna=True))

            # determine semantic type
            if ptypes.is_categorical_dtype(s.dtype):
                semantic = "ordinal" if getattr(s.dtype, "ordered", False) else "categorical"
            elif ptypes.is_bool_dtype(s.dtype):
                semantic = "categorical"
            elif ptypes.is_numeric_dtype(s.dtype):
                # heuristic: small-integer domains likely ordinal (e.g., ratings)
                if ptypes.is_integer_dtype(s.dtype) or n_unique < 20:
                    semantic = "ordinal"
                else:
                    semantic = "continuous"
            else:
                # object, string, datetime, etc.
                # treat datetimes separately as continuous-like
                if ptypes.is_datetime64_any_dtype(s.dtype) or ptypes.is_timedelta64_dtype(s.dtype):
                    semantic = "continuous"
                else:
                    # Check if categorical values are numbers with suffixes M, K, or B
                    if s.dropna().astype(str).str.fullmatch(r'\d+(\.\d+)?[MKB]').all():
                        s = s.map(convert_to_float)
                        semantic = "continuous"
                    else:
                        semantic = "categorical"

            type_counts[semantic] += 1

            col_summary = {
            "dtype": str(s.dtype),
            "semantic_type": semantic,
            "n_rows": n_rows,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "n_unique": n_unique,
            }

            if semantic == "continuous":
                # compute robust numeric summaries (skip NA)
                ser = pd.to_numeric(s, errors="coerce")
                col_summary.update({
                    "count": int(ser.count()),
                    "mean": _safe_float(ser.mean()),
                    "std": _safe_float(ser.std()),
                    "min": _safe_float(ser.min()),
                    "25%": _safe_float(ser.quantile(0.25)),
                    "50%": _safe_float(ser.quantile(0.5)),
                    "75%": _safe_float(ser.quantile(0.75)),
                    "IQR": _safe_float(ser.quantile(0.75) - ser.quantile(0.25)),
                    "max": _safe_float(ser.max()),
                    "var": _safe_float(ser.var())
                })
            else:
                # categorical / ordinal: top categories and frequencies
                vc = s.astype(object).value_counts(dropna=True)
                top = vc.index[0] if len(vc) > 0 else None
                top_freq = int(vc.iloc[0]) if len(vc) > 0 else 0

                # include up to 20 most frequent values
                top_items = []
                for k, v in vc.iloc[:10].items():
                    # convert numpy types to native python types for JSON serialization
                    try:
                        key = k.item() if hasattr(k, "item") else k
                    except Exception:
                        key = str(k)
                    top_items.append({"value": key, "count": int(v)})
                col_summary.update({
                    "top": top,
                    "top_freq": top_freq,
                    "top_values": top_items
                })

            summary[col] = col_summary

        # store results in widget traits for frontend sync
        return summary
