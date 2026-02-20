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

    error_report = {}


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

    # drop rows where nans are present
    row_count = o_df.shape[0]
    o_df = o_df.dropna()
    row_diff = row_count-o_df.shape[0]
    if(row_diff>0):
        rmvd_cols = ', '.join(col_diff)
        report = {"na_rows_dropped": row_diff}
        if(not supress_warnings):
            if warn_supported_version:
                warnings.warn("Some rows were dropped because at least one column contained 'na' values which guidepost does not support.", skip_file_prefixes=_warn_skips)
            else:
                print("Warning: Some rows were dropped because at least one column contained 'na' values which guidepost does not support.")
        original_cols = o_df.columns

    #drop columns which are timedelta type
    o_df = o_df.select_dtypes(exclude=['timedelta64[ns]'])
    col_diff = original_cols.difference(o_df.columns)
    if(len(col_diff)>0):
        rmvd_cols = ', '.join(col_diff)
        report = {"timedelta_columns": col_diff}

        if(not supress_warnings):
            if warn_supported_version:
                warnings.warn("The following columns were dropped because they contained 'timedelta' values which guidepost does not support:[{}]. Consider converting these to an interger representation.".format(rmvd_cols), skip_file_prefixes=_warn_skips)
            else:
                print("Warning: The following columns were dropped because they contained 'timedelta' values which guidepost does not support:[{}]. Consider converting these to an interger representation.".format(rmvd_cols))
        original_cols = o_df.columns

    #drop arrays/complex datatypes
    col_diff = []
    for col in o_df.columns:
        if(type(o_df[col].iloc[0]) == type(np.ndarray([]))):
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
                    "mean": None if ser.count() == 0 else float(ser.mean()),
                    "std": None if ser.count() == 0 else float(ser.std()),
                    "min": None if ser.count() == 0 else float(ser.min()),
                    "25%": None if ser.count() == 0 else float(ser.quantile(0.25)),
                    "50%": None if ser.count() == 0 else float(ser.quantile(0.5)),
                    "75%": None if ser.count() == 0 else float(ser.quantile(0.75)),
                    "IQR": None if ser.count() == 0 else float(ser.quantile(0.75) - ser.quantile(0.25)),
                    "max": None if ser.count() == 0 else float(ser.max()),
                    "var": None if ser.count() == 0 else float(ser.var())
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
