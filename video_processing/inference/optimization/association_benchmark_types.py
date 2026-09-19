from enum import Enum


class BenchmarkTracker(str, Enum):
    OC_SORT = 'boxmot_oc_sort'
    BOT_SORT = 'boxmot_bot_sort_gmc_no_reid'
    PRODUCTION = 'production_preprocessor_ilp'
