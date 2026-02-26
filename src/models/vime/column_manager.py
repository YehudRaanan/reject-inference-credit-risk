import logging
import config

logger = logging.getLogger(__name__)

def identify_feature_groups(columns, high_missing_rate_threshold=0.5, df_for_stats=None, critical_features=None):
    """
    Identifies Critical, High Missing, and Regular feature groups.

    Args:
        columns (list): List of all column names.
        high_missing_rate_threshold (float): Threshold for Level 2 (High Missing).
        df_for_stats (pd.DataFrame, optional): DF to calculate actual missing rates.
        critical_features (list, optional): Override for critical features. Defaults to config.

    Returns:
        dict: {
            'critical': [col_names],
            'high_missing': [col_names],
            'regular': [col_names]
        }
    """
    groups = {
        'critical': [],
        'high_missing': [],
        'regular': []
    }

    # 1. Critical Columns (Level 1 - p_critical)
    # Configurable via config.VIME_CRITICAL_FEATURES
    if critical_features is None:
        critical_features = config.VIME_CRITICAL_FEATURES

    for col in columns:
        if col in critical_features:
            groups['critical'].append(col)
            
    # 2. High Missing Columns (Level 2 - p_high_missing)
    # Strategy: 
    # A. Explicit missing indicators (missingindicator_*)
    # B. Features with corresponding indicators having high mean (if df provided)
    
    indicator_cols = [c for c in columns if c.startswith('missingindicator_')]
    high_missing_candidates = set()
    
    if df_for_stats is not None:
        # data-driven detection
        if indicator_cols:
            means = df_for_stats[indicator_cols].mean()
            high_miss = means[means > high_missing_rate_threshold].index.tolist()
            for ind in high_miss:
                high_missing_candidates.add(ind) # The indicator itself
                base = ind.replace('missingindicator_', '')
                if base in columns:
                    high_missing_candidates.add(base) # The base feature
    else:
        # Fallback: Treat ALL missing indicators as "high missing" (safe default)
        # or just leave empty if we strictly need data.
        # Let's assume indicators themselves are Level 2 by default if we lack stats.
        high_missing_candidates.update(indicator_cols)
        
    groups['high_missing'] = list(high_missing_candidates)
    
    # 3. Regular (Level 3 - p_regular)
    # Everything else
    assigned = set(groups['critical']) | set(groups['high_missing'])
    groups['regular'] = [c for c in columns if c not in assigned]
    
    return groups
