"""
An adaptation of metrics.py 
https://github.com/sebp/scikit-survival/blob/v0.20.0/sksurv/metrics.py
"""

import numpy
from scipy.integrate import trapz
from sklearn.utils import check_consistent_length, check_array

from sksurv.exceptions import NoComparablePairException
from sksurv.nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator
from sksurv.util import check_y_survival

__all__ = [
    'brier_score',
    'concordance_index_censored',
    'concordance_index_ipcw',
    'cumulative_dynamic_auc',
    'integrated_brier_score',
]

def _check_estimate(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim != 1:
        raise ValueError(
            'Expected 1D array, got {:d}D array instead:\narray={}.\n'.format(
                estimate.ndim, estimate))
    check_consistent_length(test_time, estimate)
    return estimate


def _check_inputs(event_indicator, event_time, estimate):
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = _check_estimate(estimate, event_time)

    if not numpy.issubdtype(event_indicator.dtype, numpy.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels for survival analysis, got {0}'.format(
                event_indicator.dtype))

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _check_times(test_time, times):
    times = check_array(numpy.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    times = numpy.unique(times)

    if times.max() >= test_time.max() or times.min() < test_time.min():
        raise ValueError(
            'all times must be within follow-up time of test data: [{}; {}['.format(
                test_time.min(), test_time.max()))

    return times


def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = numpy.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time


def _estimate_concordance_index(event_indicator, event_time, estimate, weights, tied_tol=1e-8):
    order = numpy.argsort(event_time)

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    if len(comparable) == 0:
        raise NoComparablePairException(
            "Data has no comparable pairs, cannot estimate concordance index.")

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]
        est = estimate[order[mask]]
        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = numpy.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-8):
    """
    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    event_indicator, event_time, estimate = _check_inputs(
        event_indicator, event_time, estimate)

    w = numpy.ones_like(estimate)

    return _estimate_concordance_index(event_indicator, event_time, estimate, w, tied_tol)


def concordance_index_ipcw(survival_train, survival_test, estimate, tau=None, tied_tol=1e-8):    
    """
    References
    ----------
    [1] Uno, H., Cai, T., Pencina, M. J., D’Agostino, R. B., & Wei, L. J. (2011).
           "On the C-statistics for evaluating overall adequacy of risk prediction
           procedures with censored survival data".
           Statistics in Medicine, 30(10), 1105–1117.
    """
    test_event, test_time = check_y_survival(survival_test)

    if tau is not None:
        mask = test_time < tau
        survival_test = survival_test[mask]

    estimate = _check_estimate(estimate, test_time)

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw_test = cens.predict_ipcw(survival_test)
    if tau is None:
        ipcw = ipcw_test
    else:
        ipcw = numpy.empty(estimate.shape[0], dtype=ipcw_test.dtype)
        ipcw[mask] = ipcw_test
        ipcw[~mask] = 0

    w = numpy.square(ipcw)

    return _estimate_concordance_index(test_event, test_time, estimate, w, tied_tol)


def cumulative_dynamic_auc(survival_train, survival_test, estimate, times, verbose= False, tied_tol=1e-8, ci=.95, debug=False ):
    """
    References
    ----------
    .. [1] H. Uno, T. Cai, L. Tian, and L. J. Wei,
           "Evaluating prediction rules for t-year survivors with censored regression models,"
           Journal of the American Statistical Association, vol. 102, pp. 527–537, 2007.
    .. [2] H. Hung and C. T. Chiang,
           "Estimation methods for time-dependent AUC models with survival data,"
           Canadian Journal of Statistics, vol. 38, no. 1, pp. 8–26, 2010.
    .. [3] J. Lambert and S. Chevret,
           "Summary measure of discrimination in survival models based on cumulative/dynamic time-dependent ROC curves,"
           Statistical Methods in Medical Research, 2014.
    """
    test_event, test_time = check_y_survival(survival_test)
    estimate = _check_estimate(estimate, test_time)
    times = _check_times(test_time, times)

    # sort by risk score (descending)
    o = numpy.argsort(-estimate)
    test_time = test_time[o]
    test_event = test_event[o]
    estimate = estimate[o]
    survival_test = survival_test[o]

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw = cens.predict_ipcw(survival_test)      
    
    import scipy.stats as st
    
    if ci==.95:
        delta = (1-ci )/2
        upper = st.norm.ppf( .975 )
        lower = st.norm.ppf( .025 )
    else:
        delta = (1-ci)/2
        upper = st.norm.ppf( ci+delta)
        lower = st.norm.ppf( delta )
        
        if debug:
            print(ci, 'c.i.: ', delta , ci+delta )
    
    se_auc = []
    SENS = []
    FPR = []
    n_samples = test_time.shape[0]
    scores = numpy.empty(times.shape[0], dtype=float)
    for k, t in enumerate(times):
        is_case = (test_time <= t) & test_event
        is_control = test_time > t
        n_controls = is_control.sum()
        n_cases = is_case.sum()
        
        if verbose:
            print(n_controls, 'controls', n_cases, 'cases' )
        
        true_pos = []
        false_pos = []
        tp_value = 0.0
        fp_value = 0.0
        est_prev = numpy.infty

        for i in range(n_samples):
            est = estimate[i]
            if numpy.absolute(est - est_prev) > tied_tol:
                true_pos.append(tp_value)
                false_pos.append(fp_value)
                est_prev = est
            if is_case[i]:
                tp_value += ipcw[i]
            elif is_control[i]:
                fp_value += 1
        true_pos.append(tp_value)
        false_pos.append(fp_value)

        sens = numpy.array(true_pos) / ipcw[is_case].sum()
        fpr = numpy.array(false_pos) / n_controls
        auc = scores[k] = trapz(sens, fpr)
        
        
        FPR.append( fpr )
        SENS.append( sens )
        
        
        Q1 = (auc)/(2-auc )
        Q2 = 2*(auc**2)/(1+auc)
        
        n1 = n_cases + 1e-10
        n2 = n_controls + 1e-10       
        if 0:
            n1 = (n1 + n2) /2
            n2 = n1
        se = numpy.sqrt(  ( auc *(1-auc) + (n1 - 1)*(Q1 - auc**2) +  (n2 - 1)*(Q2 - auc**2) )/ n1/ n2 )
        se_auc.append( (auc+lower*se, auc, auc+ upper*se) )
        
    
    if times.shape[0] == 1:
        mean_auc = scores[0]
        if verbose:
            print( 'Single timepoint', scores )                
    else:
        surv = SurvivalFunctionEstimator()
        surv.fit(survival_test)
        s_times = surv.predict_proba(times)
        # compute integral of AUC over survival function
        d = -numpy.diff(numpy.concatenate(([1.0], s_times)))
        integral = (scores * d).sum()
        mean_auc = integral / (1.0 - s_times[-1])
        if verbose:        
            print( 'multiple timepoint', mean_auc )
    
    return scores, mean_auc, se_auc, SENS, FPR, ipcw 


def brier_score(survival_train, survival_test, estimate, times):    
    
    test_event, test_time = check_y_survival(survival_test)
    times = _check_times(test_time, times)

    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)

    if estimate.shape[0] != test_time.shape[0]:
        raise ValueError("expected estimate with {} samples, but got {}".format(
            test_time.shape[0], estimate.shape[0]
        ))

    if estimate.shape[1] != times.shape[0]:
        raise ValueError("expected estimate with {} columns, but got {}".format(
            times.shape[0], estimate.shape[1]))

    # fit IPCW estimator
    cens = CensoringDistributionEstimator().fit(survival_train)
    # calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = numpy.inf
    
    # calculate inverse probability of censoring weights at observed time point
    prob_cens_y = cens.predict_proba(test_time)
    prob_cens_y[prob_cens_y == 0] = numpy.inf

    # Calculating the brier scores at each time point
    brier_scores = numpy.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[:, i]
        is_case = (test_time <= t) & test_event
        is_control = test_time > t

        brier_scores[i] = numpy.mean(numpy.square(est) * is_case.astype(int) / prob_cens_y
                                     + numpy.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i])

    return times, brier_scores


def integrated_brier_score(survival_train, survival_test, estimate, times):     
    """
    [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    # Computing the brier scores
    times, brier_scores = brier_score(survival_train, survival_test, estimate, times)

    if times.shape[0] < 2:
        raise ValueError("At least two time points must be given")

    # Computing the IBS
    ibs_value = trapz(brier_scores, times) / (times[-1] - times[0])

    return ibs_value
