from scipy.stats import ttest_ind, wilcoxon


def test_ditr_significantly_greater_than_another(data1, data2, test_type, alpha=0.05):

    print(f'----------------------- {test_type.capitalize()} -----------------------')

    if test_type == 'ttest':
        test_func = ttest_ind
    elif test_type == 'wilcoxon':
        test_func = wilcoxon
    else:
        raise ValueError(f'test_type must be ttest or wilcoxon not {test_type}')

    statistic, p_value = test_func(data1, data2, alternative='greater')
    
    if p_value < alpha:
        print("The mean of the first distribution is significantly" +
            " greater than the mean of the second one.")
    else:
        print("The mean of the first distribution is not significantly" +
            " greater than the mean of the second one.")
        
    return statistic, p_value
