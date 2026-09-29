import numpy as np
from caa_model.utils.get_yaml_data import ERStruct, RTConf

def generate_RT_Conf(p_grid, t, n_trials):
    """
    Generate synthetic RT and confidence arrays for a single response category.
    """
    if n_trials <= 0:
        return np.array([]), np.array([])
    
    #for miss and CR (confidence levels = 0)
    if p_grid.ndim == 1:
        cdf = np.cumsum(p_grid)
        cdf_norm = cdf / cdf[-1] if cdf[-1] > 0 else cdf
        u = np.random.uniform(0.0, 1.0, size=n_trials)
        sampled_rts = np.interp(u, cdf_norm, t)
        sampled_confs = np.zeros(n_trials, dtype=int)
        return sampled_rts, sampled_confs
    
    # for rem and know
    else:
        nr_conf_levels = len(p_grid)

        mass_per_conf = np.sum(p_grid, axis=-1)
        total_mass = np.sum(mass_per_conf)
        conf_probs = mass_per_conf / total_mass if total_mass > 0 else np.full(nr_conf_levels, 1.0 / nr_conf_levels)

        sampled_confs = np.random.choice(nr_conf_levels, size=n_trials, p=conf_probs)
        sampled_rts = np.zeros(n_trials)

        for i in range(n_trials):
            conf_idx = sampled_confs[i]  
            cdf = np.cumsum(p_grid[conf_idx])          
            cdf_norm = cdf / cdf[-1] if cdf[-1] > 0 else cdf
            u = np.random.uniform(0.0, 1.0)
            sampled_rts[i] = np.interp(u, cdf_norm, t)  
        return sampled_rts, sampled_confs


def simulation(model, params, template_data):
    """
    Generate a synthetic dataset from a model, parameters, and template_data.
    - model: DDM model instance (e.g., DISPClassicDDM)
    - params: Model parameters tuple
    - template_data: Empirical ERStruct dataset for trial counts
    """
    # Separate target words and lure words
    target_params, lure_params = model.split_params(params)
    
    # Compute total target and lure word counts from template data
    n_targets = len(template_data.rem_hit.rt) + len(template_data.know_hit.rt) + len(template_data.miss.rt)
    n_lures = len(template_data.rem_fa.rt) + len(template_data.know_fa.rt) + len(template_data.CR.rt)
    
    # Compute probability for each category 
    p_rem_target, p_know_target, p_miss_target, t = model.predicted_proportions(target_params)
    p_rem_lure, p_know_lure, p_cr_lure, t = model.predicted_proportions(lure_params)

    # Compute proportion of each response category from model predictions
    mass_rem_target = np.sum(p_rem_target)
    mass_know_target = np.sum(p_know_target)
    mass_miss_target = np.sum(p_miss_target)
    target_probs = np.array([mass_rem_target, mass_know_target, mass_miss_target]) / (mass_rem_target + mass_know_target + mass_miss_target)

    mass_rem_lure = np.sum(p_rem_lure)
    mass_know_lure = np.sum(p_know_lure)
    mass_cr_lure = np.sum(p_cr_lure)
    lure_probs = np.array([mass_rem_lure, mass_know_lure, mass_cr_lure]) / (mass_rem_lure + mass_know_lure + mass_cr_lure)

    # Sample trial counts for each response category using multinomial sampling
    n_rem_hit, n_know_hit, n_miss = np.random.multinomial(n_targets, target_probs)
    n_rem_fa, n_know_fa, n_cr = np.random.multinomial(n_lures, lure_probs)

    rem_hit_rt, rem_hit_conf = generate_RT_Conf(p_rem_target, t, n_rem_hit)
    know_hit_rt, know_hit_conf = generate_RT_Conf(p_know_target, t, n_know_hit)
    miss_rt, miss_conf = generate_RT_Conf(p_miss_target, t, n_miss)

    rem_fa_rt, rem_fa_conf = generate_RT_Conf(p_rem_lure, t, n_rem_fa)
    know_fa_rt, know_fa_conf = generate_RT_Conf(p_know_lure, t, n_know_fa)
    cr_rt, cr_conf = generate_RT_Conf(p_cr_lure, t, n_cr)

    # Create RTConf structure
    def make_rt_conf(rts, confs, target_val, template_cat):
        subj_val = template_cat.subj[0] if len(template_cat.subj) > 0 else "synthetic"
        return RTConf(
            rt=rts,
            conf=confs,
            target=np.full(len(rts), target_val),
            subj=np.full(len(rts), subj_val)
        )

    synthetic_data = ERStruct(
        know_hit=make_rt_conf(know_hit_rt, know_hit_conf, 1, template_data.know_hit),
        rem_hit=make_rt_conf(rem_hit_rt, rem_hit_conf, 1, template_data.rem_hit),
        know_fa=make_rt_conf(know_fa_rt, know_fa_conf, 0, template_data.know_fa),
        rem_fa=make_rt_conf(rem_fa_rt, rem_fa_conf, 0, template_data.rem_fa),
        CR=make_rt_conf(cr_rt, cr_conf, 0, template_data.CR),
        miss=make_rt_conf(miss_rt, miss_conf, 1, template_data.miss),
    )

    return synthetic_data


