'''
    some special requirements for some experiment configuration
'''

import init_path

USE_TIME_AS_X = 2
BF_EVOLUTION = 1
SINGLE_TOPO_FILE = 3

special_requirement = {
    'bump': {
        '1': 0,
        '2': 0,
        'FishSpeed': 0
    },
    'fishspeed': {
        'BruteForce_100': 1,
        'BruteForce_20': 1,
        'BruteForce_200': 1,
        'BruteForce': 1, 
        'MLP_100': 0,
        'MLP_20': 0,
        'MLP_200': 0,
        'MLP': 0,
        'NGE': 0,
        'NTE': 0
    },
    'fishspeed_full': {
        'BruteForce_100': 1,
        'BruteForce_20': 1,
        'BruteForce_200': 1,
        'BruteForce': 1, 
        'MLP_100': 0,
        'MLP_20': 0,
        'MLP_200': 0,
        'MLP': 0,
        'NGE': 0,
        'NTE': 0
    },
    'walkerspeed_rebuttal': {
        'sims-bodyshare1': 0,
        'sims-bodyshare2': 0,
        'Sims_GM-UC': 0,
        'sims-gm-uc-20': 0,
        'sims-gm-uc-100': 0,
        'sims-gm-uc-200': 0,
        'sims-gm-uc': 0,
        'sims-100': 0,
        'sims-20': 0,
        'BruteForce_100': 1,
        'BruteForce_20': 1,
        'BruteForce_200': 1,
        'BruteForce': 1, 
        'MLP_100': 0,
        'MLP_20': 0,
        'MLP_200': 0,
        'MLP': 0,
        'NGE': 0,
        'NTE': 0
    },
    'fishspeed_rebuttal': {
        'Sims_bodyshare1': 0,
        'Sims_bodyshare2': 0,
        'Sims_GM-UC': 0,
        'sims-gm-uc-20': 0,
        'sims-gm-uc-100': 0,
        'sims-gm-uc-200': 0,
        'Sims_100': 0,
        'Sims_20': 0,
        'BruteForce_100': 1,
        'BruteForce_20': 1,
        'BruteForce_200': 1,
        'BruteForce': 1, 
        'MLP_100': 0,
        'MLP_20': 0,
        'MLP_200': 0,
        'MLP': 0,
        'NGE': 0,
        'NTE': 0
    },
    'nervenet_plus-nervenet': {
        'baseline_noninput': 2,
        'baseline_para': 2,
        'nervenetplus_noninput': 2,
        'nervenetplus_para': 2
    },
    'nnpp_cheetah': {
        'NerveNet': 3,
        'NerveNet++': 3
    },
    'nnpp_walker': {
        'NerveNet': 3,
        'NerveNet++': 3
    },
    'pruning': {
        '16_baseline': 0,
        '16_pruning': 0,
        '32_baseline': 0,
        '32_pruning': 0
    },
    'resetting': {
        'baseline': 0,
        'Reset': 0,
        'Baseline': 0,
        'continuous_reset_100': 0,
        'continuous_reset_5': 0
    },
    'finetuning': {
        'cheetah': 0,
        'fish': 0,
        'hopper': 0,
        'walker': 0
    },
    'walker-speed': {
        'BruteForce_100': 1,
        'BruteForce_20': 1,
        'BruteForce_200': 1,
        'BruteForce': 1,
        'MLP': 0, 
        'MLP_100': 0,
        'MLP_200': 0,
        'MLP_20': 0,
        'NGE': 0,
        'NTE': 0
    },
    'walker-speed_full': {
        'BruteForce_100': 1,
        'BruteForce_20': 1,
        'BruteForce_200': 1,
        'BruteForce': 1,
        'MLP': 0, 
        'MLP_100': 0,
        'MLP_200': 0,
        'MLP_20': 0,
        'NGE': 0,
        'NTE': 0
    },
    'finetune-fish': {
        'Fish': 0,
        'FixedTopo': 3,
        'Fixed': 3,
        'NGE': 0,
        'NGE-Unconstrained': 0,
        'NGE-FixedTopo': 0,
        'NGE-ConstrainTopo': 0,
        'NTE': 0
    },
    'finetune-walker': {
        'Walker': 0,
        'FixedTopo': 3,
        'Fixed': 3,
        'NGE': 0,
        'NGE-Unconstrained': 0,
        'NGE-FixedTopo': 0,
        'NGE-ConstrainTopo': 0,
        'NTE': 0
    },
    'finetune-hopper': {
        'Hopper': 0,
        'FixedTopo': 3,
        'Fixed': 3,
        'NGE': 0,
        'NTE': 0
    },
    'finetune-cheetah': {
        'Cheetah': 0,
        'FixedTopo': 3,
        'Fixed': 3,
        'NGE': 0,
        'NGE-Unconstrained': 0,
        'NGE-FixedTopo': 0,
        'NGE-ConstrainTopo': 0,
        'NTE': 0
    },
    'prune16': {
        'fish_16_01pruning': 0,
        'walker_16_01pruning': 0,
        'fish_16_baseline': 0,
        'walker_16_baseline': 0
    },
    'prune16_fish': {
        'fish_16_01pruning': 0,
        'fish_16_baseline': 0
    },
    'prune64_fish': {
        'FishBaseline': 0,
        'walker_baseline_64_2': 0,
        'walker_baseline_64_1': 0,
        'FishWithPruning': 0,
        'NTE+Pruning': 0,
        'NTEWithoutPruning': 0,
        'NGE+Pruning': 0,
        'NGEWithoutPruning': 0,
        'NGE': 0,
        'NTE': 0,
        'walker_pruning': 0
    },
    'resource_fish': {
        '16-Core': 0,
        '64-Core': 0
    },
    'resource_walker': {
        '16-Core': 0,
        '64-Core': 0
    },
    'prune64_walker': {
        'walker_baseline_64_2': 0,
        'walker_baseline_64_1': 0,
        'WalkerBaseline_1': 0,
        'WalkerBaseline_2': 0,
        'WalkerBaseline': 0,
        'NTE+Pruning': 0,
        'NGE+Pruning': 0,
        'NGE': 0,
        'NTE': 0,
        'NTEWithoutPruning': 0,
        'NGEWithoutPruning': 0,
        'WalkerWithPruning': 0
    }

}

remote_data_dir = {
    'bump': {
        '1': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_0509_night/\
topology_rl/evolution_data/evofish3d-speed_64_continuous_baseline''',
        '2': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl/\
evolution_data/evofish3d-speed_2018_05_08-04:17:08'''
    },
    'fishspeed': {
        'BruteForce': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search_100''',
        'BruteForce_100': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search_100-continuous''',
        'BruteForce_20': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search-20''',
        'BruteForce_200': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search_discrete_200''',
        'fc_100': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines\
/topology_rl/evolution_data/evofish3d-speed_051117fc_100-continuous''',
        'fc_20': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117fc_20_continuous''',
        'fc_200': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/\
topology_rl/evolution_data/evofish3d-speed_051117fc_discrete_200''',
        'ourmodel': None
    },
    'fishspeed_full': {
        'BruteForce': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search_100''',
        'BruteForce_100': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search_100-continuous''',
        'BruteForce_20': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search-20''',
        'BruteForce_200': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search_discrete_200''',
        'fc_100': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines\
/topology_rl/evolution_data/evofish3d-speed_051117fc_100-continuous''',
        'fc_20': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117fc_20_continuous''',
        'fc_200': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/\
topology_rl/evolution_data/evofish3d-speed_051117fc_discrete_200''',
        'ourmodel': None
    },
    'fishspeed_rebuttal': {
        'BruteForce': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search_100''',
        'BruteForce_100': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search_100-continuous''',
        'BruteForce_20': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search-20''',
        'BruteForce_200': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117brute_search_discrete_200''',
        'fc_100': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines\
/topology_rl/evolution_data/evofish3d-speed_051117fc_100-continuous''',
        'fc_20': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evofish3d-speed_051117fc_20_continuous''',
        'fc_200': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/\
topology_rl/evolution_data/evofish3d-speed_051117fc_discrete_200''',
        'ourmodel': None
    },
    'walker-speed': {
        'BruteForce_100': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search_100''',
        'BruteForce_20': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search-20''',
        'BruteForce_200': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search_discrete_200''',
        'BruteForce': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search_discrete_200'''
    },
    'walker-speed_full': {
        'BruteForce_100': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search_100''',
        'BruteForce_20': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search-20''',
        'BruteForce_200': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search_discrete_200''',
        'BruteForce': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search_discrete_200'''
    },
    'walkerspeed_rebuttal': {
        'BruteForce_100': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search_100''',
        'BruteForce_20': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search-20''',
        'BruteForce_200': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search_discrete_200''',
        'BruteForce': '''henryzhou@cluster12:\
/ais/gobi6/tingwuwang/topology_rl_baselines/topology_rl/\
evolution_data/evowalker-speed_walker_baselinesbrute_search_discrete_200'''
    },
    'nervenet_plus-nervenet': {
        'baseline_noninput': 2,
        'baseline_para': 2,
        'nervenetplus_noninput': 2,
        'nervenetplus_para': 2
    },
    'pruning': {
        '16_baseline': None,
        '16_pruning': None,
        '32_baseline': None,
        '32_pruning': None
    },
    'resetting': {
        'baseline': None,
        'continuous_reset_100': None,
        'continuous_reset_5': None
    }
}

remote_log_dir = {
    'bump': {
        '1': None,
        '2': None
    },
    'fishspeed': {
        'bf_100': None,
        'bf_20': None,
        'bf_200': None,
        'fc_100': None,
        'fc_20': None,
        'fc_200': None,
        'ourmodel': None
    },
    'nervenet_plus-nervenet': {
        'baseline_noninput': '''henryzhou@cluster12:/ais/gobi6/tingwuwang/\
topology_ablation/topology_rl/log/\
mujoco_evofish3d-speedablation_1132_noninput_baseline.log''',
        'baseline_para': '''henryzhou@cluster12:/ais/gobi6/tingwuwang/\
topology_ablation/topology_rl/log/\
mujoco_evofish3d-speedablation_1132_para_baseline.log''',
        'nervenetplus_noninput': '''henryzhou@cluster12:/ais/gobi6/tingwuwang/\
topology_ablation/topology_rl/log/\
mujoco_evofish3d-speedablation_1132_noninput_nervenet.log''',
        'nervenetplus_para': '''henryzhou@cluster12:/ais/gobi6/tingwuwang/\
topology_ablation/topology_rl/log/\
mujoco_evofish3d-speedablation_1132_para_nervenet.log'''
    },
    'pruning': {
        '16_baseline': None,
        '16_pruning': None,
        '32_baseline': None,
        '32_pruning': None
    },
    'resetting': {
        'baseline': None,
        'continuous_reset_100': None,
        'continuous_reset_5': None
    }
}

if __name__ == '__main__':
    from visual import visualize_species_struct
    import subprocess

    # sync with the remote dir
    for task, experiment_dict in remote_data_dir.items():
        for experiment_name, remote_dir in experiment_dict.items():
            if remote_dir is None: continue
            print(task, experiment_name)
            local_dir = visualize_species_struct.sync_dir(
                remote_dir,
                synced_flag=False
            )

    # generate the images
    