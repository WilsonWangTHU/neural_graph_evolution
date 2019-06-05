# ------------------------------------------------------------------------------
#   @brief:
#       All the parameters of evolutionary agents are listed here
#   @author:
#       Tingwu Wang, 2017, June, 12th
# ------------------------------------------------------------------------------
import numpy as np


def get_evolutionary_config(parser):

    parser.add_argument('--brute_force_search', action='store_true', default=False)

    parser.add_argument('--optimize_creature', action='store_true', default=False)

    parser.add_argument('--new_species_struct', action='store_true', default=False)
    parser.add_argument('--allow_hierarchy', action='store_true', default=True)
    parser.add_argument('--discrete_mutate', action='store_true', default=False)
    parser.add_argument('--force_symmetric', action='store_true', default=False)

    parser.add_argument("--evolutionary_sub_iteration", type=int, default=10)
    parser.add_argument("--num_total_generation", type=int, default=100)
    parser.add_argument("--speciesbank_path", type=str, default=None)
    parser.add_argument("--start_num_species", type=int, default=8)
    parser.add_argument('--more_body_nodes_at_start', action='store_true', default=False)

    parser.add_argument("--reward_count_percentage", type=float, default=0.70,
                        help='the last % used to calculate the average reward' +
                        'for the evolutionary agent')

    # natural selection
    parser.add_argument("--elimination_rate", type=float, default=0.20)
    parser.add_argument("--mutation_add_ratio", type=float, default=0.20)
    parser.add_argument("--mutation_delete_ratio", type=float, default=0.20)
    parser.add_argument('--node_mutate_ratio', type=float, default=0.1)
    parser.add_argument('--self_cross_ratio', type=float, default=0.0)

    # saving options
    parser.add_argument("--save_top_species", type=int, default=5)
    parser.add_argument("--species_bank_save_freq", type=int, default=5)
    parser.add_argument("--visualize_top_species", type=int, default=4)

    parser.add_argument("--species_visualize_freq", type=int, default=1)

    # walker environnment coeff
    parser.add_argument('--walker_ctrl_coeff', type=float, default=0.001)
    parser.add_argument('--walker_force_no_sym', action='store_true',
                        default=False)
    parser.add_argument(
        '--walker_more_constraint',
        action='store_true', default=False,
        help='Force 2 body nodes from torso and 1 nodes from other nodes'
    )
    parser.add_argument('--force_grow_at_ends', action='store_true', default=False)

    # fish environment coeff
    parser.add_argument("--fish_target_angle", type=float, default=np.pi / 6)
    parser.add_argument('--fish_target_speed', type=float, default=0.002,
                        help='the speed of the target')
    # some unique fish environment
    # fish curlswim parameter x = A * sin( omega * y )
    parser.add_argument('--fish_target_var', type=float, default=10,
                        help='the angular velocity it osicallates (omega)')
    parser.add_argument('--fish_target_bound', type=float, default=0.1,
                        help='the amplitute of the osicallation (A)')
    # fish circle swim parameter
    parser.add_argument('--fish_circle_radius', type=float, default=0.5,
                        help='the radius of the trajectory (a circle)')

    # walker settings
    # pruning config
    parser.add_argument("--pruning_batch_size", type=int, default=64)
    parser.add_argument("--maximum_num_species", type=int, default=32)
    parser.add_argument("--maximum_num_species_generated", type=int, default=30)
    parser.add_argument("--start_temp", type=float, default=10)
    parser.add_argument("--end_temp", type=float, default=0.1)
    parser.add_argument("--temp_iteration", type=int, default=30)
    parser.add_argument("--use_pruning", type=int, default=0)
    parser.add_argument("--bayesian_pruning", type=int, default=0)
    parser.add_argument("--softmax_pruning", type=int, default=1)
    parser.add_argument("--gumble_temperature", type=float, default=0.1)

    # optimization config
    parser.add_argument("--master_mind", type=int, default=0)

    parser.add_argument("--controller_reset_freq", type=int, default=0,
                        help='0 means not resetting, otherwise it will'
                             'be the freq')
    parser.add_argument("--reset_progressive_multiplier", type=float, default=1,
                        help='0 means not resetting, otherwise it will'
                             'be the freq')
    parser.add_argument("--max_reset_freq", type=int, default=60)

    parser.add_argument("--reset_elimination_rate", type=int, default=0.9)
    parser.add_argument("--use_disaster_elimination", type=int, default=0)

    parser.add_argument("--tree_net", type=int, default=0)
    parser.add_argument("--fc_amortized_fitness", type=int, default=1)
    parser.add_argument("--fc_pruning", type=int, default=1)

    parser = evolution_viz_config(parser)

    return parser


def evolution_viz_config(parser):
    ''' configs used for drawing generation visualization
    '''
    # the default server is the new server (with a bigger memory)
    # parser.add_argument('--vis_server', type=str, default='http://18.188.157.25')
    parser.add_argument('--vis_server', type=str, default='http://13.58.242.101')
    parser.add_argument('--vis_port', type=int, default=4214)

    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--mute_info', action='store_true')

    # pca coefficient for the
    parser.add_argument('--pca_size_dist', action='store_true')
    parser.add_argument('--pca_pc_dist', action='store_true')
    parser.add_argument('--pca_range_dist', action='store_true')

    # species-level stats
    parser.add_argument('--body_num_line', action='store_true')
    parser.add_argument('--body_size_line', action='store_true')

    # plotting pca
    parser.add_argument('--pca_size_eigenvalue', action='store_true')
    parser.add_argument('--pca_pc_eigenvalue', action='store_true')

    parser.add_argument('--size_v_dist', action='store_true')
    parser.add_argument('--size_v_line', action='store_true')

    return parser
