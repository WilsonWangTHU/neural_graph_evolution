''' plot learning curves for evolution data
'''

# setups
DATA_DIR = '/Users/Jarvis/Desktop/evolutionary_data'
# DATA_DIR = '/Users/Jarvis/Desktop/evo_data/evolutionary_data'
DATA_TEST_CSV_FILE1 = \
    '/Users/Jarvis/Desktop/evolutionary_data/fishspeed/FC_100.csv'
DATA_TEST_CSV_FILE2 = \
    '/Users/Jarvis/Desktop/evolutionary_data/fishspeed/FC_20.csv'
import init_path

# system imports
import os
import glob
from copy import deepcopy
import argparse

# compute imports
import numpy as np

# local ipmorts
from visual import visual_util
from visual import visual_info


def task_handle(task_path, expfilter=None):
    ''' handle a task DATA_DIR / <task>
    '''
    task_name = task_path.split('/')[-1]
    # if 'finetune' not in task_name: return None
    # if 'nervenet' not in task_name: return None
    if expfilter == None: pass
    else:
        if expfilter not in task_name: return None 
    if 'pruning' in task_name: return None
    if 'hopper' in task_name: return None
    if 'nervenet' in task_name: return None

    # iterate through the result file
    task_data = {}
    for experiment_file in glob.glob(task_path + '/*'):
        if experiment_file.endswith('.pdf'): continue
        if 'ignore' in experiment_file: continue
        if os.path.isdir(experiment_file): continue

        experiment_name = experiment_file.split('/')[-1].split('.')[0]
        experiment_name = experiment_name.replace('NTE', 'NGE')
        print('\t', experiment_name)
        
        if experiment_file.endswith('.csv'):
            data = visual_util.read_csv(experiment_file)
        elif experiment_file.endswith('.npy'):
            data = visual_util.read_topo(experiment_file)
        else:
            raise RuntimeError('Found a file type that is not supported')

        experiment_data = {}

        experiment_data['generation'] = [x[0] for x in data]


        if visual_info.special_requirement[task_name][experiment_name] == \
               visual_info.BF_EVOLUTION:
            print('Start to retrieve brute force evolution process')
            experiment_data2 = \
                visual_util.examine_brute_force(task_name, experiment_name)
            experiment_data = {**experiment_data, **experiment_data2}
        elif visual_info.special_requirement[task_name][experiment_name] == \
                visual_info.USE_TIME_AS_X:
            print('Retrieving time information to plot time as x')
            experiment_data['time'] = \
                visual_util.examine_log_time(task_name, experiment_name)

            experiment_data['x1'] = deepcopy(experiment_data['time'][1:])
            experiment_data['x2'] = deepcopy(experiment_data['time'][1:])
            experiment_data['x3'] = deepcopy(experiment_data['time'][1:])

            experiment_data['avg_r'] = [x[1] for x in data]
            experiment_data['max_r'] = [x[2] for x in data]
            experiment_data['min_r'] = [x[3] for x in data]
        elif visual_info.special_requirement[task_name][experiment_name] == \
                 visual_info.SINGLE_TOPO_FILE:
            experiment_data = data
        else:
            experiment_data['x1'] = [x[0] for x in data]
            experiment_data['x2'] = [x[0] for x in data]
            experiment_data['x3'] = [x[0] for x in data]
            experiment_data['avg_r'] = [x[1] for x in data]
            experiment_data['max_r'] = [x[2] for x in data]
            experiment_data['min_r'] = [x[3] for x in data]

            if 'prune' in task_name:
                experiment_data['x1'] = [x[0] for x in data][25:200]
                experiment_data['x2'] = [x[0] for x in data][25:200]
                experiment_data['x3'] = [x[0] for x in data][25:200]
                experiment_data['avg_r'] = [x[1] for x in data][25:200]
                experiment_data['max_r'] = [x[2] for x in data][25:200]
                experiment_data['min_r'] = [x[3] for x in data][25:200]



        length = len(experiment_data['generation'])
        if length > 200 and 'reset' not in task_name:
            try: experiment_data['generation'] = experiment_data['generation'][:200]
            except: pass
            try: experiment_data['reward'] = experiment_data['reward'][:200]
            except: pass
            try: experiment_data['x1'] = experiment_data['x1'][:200] 
            except: pass
            try: experiment_data['x2'] = experiment_data['x2'][:200] 
            except: pass
            try: experiment_data['x3'] = experiment_data['x3'][:200] 
            except: pass
            try: experiment_data['avg_r'] = experiment_data['avg_r'][:200] 
            except: pass
            try: experiment_data['max_r'] = experiment_data['max_r'][:200] 
            except: pass
            try: experiment_data['min_r'] = experiment_data['min_r'][:200] 
            except: pass

        task_data[ experiment_name ] = experiment_data

    # work on the existing data: interpolate according to the range
    longest_generation = max([len(v['generation']) for k, v in task_data.items()])

    for name_t, data_t in task_data.items():
        
        print(task_name, name_t)
        if 'nnpp' in task_name and '++' in name_t:
            data_points = len(data_t['x1'])
            data_t['x1'] = data_t['x1'][10:]
            data_t['reward'] = data_t['reward'][10:]
            data_t['x1'] = np.array(data_t['x1']) / data_points * 34
            continue
        elif 'nnpp' in task_name and '++' not in name_t:
            data_points = len(data_t['x1'])
            data_t['x1'] = data_t['x1'][10:int(5 / 7 * data_points)]
            data_t['reward'] = data_t['reward'][10:int(5 / 7 * data_points)]
            data_t['x1'] = np.array(data_t['x1']) / data_points * 70
            continue

        if visual_info.special_requirement[task_name][name_t] == \
               visual_info.USE_TIME_AS_X: continue
        
        ratio = longest_generation / len(data_t['generation']) + 1e-6
        if 'x1' in data_t: data_t['x1'] = np.array(data_t['x1']) * ratio
        if 'x2' in data_t: data_t['x2'] = np.array(data_t['x2']) * ratio
        if 'x3' in data_t: data_t['x3'] = np.array(data_t['x3']) * ratio

    # format the data for plotting
    lines_x = []
    lines_y = []
    style = []
    color = []
    legends = []

    # get our model to the first
    task_list = list( task_data.keys() )
    
    if 'fishspeed' == task_name or 'walker-speed' == task_name:
        task_list.insert(0, task_list.pop(task_list.index('NGE')))
    if 'finetune' in task_name:
        task_list.insert(2, task_list.pop(task_list.index('Fixed')))
        task_list.insert(0, task_list.pop(task_list.index('NGE-Unconstrained')))
    if 'prune64_fish' in task_name:
        task_list.insert(0, task_list.pop(task_list.index('NGE+Pruning')))
        # task_list.insert(-1, task_list.pop(task_list.index('NGE')))
    if 'prune64_walker' in task_name:
        task_list.insert(0, task_list.pop(task_list.index('NGE+Pruning')))
        # task_list.insert(-1, task_list.pop(task_list.index('NGE')))
    if 'reset' in task_name:
        task_list.insert(0, task_list.pop(task_list.index('Reset')))


    for i, name_exp in enumerate(task_list):
        data_exp = task_data[name_exp]
        
        if 'Fixed' in name_exp or 'nnpp' in task_name:
            lines_x += [data_exp['x1']]
            lines_y += [data_exp['reward']]
            style += ['-']
            color += [visual_util.con_color[i]] * 1
            legends += [('%s' % name_exp)]
        elif 'bump' in name_exp:
            lines_x += [data_exp['x2'], data_exp['x1']]
            lines_y += [data_exp['max_r'], data_exp['avg_r']]
            style += ['--', '-']
            color += [visual_util.con_color[i], visual_util.con_color[i+1]]
            legends += ['Max', 'Avg']
        else:
            lines_x += [data_exp['x2']]
            lines_y += [data_exp['max_r']]
            style += ['-']
            color += [visual_util.con_color[i % len(visual_util.con_color)]]
            
            if 'prune' in task_name and name_exp == 'NGE':
                legends += ['%s+Greedy' % name_exp]
            else:
                legends += [('%s' % name_exp)]
    


    # for extra smoothing
    if 'walker' in task_name:

        for i in range( len(lines_x) ):
            lines_x[i], lines_y[i] = visual_util.smooth_curve(lines_x[i], lines_y[i], 5)


    # actual plotting
    file_path = os.path.join(task_path, '%s-learningcurve.pdf' % (task_name))

    if 'nervenet' in task_name:
        xlabel = 'time in hours'
    elif 'nnpp' in task_name:
        xlabel = 'wall-clock in minutes'
    elif 'finetune' in task_name:
        xlabel = 'Updates'
    else:
        xlabel = 'evolution generation'

    if 'resource' in task_name or 'prune' in task_name:
        fig_size=(8, 3)
    else:
        fig_size=(8, 6)

    lines_x = [np.array(x) for x in lines_x]
    lines_y = [np.array(y) for y in lines_y]


    

    if 'full' in file_path:
        
        
        legends = ['NGE', 'ES', 'RGS']
        color = visual_util.con_color[:len(legends)]
        style = ['-'] * len(legends)

        lines_x = [np.array(x) for x in lines_x]
        lines_y = [np.array(y) for y in lines_y]
        # visual_util.plot_curves_shadedV2(lines_x, lines_y, xlabel.capitalize(), 'Reward',
        #         '', legends=legends, filename=file_path.replace('.pdf', '_all_shaded_V2.pdf'),
        #         line_color=color, line_style=style)

        visual_util.plot_curves_shaded_seaborn(lines_x, lines_y, xlabel.capitalize(), 'Reward',
                '', legends=legends, filename=file_path.replace('.pdf', '_all_shaded_seaborn.pdf'),
                line_color=color, line_style=style)

    elif 'rebuttal' in file_path:


        legends = ['NGE', 'ESS', 'RGS', 'Sims', 'Sims-GM-UC', 'Sims-BE']
        color = visual_util.con_color[:len(legends)]
        style = ['-'] * len(legends)

        lines_x = [np.array(x) for x in lines_x]
        lines_y = [np.array(y) for y in lines_y]

        if 'fish' in file_path:
            
            visual_util.plot_curves_shaded_seaborn_fish_rebuttal(
                    lines_x, lines_y, xlabel.capitalize(), 'Reward',
                    '', legends=legends, 
                    filename=file_path.replace('.pdf', '_all_shaded_rebuttal'),
                    line_color=color, line_style=style)

        elif 'walker' in file_path:
            
            visual_util.plot_curves_shaded_seaborn_walker_rebuttal(
                    lines_x, lines_y, xlabel.capitalize(), 'Reward',
                    '', legends=legends, 
                    filename=file_path.replace('.pdf', '_all_shaded_rebuttal'),
                    line_color=color, line_style=style)
        return None

    # if 'finetune' in task_name:
    #     lines_x = [np.array(x) * 20 for x in lines_x]

    visual_util.plot_curves(lines_x, lines_y, xlabel.capitalize(), 'Reward',
        '', legends=legends, filename=file_path,
        line_color=color, line_style=style,
        figsize=fig_size
    )

    visual_util.plot_curves_shaded(lines_x, lines_y, xlabel.capitalize(), 'Reward',
            '', legends=legends, filename=file_path.replace('.pdf', '_shaded.pdf'),
            line_color=color, line_style=style)
            

    return None


def get_config():
    parser = argparse.ArgumentParser(description='Generate data visualization')
    parser.add_argument('--filter', type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_config()
    
    for task_path in glob.glob(DATA_DIR + '/*'):
        task_handle(task_path, expfilter=args.filter)

    

