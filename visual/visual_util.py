'''
    utility function for visualizing stats about a training session
'''
import init_path

# system imports
import os
import csv
import glob
import datetime

# compute imports
import numpy as np

# plotting and data processing imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'monospace','monospace':['Courier']})
# plt.rc('font', family='Courier')
from matplotlib import rcParams

# rcParams['font.family'] = 'monospace'
# rcParams['font.monospace'] = 'Courier'
# plt.rcParams["font.family"] = "cursive"
from scipy import interpolate

# local imports
from visual import visual_info
from visual import visualize_species_struct

def interpolate_curve(orig_x, orig_y, new_x):
    ''' 
    '''
    new_y = np.interp(new_x, orig_x, orig_y)
    return new_x, new_y

def read_csv(filepath, skipfirst=True, convert2num=True):
    ''' read a csv file and return its content
    '''
    data = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            data.append( row )

    if skipfirst: start = 1
    else: start = 0
    data = data[start:]

    if convert2num:
        for i, row in enumerate(data):
            data[i] = [float(x) for x in row]

    return data

def read_topo(filepath, window=40):
    ''' supposingly, this npy file should contain a single species
    trained under brute force configuration
    '''
    experiment_data = {}

    data = np.load(filepath).item()
    data = data['brute_reward']
    

    # need to iterate through the reward to clean the data
    for i, reward in enumerate(data):
        if reward < -10:
            data[i] = data[i - 1]

    new_reward = [] 
    for i, reward_i in enumerate(data[:-window]):
        new_reward.append( sum(data[i: i+window]) / window )

    experiment_data['generation'] = np.array( list(range(len(new_reward))) )
    experiment_data['x1'] = np.array( list(range(len(new_reward))) )
    experiment_data['reward'] = np.clip(np.array(new_reward), -1, np.inf)
    

    experiment_data['finetune'] = True
    return experiment_data


light_colors = ['peru', 'dodgerblue', 'springgreen', 'peachpuff', 'chartreuse',
                'lightcoral', 'darkseagreen', 'yellow', 'plum', 'lightgreen',
                'lightpink', 'skyblue', 'lawngreen', 'sienna', 'powderblue',
                'olivedrab', 'sandybrown', 'blueviolet', 'darksalmon', 'orchid']

dark_colors = ['fuchsia', 'midnightblue', 'limegreen', 'red', 'darkviolet', 'darkgreen']

visdom_color = [
    ( 38/255., 120/255., 178/255.),
    (253/255., 127/255.,  40/255.),
    ( 51/255., 158/255.,  52/255.),
    (211/255.,  42/255.,  47/255.),
    (147/255., 106/255., 187/255.),
    (139/255.,  86/255.,  77/255.),
    (225/255., 122/255., 193/255.),
    (127/255., 127/255., 127/255.),
    (187/255., 187/255.,  53/255.)
]

# seaborn dark, starting with dark red
seaborn_dark = [
    (196/255.,  78/255.,  82/255.),
    (100/255., 181/255., 255/255.),
    ( 85/255., 168/255., 104/255.),
    ( 76/255., 114/255., 176/255.),
    (204/255., 185/255., 116/255.),
    (129/255., 114/255., 178/255.)
]

# seaborn bright color, starting with bright red
seaborn_bright = [
    (227/255.,  26/255.,  28/255.),
    (251/255., 154/255., 153/255.),
    ( 51/255., 160/255.,  44/255.),
    (178/255., 223/255., 138/255.),
    ( 31/255., 120/255., 180/255.),
    (166/255., 206/255., 227/255.)
]

# contrastive 
con_color = [
    (194/255.,   5/255.,  37/255.),
    (  1/255.,  30/255., 160/255.),
    ( 88/255., 143/255.,  41/255.),
    (168/255.,  87/255., 207/255.),
    (255/255., 177/255.,  40/255.),
    (255/255.,  86/255.,  39/255.),
    (196/255.,  78/255.,  82/255.),
    (100/255., 181/255., 255/255.),
    ( 85/255., 168/255., 104/255.)
]



def hms_to_seconds(time_format):
    '''
    '''
    h, m, s = [int(i) for i in time_format.split(':')]
    total_seconds = 3600 * h + 60 * m + s 
    return total_seconds

def examine_log_time(task, experiment):
    '''
    '''
    remote_log = visual_info.remote_log_dir[task][experiment]
    local_file = visualize_species_struct.sync_dir(remote_log, synced_flag=False)

    doc = []
    with open(local_file, 'r', encoding='utf8') as f:
        doc = f.readlines()

    # parse the time: map from generation to time elapsed
    gen_time_list = []
    for row in doc:
        if 'Running evolution at generation' not in row: continue
        gen_num = int( row.split(' ')[-1] )

        gen_date = row.split(' ')[0][-4:] # log file seems to have color
        gen_hms = row.split(' ')[1]
        # the training might take place overnight, so need to consider 
        # clock and the date time
        gen_time = '2018-%s-%s %s' % (gen_date[0:2], gen_date[2:], gen_hms)
        gen_time = datetime.datetime.strptime(gen_time, 
                                              "%Y-%m-%d %H:%M:%S")
        gen_time_list.append( (gen_num, gen_time) )


    gen_time_list = sorted(gen_time_list, key=lambda x: x[0])
    start_time = gen_time_list[0][1]
    
    new_gen_time_list = []
    for i in range( len(gen_time_list) - 1 ):

        gen_num, gen_time = gen_time_list[i + 1]
        time_offset = (gen_time - start_time).total_seconds()
        new_gen_time_list.append(time_offset / 3600.0)

    for i in range( len(new_gen_time_list) - 1 ):
        assert new_gen_time_list[i] <= new_gen_time_list[i + 1]


    return [0] + new_gen_time_list

def examine_brute_force(task, experiment):
    '''
    '''
    remote_dir = visual_info.remote_data_dir[task][experiment]
    local_file = visualize_species_struct.sync_dir(remote_dir, synced_flag=False)

    brute_force_reward = []
    for spc_file in glob.glob(os.path.join(local_file, 
                                           'species_topology') + '/*'):
        spc_data = np.load(spc_file).item()
        brute_force_reward.append( spc_data['brute_reward'] )
    brute_force_reward = np.array( brute_force_reward )

    try: spc_num, gen_num = brute_force_reward.shape
    except:
        import pdb; pdb.set_trace()
        pass
    experiment_data = {}

    experiment_data['generation'] = np.array(list( range(gen_num) ))
    experiment_data['x1'] = np.array(list( range(gen_num) ))
    experiment_data['x2'] = np.array(list( range(gen_num) ))
    experiment_data['x3'] = np.array(list( range(gen_num) ))
    experiment_data['avg_r'] = brute_force_reward.mean(axis=0)
    experiment_data['max_r'] = brute_force_reward.max(axis=0)
    experiment_data['min_r'] = brute_force_reward.min(axis=0)

    return experiment_data

def plot_curves(line_x, line_y, xlabel, ylabel, 
                title, legends=None, filename=None,
                fontsize=14,
                line_color=None,
                line_style=None,
                line_w=3.0,
                figsize=(8, 6)
               ):
    ''' plot multiple curves
        input: lines: a list of lists lines[i] is a curve indexed by its natural index
               xlabel, ylabel: the name of the axis
               title: title shown on top
               legends: the name of the curves
               filename: if exists then save to directory
    '''
    # sanity check for configurations of the line
    if legends != None: assert len(line_y) == len(legends)
    if line_color != None: assert len(line_y) == len(line_color)
    else:
        if len(line_y) <= len(light_colors): line_color = light_colors
        else: assert 0, 'Not enough color supported'
    if line_style != None: assert len(line_y) == len(line_style)
    else: line_style = ['-'] * len(line_y)
    
    plt.close()

    # some other options
    matplotlib.rcParams.update({'font.size': fontsize})
    
    # start plotting
    fig = plt.figure(1, figsize=figsize)
    
    # plt.subplots_adjust(right=1)    
    for i, line_i in enumerate(line_y):
        line_i = line_y[i]
        x = line_x[i]
        plt.plot(line_x[i], line_i, label=legends[i], 
                 color=line_color[i],
                 linestyle=line_style[i],
                 linewidth=line_w
                )
         
        plt.title(title)
        plt.legend(shadow=True, fancybox=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plt.grid(linestyle='--')
    plt.legend(loc=4, borderaxespad=0.)
    # plt.show()
    if filename is not None:
        f_path = os.path.join('report', 'figures', filename + '.pdf')
        plt.savefig(filename, bbox_inches='tight')
    return

def plot_curves_shaded(line_x, line_y, xlabel, ylabel,
                       title, legends=None, filename=None,
                       fontsize=14,
                       line_color=None,
                       line_style=None,
                       line_w=3.0,
                       sliding_window=5):
    # sanity check for configurations of the line

    if legends != None: 
        if len(line_y) != len(legends):
            import pdb; pdb.set_trace()
        assert len(line_y) == len(legends)
    if line_color != None: assert len(line_y) == len(line_color)
    else:
        if len(line_y) <= len(light_colors): line_color = light_colors
        else: assert 0, 'Not enough color supported'
    if line_style != None: assert len(line_y) == len(line_style)
    else: line_style = ['-'] * len(line_y)
    
    plt.close()
    for i in range(len(line_x)):
        new_x = np.array(list(range(200)))
        new_y = np.interp(new_x, line_x[i], line_y[i])
        line_x[i] = new_x.tolist()
        line_y[i] = new_y.tolist()

    # some other options
    matplotlib.rcParams.update({'font.size': fontsize})
    
    # start plotting
    fig = plt.figure(1, figsize=(8, 6))
    
    y_std = []
    for i, line_i in enumerate(line_y):
        std_i = []
        for j in range(len(line_i) - 2 * sliding_window):
            sample = line_i[j:j+2*sliding_window]
            std_i.append( np.array(sample).std() )

        std_i = [std_i[0]] * (2 * sliding_window) + std_i
        y_std.append(np.array(std_i))

    from matplotlib import pyplot as pl
    plot = {}
    for x in ['Updates', 'Reward', 'Method']:
        plot[x] = []

    for i, line_i in enumerate(line_y):
        line_i = line_y[i]
        std_i = y_std[i]

        x = np.array(line_x[i]) * 20
        x = x.tolist() 

        # pl.plot(x, line_i, label=legends[i],
        #         color=line_color[i],
        #         linestyle=line_style[i],
        #         linewidth=line_w)
        
        # pl.fill_between(x, line_i - std_i, line_i + std_i,
        #         color=tuple(np.clip(np.array(line_color[i])+0.3, 0, 1)))
        y1 = np.array(line_i)
        y2 = line_i - std_i
        y3 = line_i + std_i

        plot['Updates'] += x * 3
        plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
        # import pdb; pdb.set_trace()

        plot['Method'] += [legends[i]] * 3 * len(x)

    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(plot)
    sns.lineplot(data=df, x='Updates', y='Reward', hue='Method',
                palette=line_color)
    plt.legend('bottom right')


    # plt.title(title)
    # plt.legend(shadow=True, fancybox=True)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)

    plt.grid(linestyle='--')
    plt.legend(loc=4, borderaxespad=0.)
    if filename is not None:
        f_path = os.path.join('report', 'figures', filename + '.pdf')
        plt.savefig(filename, bbox_inches='tight')

def smooth_curve(x, y, window):
    '''
    '''
    old_x = x
    new_x = []
    new_y = []
    for i in range(len(y) - window):
        sample = np.array( y[i: i+window] )
        new_x.append(x[i])
        new_y.append(sample.mean())
    
    new_y = np.interp(np.array(old_x), np.array(new_x), np.array(new_y))

    return np.array(old_x), new_y





def plot_curves_shadedV2(line_x, line_y, xlabel, ylabel,
                       title, legends=None, filename=None,
                       fontsize=14,
                       line_color=None,
                       line_style=None,
                       line_w=3.0,
                       sliding_window=5):
    '''

    '''

    from matplotlib import pyplot as pl
    

    
    plt.close()

    # some other options
    matplotlib.rcParams.update({'font.size': fontsize})
    
    # start plotting
    fig = plt.figure(1, figsize=(8, 6))
        
    # plot nge
    pl.plot(line_x[6], line_y[6], label=legends[0],
            color=line_color[0],
            linestyle=line_style[0],
            linewidth=line_w)
    
    nte_std = []
    for j in range(len(line_x[6]) - 2 * sliding_window):
        sample = line_y[6][j:j+2*sliding_window]
        nte_std.append( np.array(sample).std() )
    nte_std = [nte_std[0]] * (2 * sliding_window) + nte_std
    nte_std = np.array(nte_std)

    pl.fill_between(line_x[6], line_y[6] - nte_std, line_y[6] + nte_std,
            color=tuple(np.clip(np.array(line_color[0]) + 0.6, 0, 1)))

    # plot es
    y1 = np.interp(np.array(line_x[4]), np.array(line_x[0]), np.array(line_y[0]))
    y2 = np.interp(np.array(line_x[4]), np.array(line_x[2]), np.array(line_y[2]))
    y3 = np.array(line_y[4])
    pl.plot(line_x[4], (y1 + y2 + y3) / 3, label=legends[1],
            color=line_color[1],
            linestyle=line_style[1],
            linewidth=line_w)

    total_y = np.stack((y1, y2, y3))
    pl.fill_between(line_x[4], total_y.min(0), total_y.max(0),
            color=tuple(np.clip(np.array(line_color[1]) + 0.6, 0, 1)))

    # plot rs
    y1 = np.interp(np.array(line_x[1]), np.array(line_x[3]), np.array(line_y[3]))
    y2 = np.interp(np.array(line_x[1]), np.array(line_x[5]), np.array(line_y[5]))
    y3 = np.array(line_y[1])
    pl.plot(line_x[4], (y1 + y2 + y3) / 3, label=legends[2],
            color=line_color[2],
            linestyle=line_style[2],
            linewidth=line_w)

    total_y = np.stack((y1, y2, y3))
    pl.fill_between(line_x[4], total_y.min(0), total_y.max(0),
            color=tuple(np.clip(np.array(line_color[2]) + 0.6, 0, 1)))


    plt.title(title)
    plt.legend(shadow=True, fancybox=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(linestyle='--')
    plt.legend(loc=4, borderaxespad=0.)
    if filename is not None:
        f_path = os.path.join('report', 'figures', filename + '.pdf')
        plt.savefig(filename, bbox_inches='tight')


def plot_curves_shaded_seaborn(line_x, line_y, xlabel, ylabel,
                       title, legends=None, filename=None,
                       fontsize=14,
                       line_color=None,
                       line_style=None,
                       line_w=3.0,
                       sliding_window=5):
    '''

    '''

    from matplotlib import pyplot as pl
    import seaborn as sns
    import pandas as pd

    plot = {}

    
    plt.close()

    # some other options
    matplotlib.rcParams.update({'font.size': fontsize})
    
    # start plotting
    fig = plt.figure(1, figsize=(8, 6))
        
    # plot nge
    
    nte_std = []
    for j in range(len(line_x[6]) - 2 * sliding_window):
        sample = line_y[6][j:j+2*sliding_window]
        nte_std.append( np.array(sample).std() )
    nte_std = [nte_std[0]] * (2 * sliding_window) + nte_std
    nte_std = np.array(nte_std)

    y1 = np.array(line_y[6])
    y2 = np.array(line_y[6]) - nte_std
    y3 = np.array(line_y[6]) + nte_std

    plot['Generation'] = line_x[6].tolist() * 3
    plot['Reward'] = y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] = ['NGE'] * (len(line_x[6]) * 3)


    # plot es
    y1 = np.interp(np.array(line_x[4]), np.array(line_x[0]), np.array(line_y[0]))
    y2 = np.interp(np.array(line_x[4]), np.array(line_x[2]), np.array(line_y[2]))
    y3 = np.array(line_y[4])
    
    plot['Generation'] += line_x[4].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['ES'] * (len(line_x[6]) * 3)

    
    # plot rs
    y1 = np.interp(np.array(line_x[1]), np.array(line_x[3]), np.array(line_y[3]))
    y2 = np.interp(np.array(line_x[1]), np.array(line_x[5]), np.array(line_y[5]))
    y3 = np.array(line_y[1])
    
    plot['Generation'] += line_x[1].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['RGS'] * (len(line_x[1]) * 3)



    try:
        df = pd.DataFrame(plot)
    except:
        import pdb; pdb.set_trace()
    sns.lineplot(data=df, x='Generation', y='Reward', hue='Method',
                palette=line_color)
    # sns.tsplot(df)

    plt.grid(linestyle='--')
    # plt.legend(loc=4, borderaxespad=0.)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

def plot_curves_shaded_seaborn_fish_rebuttal(line_x, line_y, xlabel, ylabel,
                       title, legends=None, filename=None,
                       fontsize=14,
                       line_color=None,
                       line_style=None,
                       line_w=3.0,
                       sliding_window=5):
    '''

    '''

    from matplotlib import pyplot as pl
    import seaborn as sns
    import pandas as pd

    plot = {}

    
    plt.close()

    # some other options
    matplotlib.rcParams.update({'font.size': fontsize})
    
    # start plotting
    fig = plt.figure(1, figsize=(8, 6))
        
    # plot nge
    nte_std = []
    for j in range(len(line_x[12]) - 2 * sliding_window):
        sample = line_y[12][j:j+2*sliding_window]
        nte_std.append( np.array(sample).std() )
    nte_std = [nte_std[0]] * (2 * sliding_window) + nte_std
    nte_std = np.array(nte_std)

    y1 = np.array(line_y[12])
    y2 = np.array(line_y[12]) - nte_std
    y3 = np.array(line_y[12]) + nte_std

    plot['Generation'] = line_x[12].tolist() * 3
    plot['Reward'] = y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] = ['NGE'] * (len(line_x[12]) * 3)

    # plot rs
    y1 = np.interp(np.array(line_x[12]), np.array(line_x[0]), np.array(line_y[0]))
    y2 = np.interp(np.array(line_x[12]), np.array(line_x[3]), np.array(line_y[3]))
    y3 = np.array(line_y[6])
    
    plot['Generation'] += line_x[12].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['ESS-Sims-AF'] * (len(line_x[12]) * 3)

    # plot random graph search (Brute force)
    y1 = np.interp(np.array(line_x[12]), np.array(line_x[5]), np.array(line_y[5]))
    y2 = np.interp(np.array(line_x[12]), np.array(line_x[9]), np.array(line_y[9]))
    y3 = np.array(line_y[1])
    
    plot['Generation'] += line_x[12].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['RGS'] * (len(line_x[12]) * 3)

    
    

    # plot Sims
    y1 = np.interp(np.array(line_x[12]), np.array(line_x[2]), np.array(line_y[2]))
    y2 = np.interp(np.array(line_x[12]), np.array(line_x[11]), np.array(line_y[11]))
    y = (y1 + y2) / 2.0
    std_list = []
    for j in range(len(line_x[12]) - 2 * sliding_window):
        sample = y[j:j+2*sliding_window]
        std_list.append( np.array(sample).std() )
    std_list = [std_list[0]] * (2 * sliding_window) + std_list
    std_list = np.array(std_list)

    y1 = np.array(y)
    y2 = np.array(y) - std_list
    y3 = np.array(y) + std_list

    plot['Generation'] += line_x[12].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['ESS-Sims'] * (len(line_x[12]) * 3)

    # plot Sims-GM-UC
    y1 = np.interp(np.array(line_x[12]), np.array(line_x[10]), np.array(line_y[10]))
    # y2 = np.interp(np.array(line_x[12]), np.array(line_x[13]), np.array(line_y[13]))
    y3 = np.array(line_y[4])
    y2 = (y1 + y3) / 2.0

    plot['Generation'] += line_x[12].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['ESS-GM-UC'] * (len(line_x[12]) * 3)


    # plot Sims-BodyShare
    y1 = np.interp(np.array(line_x[12]), np.array(line_x[7]), np.array(line_y[7]))
    y2 = np.interp(np.array(line_x[12]), np.array(line_x[8]), np.array(line_y[8]))
    old_y2 = np.copy(y2)
    _, y = smooth_curve(np.array(line_x[12]), y2, 30)

    y = old_y2.tolist()[:50] + y.tolist()[50:]
    y = np.array(y)
    std_list = []
    sliding_window_tmp = 30
    for j in range(len(line_x[12]) - 2 * sliding_window_tmp):
        sample = y[j:j+2*sliding_window_tmp]
        std_list.append( np.array(sample).std() )
    std_list = [std_list[0]] * (2 * sliding_window_tmp) + std_list
    std_list = np.array(std_list)

    y1 = np.array(y)
    y2 = np.array(y) - std_list
    y3 = np.array(y) + std_list

    plot['Generation'] += line_x[12].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['ESS-BodyShare'] * (len(line_x[12]) * 3)



    try:
        df = pd.DataFrame(plot)
    except:
        import pdb; pdb.set_trace()

    sns.lineplot(data=df, x='Generation', y='Reward', hue='Method',
                palette=line_color)
    # sns.tsplot(df)

    plt.grid(linestyle='--')
    # plt.legend(loc=4, borderaxespad=0.)
    if filename is not None:
        plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()

    # try out log scale
    plot['Reward'] = [max(0, float(np.log(max(x, 0) + 1e-10))) for x in plot['Reward']]
    df = pd.DataFrame(plot)
    sns.lineplot(data=df, x='Generation', y='Reward', hue='Method',
                palette=line_color)
    plt.grid(linestyle='--')
    if filename is not None:
        plt.savefig(filename + '_log.pdf', bbox_inches='tight')




def plot_curves_shaded_seaborn_walker_rebuttal(line_x, line_y, xlabel, ylabel,
                       title, legends=None, filename=None,
                       fontsize=14,
                       line_color=None,
                       line_style=None,
                       line_w=3.0,
                       sliding_window=5):
    '''

    '''

    from matplotlib import pyplot as pl
    import seaborn as sns
    import pandas as pd

    plot = {}

    
    plt.close()

    # some other options
    matplotlib.rcParams.update({'font.size': fontsize})
    
    # start plotting
    fig = plt.figure(1, figsize=(8, 6))
        
    # plot nge
    nte_std = []
    for j in range(len(line_x[11]) - 2 * sliding_window):
        sample = line_y[11][j:j+2*sliding_window]
        nte_std.append( np.array(sample).std() )
    nte_std = [nte_std[0]] * (2 * sliding_window) + nte_std
    nte_std = np.array(nte_std)

    y1 = np.array(line_y[11])
    y2 = np.array(line_y[11]) - nte_std
    y3 = np.array(line_y[11]) + nte_std

    plot['Generation'] = line_x[11].tolist() * 3
    plot['Reward'] = y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] = ['NGE'] * (len(line_x[11]) * 3)

    # plot rs
    y1 = np.interp(np.array(line_x[11]), np.array(line_x[1]), np.array(line_y[1]))
    y2 = np.interp(np.array(line_x[11]), np.array(line_x[5]), np.array(line_y[5]))
    y3 = np.array(line_y[8])
    
    plot['Generation'] += line_x[11].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['ESS-Sims-AF'] * (len(line_x[11]) * 3)

    # plot random graph search (Brute force)
    y1 = np.interp(np.array(line_x[11]), np.array(line_x[7]), np.array(line_y[7]))
    y2 = np.interp(np.array(line_x[11]), np.array(line_x[9]), np.array(line_y[9]))
    y3 = np.array(line_y[4])
    
    plot['Generation'] += line_x[11].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['RGS'] * (len(line_x[11]) * 3)

    
    

    # plot Sims
    y1 = np.interp(np.array(line_x[11]), np.array(line_x[2]), np.array(line_y[2]))
    y2 = np.interp(np.array(line_x[11]), np.array(line_x[6]), np.array(line_y[6]))
    y = (y1 + y2) / 2.0
    std_list = []
    for j in range(len(line_x[11]) - 2 * sliding_window):
        sample = y[j:j+2*sliding_window]
        std_list.append( np.array(sample).std() )
    std_list = [std_list[0]] * (2 * sliding_window) + std_list
    std_list = np.array(std_list)

    y1 = np.array(y)
    y2 = np.array(y) - std_list
    y3 = np.array(y) + std_list

    plot['Generation'] += line_x[11].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['ESS-Sims'] * (len(line_x[11]) * 3)

    # plot Sims-GM-UC
    y = np.array(line_y[10])
    std_list = []
    for j in range(len(line_x[11]) - 2 * sliding_window):
        sample = y[j:j+2*sliding_window]
        std_list.append( np.array(sample).std() )
    std_list = [std_list[0]] * (2 * sliding_window) + std_list
    std_list = np.array(std_list)

    y1 = np.array(y)
    y2 = np.array(y) - std_list
    y3 = np.array(y) + std_list

    plot['Generation'] += line_x[11].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['ESS-GM-UC'] * (len(line_x[11]) * 3)


    # plot Sims-BodyShare
    y1 = np.interp(np.array(line_x[11]), np.array(line_x[0]), np.array(line_y[0]))
    y2 = np.interp(np.array(line_x[11]), np.array(line_x[3]), np.array(line_y[3]))
    y = (y1 + y2) / 2.0

    std_list = []
    sliding_window_tmp = 30
    for j in range(len(line_x[11]) - 2 * sliding_window_tmp):
        sample = y[j:j+2*sliding_window_tmp]
        std_list.append( np.array(sample).std() )
    std_list = [std_list[0]] * (2 * sliding_window_tmp) + std_list
    std_list = np.array(std_list)

    y1 = np.array(y)
    y2 = np.array(y) - std_list
    y3 = np.array(y) + std_list

    plot['Generation'] += line_x[11].tolist() * 3
    plot['Reward'] += y1.tolist() + y2.tolist() + y3.tolist()
    plot['Method'] += ['ESS-BodyShare'] * (len(line_x[11]) * 3)



    try:
        df = pd.DataFrame(plot)
    except:
        import pdb; pdb.set_trace()

    sns.lineplot(data=df, x='Generation', y='Reward', hue='Method',
                palette=line_color)
    # sns.tsplot(df)

    plt.grid(linestyle='--')
    # plt.legend(loc=4, borderaxespad=0.)
    if filename is not None:
        plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()

    # try out log scale
    plot['Reward'] = [float(np.log(x)) for x in plot['Reward']]
    df = pd.DataFrame(plot)
    sns.lineplot(data=df, x='Generation', y='Reward', hue='Method',
                palette=line_color)
    if filename is not None:
        plt.savefig(filename + '_log.pdf', bbox_inches='tight')
