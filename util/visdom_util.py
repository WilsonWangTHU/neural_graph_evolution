''' visdom related functions to print the curves
'''

import pdb
from visdom import Visdom
import numpy as np
import time

# viz = Visdom(server='http://18.218.194.87', port=4212) # aws server port

viz = None

def visdom_initialize(args):
    '''
    '''
    global viz

    if args.vis_port < 4212 or args.vis_port > 4223:
        assert 0, 'Visdom port %d not supported' % (args.vis_port)

    if args.vis_server == 'local':
        viz = Visdom()
    else:
        viz = Visdom(server=args.vis_server, port=args.vis_port,
                use_incoming_socket=False)

    return None


def visdom_plot_curve(it, val, viz_win=None, title=''):
    '''
    '''
    if it == 1 or viz_win == None:
        return viz.line(X=np.array([it]),
                        Y=np.array([val]),
                        win=viz_win,
                        opts={'title': title})
    else:
        return viz.line(win=viz_win, update='append',
                        X=np.array([it]),
                        Y=np.array([val]))
    return viz_win

def visdom_print_info(str_list, mytitle):
    """Visdom, display text info to the visdom window
    input: 1. a list of strings
    """
    mystr = ""

    for i in range(len(str_list)):
        mystr = mystr + str_list[i] + "\n" + "<br>"

    opts = {'title': mytitle}
    viz.text(mystr, opts=opts)

    return None

def viz_line(i, vals, viz_win=None, 
             title='', xlabel='', ylabel='', 
             legend=None):
    ''' a more robust way to print multiple values on the same plot
        NOTE:
            this function only supports 2 or more valus. for plotting only one value,
            refer to visdom_plot_curve()
        UPDATE:
            now it data input supports >= 1 plotting
    '''
    data_num = len(vals)
    if legend != None: assert data_num == len(legend)
    else: legend = ['' for _ in range(data_num)]

    # make the input compatible with visdom API
    X = [np.array([i]) for _ in range(data_num)]
    Y = [np.array([val]) for val in vals]
    if data_num != 1:
        X = np.column_stack(X)
        Y = np.column_stack(Y)
    else:
        X = X[-1]
        Y = Y[-1]
    
    if viz_win is None:
        return viz.line(X=X, Y=Y,
                        opts=dict(title=title,
                                  legend=legend,
                                  xlabel=xlabel,
                                  ylabel=ylabel))
    else:
        return viz.line(win=viz_win, update='append',
                        X=X, Y=Y,
                        opts=dict(title=title,
                                  legend=legend))

def vis_stem_as_hist(samples):
    ''' using visdom to plot the histogram of 1 or more sample sequence
    '''
    raise NotImplementedError

    return

def vis_hist_plot(i, samples, viz_win=None, title='', numbin=20):
    ''' directly use visdom histogram plot to visualize the sample sequence
    '''
    low, high = samples.min(), samples.max()
    title = title + 'low:%.2f-high:%.2f-Gen:%d' % (float(low), float(high), i)
    if viz_win == None:
        return viz.histogram(X=samples,
                             opts=dict(numbins=numbin,
                                       title=title))
    else:
        return viz.histogram(win=viz_win, X=samples,
                             opts=dict(numbins=numbin,
                                       title=title))
    return


if __name__ == '__main__':

    win = None

    # for i in range(100):
    #     win = visdom_plot_curve(i, i, viz_win=win, title='test')

    for i in range(100):
        time.sleep(1)
        samples = np.random.normal(np.random.randint(5), 1, 10000)
        # samples = np.random.uniform(0, 5, 10000)

        win = vis_hist_plot(i, samples, viz_win=win, title='misc')
