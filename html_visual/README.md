## Tools for visualizing tree structures 

### Prerequisites

1. To enable images in the tree, refer to `/env/visualize_species.py`
2. The visualization is implemented with `d3`. 
It requires the HTML file visited from a local host. 
In Python 3, start the local server from `/` of the project directory by running `python -m http.server`.
In Python 2, the command is `python -m SimpleHTTPServer <port>`


### Visualize genealogy

Genealogy describes how the parent-child relationships are in the entire 'world'.
The reward shown is the best reward for each species ever achieved.

The visualization webpage is in `/html_visual/genealogy.html`

### Visualize evolution

Selecting the top K performing models for each generation and connect different generations via either parent-child relationship, or self-preseving through generation.

The visualization webpage is in `/html_visual/evolution.html`

### Visualize species' change

By specifying a species ID, plot its evolution process over the generation, from its generated till it died. 
Specifically examine all the relavent parent-child relationship and performance increase and drop.