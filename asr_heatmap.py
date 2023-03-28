#!/usr/bin/env python

import pygraphviz as pgv
import seaborn as sns
import numpy as np
import re
import argparse

# Note: the dot file must be generated with the following command:
# nnviz resnet18k.py:make_resnet18k -d 3 -o resnet18k.dot

def make_slice(expr):
    # Convert a string to a slice or a simple integer index
    try:
        return int(expr)
    except ValueError:
        pass
    def to_piece(s):
        return s and int(s) or None
    pieces = list(map(to_piece, expr.split(':')))
    if len(pieces) == 1:
        return slice(pieces[0], pieces[0] + 1)
    else:
        return slice(*pieces)

parser = argparse.ArgumentParser()
parser.add_argument('--strat', help='Strategy to plot', default='random', choices=['random', 'targeted'])
parser.add_argument('--init', help='Initialization to plot', default='normal', choices=['uniform', 'normal'])
parser.add_argument('--slice_spec', help='Slice specification for picking stats', default=':,-1,:')
parser.add_argument('--reduce', help='Ruduction function to apply to the stats', default='mean', choices=['mean', 'max', 'min'])
parser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
parser.add_argument('dotfile', help='Path to the dot file to color')
parser.add_argument('outfile', help='Path to the output file')
args = parser.parse_args()

if args.verbose:
    import sys
    # If we have ansicolors we can use it to color the output
    try:
        from colors import color as ansicolor
    except ImportError:
        def ansicolor(s, *args, **kwargs):
            return s

# Set up the color palette
cmap = sns.color_palette("rocket", as_cmap=True)

# Create the slice specification
slice_spec = tuple([make_slice(part) for part in args.slice_spec.split(',')])
if len(slice_spec) != 3:
    parser.error('Invalid slice specification -- must be 3 slices separated by commas')

# Read in the dot file
G = pgv.AGraph(args.dotfile)

# Update the graph title
# label=<<B><FONT POINT-SIZE="48">PreActResNet</FONT></B><BR/><B>Source:</B> resnet18k<BR/><B>NNViz </B>v0.4.0<BR/> >,
title = G.graph_attr['label']
newtitle = f'PartialPGD ASR</FONT></B><BR/><FONT POINT-SIZE="32"><B>{args.strat}</B> attack, <B>{args.init}</B> init</FONT>'
G.graph_attr['label'] = '<' + title.replace('PreActResNet</FONT></B>', newtitle) + '>'

if args.verbose:
    print(f'Attack success rate by layer with {args.strat} attack, {args.init} init', file=sys.stderr)

nodes_to_remove = []
nodelist = list(G.nodes_iter())
# Sort the nodes by their vertical position in the graph
nodelist.sort(key=lambda node: int(node.attr['pos'].split(',')[1]), reverse=True)
for node in nodelist:
    label = node.attr['label']
    # Fully qualified name is in <I>italics</I>
    name = re.search(r'<I>(.*?)</I>', label).group(1)
    try:
        fname = f'data/{args.strat}/{args.init}/layer_model.{name}.npy'
        stats = np.load(fname)
    except FileNotFoundError:
        nodes_to_remove.append(node)
        continue
    # Stats shape is (10, 11, 11) for 10 trials, 11 values of eps, 11 values of alpha
    # We can use the slice_spec to pick out the values we want and then reduce them
    # to a single value with args.reduce
    success = getattr(stats[slice_spec], args.reduce)()
    color = cmap(success)
    # Get hex color
    color = color[:3]
    hexcolor = '#%02x%02x%02x' % tuple(int(c*255) for c in color)
    fillcolor = node.attr['fillcolor']
    node.attr['fillcolor'] = hexcolor
    # Add the success rate to the label
    label = label.replace('</I><BR/>', f'</I><BR/><BR/><B>success</B>: {int(100*success)}%<BR/>')
    node.attr['label'] = f'<{label}>'
    # If the background is too light, change fontcolor to black
    if color[0] + color[1] + color[2] > 1.5:
        node.attr['fontcolor'] = 'black'
    if args.verbose:
        print(ansicolor(f'{name:20} {int(success*100):2d} %', fg=hexcolor), file=sys.stderr)

for node in nodes_to_remove:
    G.delete_node(node)

G.draw(args.outfile, prog='dot')
