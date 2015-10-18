import numpy as np
import sys

class TikzPicture:

    def __init__(self, name, title, xlabel, ylabel):
        self.name = name
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plots = []

    def add_plot(self, plot):
        self.plots.append(plot)

    def generate(self):
        name = str(self.name) + '_results.txt'
        f = file(name, 'w')
        orig_out = sys.stdout
        sys.stdout = f
        self._print_header()
        for plot in self.plots:
            self._print_plot(plot)
        self._print_footer()
        f.close()
        sys.stdout = orig_out

    def _print_header(self):
        xmax = self._get_max('x')
        ymax = self._get_max('y')
        xticks = range(0, xmax+1, 124)
        print '\\begin{tikzpicture}'
        print '\\begin{axis}['
        print '    title={' + self.title + '},'
        print '    xlabel={' + self.xlabel + '},'
        print '    ylabel={' + self.ylabel + '},'
        print '    xmin=0, xmax=' + str(xmax) + ','
        print '    ymin=0, ymax=' + str(ymax) + ','
        print '    xtick={',
        for i in range(len(xticks)-1):
            print str(xticks[i]) + ',',
        if (len(xticks) > 0):
            print str(xticks[len(xticks)-1]) + '},'
        else:
            print '},'
        print '    legend pos=north west,'
        print '    ymajorgrids=true,'
        print '    grid style=dashed,'
        print ']'

    def _print_plot(self, plot):
        print '\\addplot['
        print '    color=' + plot.color + ','
        print '    mark=' + plot.mark + ','
        print '    mark size=1.4'
        print '    ]'
        print '    coordinates {'
        print '        ',
        # use average of the points
        s = {}
        xs = []
        for p in plot.points:
            xs.append(p['x'])
        xs = sorted(set(xs))
        for x in xs:
            s[x] = []
        for p in plot.points:
            s[p['x']].append(p['y'])
        for x in xs:
            s[x] = np.mean(s[x])
            print '(' + str(abs(x)) + ',' + str(abs(s[x])) + ') ',
        print
        print '    };'

    def _print_footer(self):
        print '\\legend{',
        for plot in self.plots:
            print str(plot.legend) + ', ',
        print '}'
        print ''
        print '\\end{axis}'
        print '\\end{tikzpicture}'

    def _get_max(self, axis):
        r = -1
        for plot in self.plots:
            for point in plot.points:
                r = max(r, point[axis])
        return r


class TikzPlot:

    def __init__(self, legend, color, mark):
        self.legend = legend
        self.color = color
        self.mark = mark
        self.points = []

    def add_point(self, x, y):
        self.points.append({'x': x, 'y': y})

