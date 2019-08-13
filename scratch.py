from main import UnstructuredMetaModelVisualization
# Import 

# Bokeh Imports
from bokeh.io import output_file, show, curdoc
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.models import Axis, Slider, ColumnDataSource

# Misc Imports
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import math


# ======================================================================
#                       Engine Setup
# ======================================================================
# Load Data from John:
tmp = np.loadtxt('UHB.outputFLOPS')

# net thrust = gross thrust - ram drag (and convert to N)
tmp[:, 3] = (tmp[:, 3] - tmp[:, 4])*4.44822162

# Need to replace power code with fraction Tmax since there is not
# enough precision:
for i in range(len(tmp)):
    if tmp[i, 2] == 50:
        tmax = tmp[i, 3]
    tmp[i, 2] = tmp[i, 3] / tmax

# Converting units and identifying column titles
engineOptions = {'mach':tmp[:, 0],
           'altitude':tmp[:, 1]*.3048, # Alt in m
           'throttle':tmp[:, 2],
           'thrust':tmp[:, 3],
           'tsfc':tmp[:, 6]/3600.0, # TSFC in 1/s
           'rbfType':'cubic'}

# Creating empty arrays
nt = len(tmp)
xt = np.zeros((nt, 3))
yt = np.zeros((nt, 2))

# Mach in column 0 of xt
xt[:, 0] = tmp[:, 0]
# Altitude in column 1 of xt
xt[:, 1] = tmp[:, 1] / 1e3
# Trottle in column 2 of xt
xt[:, 2] = tmp[:, 2]

# Thrust in column 0 of yt
yt[:, 0] = tmp[:, 3] / 1e5
# tsfc in column 1 of yt
yt[:, 1] = tmp[:, 6] / 3600.

# Set the limits of x
xlimits = np.array([
    [0.0, 0.9],
    [0., 43.],
    [0, 1]
])

# Initial surrogate call
#### CHANGE THIS TO KRIGING SURROGATE WHEN GENERAL PLOTS ARE WORKING
interp = om.MetaModelUnStructuredComp(default_surrogate=om.KrigingSurrogate())
# Inputs
interp.add_input('Mach', 0., training_data=xt[:, 0])
interp.add_input('Alt', 0., training_data=xt[:, 1])
interp.add_input('Throttle', 0., training_data=xt[:, 2])

# Outputs
interp.add_output('Thrust', training_data=yt[:, 0])
interp.add_output('TSFC', training_data=yt[:, 1])

# Create the problem setup
prob = om.Problem()
prob.model.add_subsystem('interp', interp)
prob.setup()


info = {'num_of_inputs':3,
        'num_of_outputs':2,
        'resolution':50,
        'dimension_names':[
            'Mach number',
            'Altitude, kft',
            'Throttle'],
        'X_dimension':0,
        'Y_dimension':1,
        'scatter_points':[xt, yt],
        'output_variable': 0,
        'dist_range': .1,
        'output_names':[
            'Thrust, 1e5 N',
            'TSFC, 1/s']
}

viz = UnstructuredMetaModelVisualization(info)



    # def contour_update(self, attr, old, new):
    #     self.layout.children[1] = self.contour_data()

    # def subplot(self):
    #     # print(self.source.data['z'])
    #     mach_value = self.mach_slider.value

    #     mach_index = np.where(np.around(self.mach, 5) == np.around(mach_value, 5))[0]
    #     z_data = self.Z[mach_index].flatten()

    #     self.source.data = dict(left_slice=z_data)

    #     s1 = figure(plot_width=200, plot_height=500, y_range=(0,max(self.alt)), title="Altitude vs Thrust")
    #     s1.xaxis.axis_label = "Thrust"
    #     s1.yaxis.axis_label = "Altitude"
    #     s1.line(self.source.data['left_slice'], self.slider_source.data['alt'])

    #     return s1

    # def subplot_update(self, attr, old, new):
    #     self.layout.children[0] = self.subplot()