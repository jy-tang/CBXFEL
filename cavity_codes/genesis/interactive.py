#
# Interactive plotting using bokeh
#
#Note: When debugging code it is helpful to call functions with curdoc doc (rather than throug show(app)) so that python will actually print errors. Use "from bokeh.io import curdoc" to get curdoc.
from genesis import parsers

from bokeh import palettes, colors
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models.widgets import Slider, Select
from bokeh.layouts import column, row

import numpy as np
import os


pal = palettes.Viridis[256]

def interactive_field_history(doc, fld=None, slice=0, dgrid=0):
    """         
    
    Use with code similar to:
    
    my_fld =  parsers.parse_genesis_fld(fld_fname, g.input_params['ncar'], nslice)
    
    from bokeh.plotting import show, output_notebook
    output_notebook()
    
    def app(doc):
       return interactive_field_history(doc, fld=my_fld, dgrid=dgrid, slice = 0)
    show(app)
    
    
    """
    
    
    nhist = len(fld)
    
    ihist = nhist-1
    
    fdat = fld[ihist][slice]
    
    d = np.angle(fdat)
    ds = ColumnDataSource(data=dict(image=[d]))
    
    xyrange = (-1000*dgrid, 1000*dgrid)
    p = figure(x_range=xyrange, y_range=xyrange, title='Phase',  
           plot_width=500, plot_height=500, x_axis_label='x (mm)', y_axis_label='y (mm)')
    p.image(image='image', source=ds, 
            x=-dgrid*1000, y=-dgrid*1000, dw=2*dgrid*1000, dh=2*dgrid*1000, palette=pal)
    
    slider = Slider(start=0, end=nhist-1, value=ihist, step=1, title='History')

    def handler(attr, old, new):
        fdat = fld[new][slice]
        d = np.angle(fdat)
        ds.data = ColumnDataSource(data=dict(image=[d])).data
    
    slider.on_change('value', handler)

    doc.add_root(column(slider, p))
    
def genesis_interactive_field_history(doc, genesis=None):
    """
    Convenience routine to pass the whole genesis object to
    """
    
    # Parameters
    p = genesis.input
    
    # Check for time dependence
    if p['itdp'] == 0:
        nslice = 1
    else:
        nslice = p['nslice']
    
    fld_fname = os.path.join(genesis.path, p['outputfile']+'.fld')
    my_fld =  parsers.parse_genesis_fld(fld_fname, p['ncar'], nslice)
    
    return interactive_field_history(doc, fld=my_fld, slice=0, dgrid=p['dgrid'] )  


#######################################
#######################################
#######################################

def interactive_particle_history(doc, bunch=None, zpos=0):
    """         
    Use with code similar to:
    
    npart = G.input['npart']
    part_fname = os.path.join(G.path, G.input['outputfile']+'.par')
    bunch=parsers.parse_genesis_dpa(part_fname,npart)
    
    from bokeh.plotting import show, output_notebook
    output_notebook()
    
    def app(doc):
        return interactive.interactive_particle_history(doc, bunch=bunch)
    show(app)
    
    
    """
    
    
    nhist = len(bunch)
    
    ihist = nhist-1
    
    height=300
    width=int(height*1.61)
    
    plotobj = figure(plot_height=height, plot_width=width, title="Particle Data")
    
    #slider for zpos
    slider = Slider(start=0, end=nhist-1, value=ihist, step=1, title='Slice')
    
    #toggles for different axes
    options=["Gamma","Phase","x","y","Px/mc","Py/mc"]
    select1=Select(title="x axis:", value="Phase", options=options)
    select2=Select(title="y axis:", value="Gamma", options=options)

    #inital params    & plot

    xdata=bunch[zpos,1,:] #phase
    ydata=bunch[zpos,0,:]*0.51099895 #gamma
    source = ColumnDataSource(data=dict(x=xdata, y=ydata)) #source is the thing being plotted. It will be updated by clicking on buttons.
    
    #create class for params. This way the callback functions can modify the plot parameters...
    class plot_params(object):
        def __init__(self,ppx="Phase",ppy="Gamma",z=0):
            self.ppx=ppx
            self.ppy=ppy
            self.z=z
        def set_ppx(self,val=None):
            self.ppx=val
        def set_ppy(self,val=None):
            self.ppy=val
        def set_z(self,val=None):
            self.z=val
    pps=plot_params()
    def update_data(pps=pps):              
        zidx=int(pps.z)
        if pps.ppx=='Gamma': 
            xdata=bunch[zidx,0,:]*0.51099895
            plotobj.xaxis.axis_label='Gamma [MeV]'
        elif pps.ppx=='Phase':
            xdata=bunch[zidx,1,:]
            plotobj.xaxis.axis_label='Phase [rad]'
        elif pps.ppx=='x':
            xdata=bunch[zidx,2,:]*1e6
            plotobj.xaxis.axis_label='x [um]'
        elif pps.ppx=='y':
            xdata=bunch[zidx,3,:]*1e6
            plotobj.xaxis.axis_label='y [um]'
        elif pps.ppx=='Px/mc':
            xdata=bunch[zidx,4,:]*1e3
            plotobj.xaxis.axis_label='Px/mc [mrad]'
        elif pps.ppx=='Py/mc':
            xdata=bunch[zidx,5,:]*1e3
            plotobj.xaxis.axis_label='Py/mc [mrad]'

        if pps.ppy=='Gamma': 
            ydata=bunch[zidx,0,:]*0.51099895
            plotobj.yaxis.axis_label='Gamma [MeV]'
        elif pps.ppy=='Phase':
            ydata=bunch[zidx,1,:]
            plotobj.yaxis.axis_label='Phase [rad]'
        elif pps.ppy=='x':
            ydata=bunch[zidx,2,:]*1e6
            plotobj.yaxis.axis_label='x [um]'
        elif pps.ppy=='y':
            ydata=bunch[zidx,3,:]*1e6
            plotobj.yaxis.axis_label='y [um]'
        elif pps.ppy=='Px/mc':
            ydata=bunch[zidx,4,:]*1e3
            plotobj.yaxis.axis_label='Px/mc [mrad]'
        elif pps.ppy=='Py/mc':
            ydata=bunch[zidx,5,:]*1e3
            plotobj.yaxis.axis_label='Py/mc [mrad]'
        source.data = dict(x=xdata, y=ydata)
        
    
    def slider_fun(attr, old, new):
        pps.set_z(new)
        update_data(pps)
    def select_fun1(attr, old, new):
        pps.set_ppx(new)
        update_data(pps)
    def select_fun2(attr, old, new):
        pps.set_ppy(new)
        update_data(pps) 
        
    slider.on_change('value', slider_fun)
    select1.on_change('value', select_fun1)
    select2.on_change('value', select_fun2)
    
    #run update once and plot
    update_data(pps)
    marker_style = dict(marker='circle',size=10,alpha=0.25)
    plotobj.scatter('x','y',source=source,**marker_style)    
    
    r1=row(select1,select2,**dict(width=width))
    doc.add_root(column(slider,r1,plotobj))
    
def genesis_interactive_particle_history(doc, genesis=None):
    """
    Convenience routine to pass the whole genesis object to
    """
    
    # Parameters
    p = genesis.input     
    npart = p['npart']
    part_fname = os.path.join(genesis.path, p['outputfile']+'.par')
    
    #Parse .par
    my_bunch=parsers.parse_genesis_dpa(part_fname,npart)
    
    return interactive_particle_history(doc, bunch=my_bunch, zpos=0)  
    