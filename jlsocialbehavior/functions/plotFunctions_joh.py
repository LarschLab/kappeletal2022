import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plotMapWithXYprojections(Map2D,projectWidth,targetSubplot,mapLim=31,projectionLim=0.03):
    #plot a heatmap with mean 'projections' left and bottom of heatmap
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 3,subplot_spec=targetSubplot,wspace=0.05, hspace=0.05)
    plt.subplot(inner_grid[:-1,:-1])   
    plt.imshow(Map2D,cmap="bwr",interpolation='gaussian', extent=[-mapLim,mapLim,-mapLim,mapLim],clim=(-projectionLim, projectionLim),origin='lower')
    #plt.title('accel=f(pos_n)')
    plt.ylabel('y [mm]')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',
        right='off',
        top='off',
        labelbottom='off') # labels along the bottom edge are off
    plt.plot([0, 0], [-mapLim, mapLim], 'k:', lw=1)
    plt.plot([-mapLim, mapLim], [0, 0], 'k:', lw=1)
    
    plt.ylim([-mapLim,mapLim])
    plt.xlim([-mapLim,mapLim])
    
    
    plt.subplot(inner_grid[:-1,-1])
    avgRange=np.arange(mapLim-projectWidth,mapLim+projectWidth)
    yprofile=np.nanmean(Map2D[:,avgRange],axis=1)
    x=np.arange(np.shape(yprofile)[0])-(np.ceil(np.shape(yprofile)[0])/2)
    plt.plot(yprofile,x,'b.-',markersize=2)
    plt.xlim([-projectionLim, projectionLim])
    plt.ylim([x[0],x[-1]])
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        top='off',
        left='off',
        labelleft='off') # labels along the bottom edge are off
    plt.locator_params(axis='x',nbins=4)
    plt.plot([0, 0], [-mapLim, mapLim], 'r:', lw=1)
    plt.plot([-mapLim, mapLim], [0, 0], 'r:', lw=1)
    
    plt.subplot(inner_grid[-1,:-1])
    yprofile=np.nanmean(Map2D[avgRange,:],axis=0)
    x=np.arange(np.shape(yprofile)[0])-(np.ceil(np.shape(yprofile)[0])/2)
    plt.plot(x,yprofile,'b.-',markersize=2)
    plt.xlabel('x [mm]')
    plt.ylabel('accel')
    plt.ylim([-projectionLim, projectionLim]) 
    plt.xlim([x[0],x[-1]])
    plt.locator_params(axis='y',nbins=4)
    plt.plot([0, 0], [-mapLim, mapLim], 'r:', lw=1)
    plt.plot([-mapLim, mapLim], [0, 0], 'r:', lw=1)
    return 1