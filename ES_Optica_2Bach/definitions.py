#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import animation

'''
An auxiliary file for Optics_Batch notebooks
Here we define functions that are too messy to appear in the notebooks explicitly.

Notebooks are designed to be used sequentally following this order:
    - SnellLaw
    - Rainbow
    - Lenses
    - Mirrors
    - OpticBench

Some functions are first defined for their use within one notebook, but also used in others.
Make sure modifications do not affect other notebooks.
'''

C = 3e8 # m/s

'''
Functions defined for: SnellLaw
'''

def Wave( fr: float, amp: float, c: float, t: float, xlims: list, phase: float = 0, n_points: int = 1000 ):
    '''
    Coordinates of a wave
    '''
    x = np.linspace( *xlims, n_points )
    y = amp * np.cos( 2 * np.pi * fr * ( x / c - t ) + phase )
    return x, y


def WaveInterface( fr: float, amp: float, c: float, n:float, t: float, xlims: list, phase: float = 0, n_points: int = 1000 ):
    '''
    Coordinates of a wave reaching an interface
    '''
    x = np.linspace( *xlims, n_points )
    x1 = x[:int(n_points/2)]
    x2 = x[int(n_points/2):]

    y1 = amp * np.cos( 2 * np.pi * fr * ( ( x1 - x[int(n_points/2)] ) / c - t ) + phase )
    y2 = amp * np.cos( 2 * np.pi * fr * ( ( x2 - x[int(n_points/2)] ) / (c/n) - t ) + phase )

    return (x1, y1), (x2, y2)


def WaveFront( x_origin: list, y_origin: list, angle: float, x_max: float, y_max: float, n_points: int = 2 ):
    '''
    Coordinates of a wave front
    '''
    def Check_l( x0: float, y0: float, dx:float, dy:float, x_max: float, y_max: float ):
        if (x0+dx)/x_max > (y0+dy)/y_max:
            return (y_max - y0)/dy
        else:
            return (x_max - x0)/dx

    x = []; y = []

    dx = np.cos(angle)
    dy = np.sin(angle)

    for x0, y0 in zip(x_origin, y_origin):

        l = Check_l( x0, y0, dx, dy, x_max, y_max)
        
        x.append( x0+l*dx )
        y.append( y0+l*dy )

    return x, y


def Arc( c, r, ang1, ang2 ):
    '''
    Coordinates of an arch given the center of the circumpherence, its radius and the limiting angles
    '''
    return c[0]+r*np.cos( np.linspace(ang1,ang2,50,endpoint=True) ), c[1]+r*np.sin( np.linspace(ang1,ang2,50,endpoint=True) )


def PlotWaveSpeed( WL: float= 400e-6, N_FRAMES: int = 100, N_OSCILLATIONS: float = 3, LIST_N: list = [1, 1.5, 2] ):
    '''
    Graphic representation of a wave propagating at different speed
    '''
    X_LIMS = [0, N_OSCILLATIONS*WL]

    fig, ax = plt.subplots(nrows=len(LIST_N), sharex=True)
    ax[0].set_autoscale_on = False
    ax[0].set_xlim( *X_LIMS )

    wave = [ ax[m].plot([], [], color="crimson")[0] for m, _ in enumerate(LIST_N) ]

    for m, n in enumerate(LIST_N):
        ax[m].set_ylim( -1.1,1.1 )
        ax[m].set_title( f"n={n} (u={1/n:.3f}c)", loc="left")
        ax[m].set_yticks([])

    def plotframe(i):
        for m, n in enumerate(LIST_N):
            x, y = Wave( fr=C/WL, amp=1, c=C/n, t=i * WL/C * N_OSCILLATIONS / N_FRAMES, xlims=X_LIMS, n_points=int(1e4) )

            wave[m].set_data( x, y )

        return wave

    fig.set_tight_layout(True)
    ax[-1].set_xlabel("Distance (m)")

    return animation.FuncAnimation( fig, plotframe, frames=int(N_FRAMES), blit=True, interval=50 )

def PlotWaveInterface( WL: float= 400e-6, N_FRAMES: int = 100, N_OSCILLATIONS: float = 3, LIST_N: list = [1, 1.5, 2]  ):
    '''
    Graphic representation of a wave reaching the interface with a medium with a different refractive index
    '''
    X_LIMS = [0, N_OSCILLATIONS*WL]

    fig, ax = plt.subplots(nrows=len(LIST_N), sharex=True)
    ax[0].set_autoscale_on = False
    ax[0].set_xlim( *X_LIMS )

    wave = [ ax[m].plot([], [], color="crimson")[0] for m, _ in enumerate(LIST_N) ]

    for m, n in enumerate(LIST_N):
        ax[m].set_ylim( -1.1,1.1 )
        ax[m].set_title( f"n={n} (u={1/n:.3f}c)", loc="right")
        ax[m].set_yticks([])
        ax[m].plot( [np.mean(X_LIMS), np.mean(X_LIMS) ], [-1.1, 1.1], ".-", color="royalblue" )

    def plotframe(i):
        for m, n in enumerate(LIST_N):
            (x1, y1), (x2, y2) = WaveInterface( fr=C/WL, amp=1, c=C, n=n, t=i * WL/C * N_OSCILLATIONS / N_FRAMES, xlims=X_LIMS, n_points=int(1e4) )

            wave[m].set_data( np.concat([x1,x2]), np.concat([y1,y2]) )
        
        return wave

    fig.set_tight_layout(True)
    ax[-1].set_xlabel("Distance (m)")

    return animation.FuncAnimation( fig, plotframe, frames=int(N_FRAMES), blit=True, interval=50 )


def PlotDiagramSnell():
    '''
    A diagram for the deduction of Snell's law
    '''

    fig, ax = plt.subplots( figsize=(8,8) )
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on = False

    X_LIMS = [-0.4, 1.8]
    Y_LIMS = [-1, 1.2]

    ax.set_xlim(X_LIMS)
    ax.set_ylim(Y_LIMS)

    ax.plot( X_LIMS,[0,0], "-", color="royalblue")
    ax.fill_between( X_LIMS,[Y_LIMS[0],Y_LIMS[0]], color="cornflowerblue", alpha=0.4)

    X_LIST = np.arange( X_LIMS[0]-2, X_LIMS[1]+2, 0.2)
    Y_LIST = X_LIST*0

    N_I = 1
    N_R = 1.5
    THETA_I = np.deg2rad(50)
    THETA_R = np.arcsin( N_I*np.sin(THETA_I)/N_R )

    xi, yi = WaveFront( X_LIST, Y_LIST, THETA_I, X_LIMS[1], Y_LIMS[1] )
    xr, yr = WaveFront( X_LIST, Y_LIST, -(np.pi-THETA_R), -X_LIMS[1], -Y_LIMS[1] )

    for n in range(len(X_LIST)):
        ax.plot( [ X_LIST[n], xi[n] ], [ Y_LIST[n], yi[n] ], ":", color="dimgray" )
        ax.plot( [ X_LIST[n], xr[n] ], [ Y_LIST[n], yr[n] ], ":", color="lightgray" )

    O = [ 0, 0 ]
    B = [ 0.8, 0 ]

    A = [ B[0]*np.sin(np.pi/2-THETA_I)*np.cos(THETA_I), B[0]*np.sin(np.pi/2-THETA_I)*np.sin(THETA_I) ]
    P = [ B[0]*np.cos(np.pi/2-THETA_R)*np.sin(THETA_R), -B[0]*np.cos(np.pi/2-THETA_R)*np.cos(THETA_R) ]

    ax.plot( *O, "wo", mec="dimgray" )
    ax.plot( *B , "wo", mec="dimgray")
    ax.plot( *A , "wo", mec="dimgray")
    ax.plot( *P , "wo", mec="dimgray")

    ax.text( *O, "O", fontsize=16, ha="right", va="top" )
    ax.text( *B, "B", fontsize=16, ha="left", va="bottom" )
    ax.text( *A, "A", fontsize=16, ha="left", va="bottom" )
    ax.text( *P, "P", fontsize=16, ha="right", va="top" )

    ax.plot( [O[0], O[0]+np.cos(np.pi/2+THETA_I)*1], [O[1], O[1]+np.sin(np.pi/2+THETA_I)*1], color="darkgray" )
    ax.plot( [B[0], B[0]+np.cos(np.pi/2+THETA_I)*2], [B[1], B[1]+np.sin(np.pi/2+THETA_I)*2], color="darkgray" )
    ax.plot( [O[0], O[0]+np.cos(np.pi/2-THETA_R)*2], [O[1], O[1]-np.sin(np.pi/2-THETA_R)*2], color="darkgray" )
    ax.plot( [B[0], B[0]+np.cos(np.pi/2-THETA_R)*2], [B[1], B[1]-np.sin(np.pi/2-THETA_R)*2], color="darkgray" )

    ax.plot( [O[0], A[0]], [O[1], A[1]], color="darkgray" )
    ax.plot( [P[0], B[0]], [P[1], B[1]], color="darkgray" )

    plt.show()


def PlotSnell( ti, n1, n2 ):
    '''
    Graphic representation of Snell's law
    '''

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on = False

    X_LIMS = [-1, 1.2]
    Y_LIMS = [-1, 1.2]

    O = [ 0, 0 ]

    ax.set_xlim(X_LIMS)
    ax.set_ylim(Y_LIMS)

    ax.plot( X_LIMS,[0,0], "-", color="royalblue")
    ax.fill_between( X_LIMS,[Y_LIMS[0],Y_LIMS[0]], color="cornflowerblue", alpha=0.4)

    ax.plot( [0,0], Y_LIMS, "-.", color="lightgray" )

    ray1, = ax.plot([],[], color="crimson")
    ray2, = ax.plot([],[], color="crimson")

    arc1, = ax.plot([], [], color="coral")
    arc2, = ax.plot([], [], color="royalblue")

    THETA_I = np.deg2rad( ti )
    ray1.set_data( [O[0], O[0]+np.cos(np.pi/2+THETA_I)*2], [O[1], O[1]+np.sin(np.pi/2+THETA_I)*2])
    arc1.set_data( *Arc( (0,0), 0.2, np.pi, np.pi/2+THETA_I ) )

    if n1*np.sin(THETA_I)/n2 < 1:
        THETA_R = np.arcsin( n1*np.sin(THETA_I)/n2 )
        ray2.set_data( [O[0], O[0]+np.cos(np.pi/2-THETA_R)*2], [O[1], O[1]-np.sin(np.pi/2-THETA_R)*2] )
        arc2.set_data( *Arc( (0,0), 0.2, 3*np.pi/2, 3*np.pi/2+THETA_R ) )

    ax.text( -0.9, 0.05, f"$n_1={n1:.2f}$", ha="left", va="bottom" )
    ax.text( -0.9, -0.05, f"$n_2={n2:.2f}$", ha="left", va="top" )

    if n1*np.sin(THETA_I)/n2 < 1:
        ax.text( 0.7, 0.9, f"$\\theta_i={ti:.2f}^\circ$\n$\\theta_r={np.rad2deg(THETA_R):.2f}^\circ$")
    else:
        ax.text( 0.7, 0.9, f"$\\theta_i={ti:.2f}^\circ$\n$\\theta_r$ no existe")

    plt.show()


'''
Functions defined for: Rainbow
'''


def Line( P1, P2, s=0.001 ):
    '''
    Coordinates of a line given two points
    '''
    if P1[0]>P2[0]:
        s= -s

    k = ( P2[1]-P1[1] ) / ( P2[0]-P1[0] )
    c = P2[1] - k*P2[0]
    X = np.arange(P1[0],P2[0]+s,s)

    #return list(X), [ c + k*i for i in X ]
    return [ X[0], X[-1] ], [ c + k*i for i in [ X[0], X[-1] ] ]


def RayTracing( h, n ):
    '''
    Path of a lightray in a sphere
    '''
    ### Get critical points and angles
    # Incidence angle
    θi = np.arcsin( h )

    # Refraction angle
    θr = np.arcsin( h/n )

    # Final deviation angle
    θf = 4*θr-2*θi

    # Indidence point
    I = np.array( [ np.cos( np.pi-θi ), np.sin( np.pi-θi ) ] )

    # Internal reflection point
    R = np.array( [ np.cos( 2*θr-θi ), np.sin( 2*θr-θi ) ] )

    # Emerging point
    E = np.array( [ np.cos( 4*θr-θi-np.pi ), np.sin( 4*θr-θi-np.pi ) ] )

    ### Draw rays
    RaysX, RaysY = [], []

    # Incident ray
    x, y = Line( (-2, I[1]), I )
    RaysX += x
    RaysY += y

    # First internal ray
    x, y = Line( I, R )
    RaysX += x
    RaysY += y

    # Second internal ray
    x, y = Line( R, E )
    RaysX += x
    RaysY += y
    
    # Emerging ray
    x, y = Line( E, (-2, E[1]+(-2-E[0])*np.tan(θf) ) )
    RaysX += x
    RaysY += y

    return RaysX, RaysY, [θi,θr,θf,I,R,E]


def RefrIndex(wl):
    '''
    Refractive index in water given the wavelength
    '''
    wl = wl/589

    a = 0.2336091059 + 0.000268678472*wl + 0.0015892057/wl**2 + 0.00245934259/( wl**2-0.229202**2 ) + 0.90070492/( wl**2-5.432937**2 )

    return np.real( np.sqrt( -2*a-1, dtype=complex )/np.sqrt( a-1, dtype=complex ) )


def MaxCaustic( n ):
    '''
    Angular position of the caustic
    '''
    max_inc = np.arcsin( ( ( 4-n**2 )/3 )**0.5 )
    max_ref = np.arcsin( ( ( 4-n**2 )/3 )**0.5 / n )

    return ( 4*max_ref - 2*max_inc ) * 180/np.pi

def MaxH( n ):
    '''
    Heigth at which the caustic appears
    '''
    return ( ( 4-n**2 )/3 )**0.5


def FindTangent( P ):
    '''
    A tangent to a vector
    '''
    return np.array( [ P[1], -P[0] ] )/np.linalg.norm( P )


def Rotation( v, ang, c=np.array([0,0]) ):
    '''
    Apply a rotation around a center
    '''
    v -= c
    R = np.array([[np.cos(ang), np.sin(ang)],[-np.sin(ang), np.cos(ang)]])
    return R @ v


def Droplet( ):
    '''
    Coordinates of a sphere
    '''
    return np.cos( np.arange(0,2*np.pi+0.001,0.001) ), np.sin( np.arange(0,2*np.pi+0.001,0.001) )


def DrawDroplet( fig: mpl.figure, ax: mpl.axes, color: str = "lightsteelblue", alpha: float = 0.2, fill: bool = True, center: bool = True ):
    '''
    Draw a droplet
    '''

    ax.plot( *Droplet(), color=color )
    if fill:
        ax.add_patch( plt.Circle((0, 0), 1, color=color, alpha=alpha, clip_on=False) )

    if center:
        ax.plot(0,0,"ko")


def PlotRaysDroplet( fig: mpl.figure, ax: mpl.axes, h: float, n: float = 1.33, color: str = "red",  draw_droplet: bool = True, alpha_indicent: float = 1,
                     draw_local_axis_I: bool = True, draw_angles_I: bool = True, draw_arrows_I: bool = True, alpha_refracted_I: float = 1, alpha_reflected_I: float = 1,
                     draw_local_axis_R: bool = True, draw_angles_R: bool = True, draw_arrows_R: bool = True, alpha_refracted_R: float = 1, alpha_reflected_R: float = 1,
                     draw_local_axis_E: bool = True, draw_angles_E: bool = True, draw_arrows_E: bool = True, alpha_refracted_E: float = 1, alpha_reflected_E: float = 1  ):
    '''
    Draw the rays in the droplet
    '''
    # Get incidence points and angles
    _, _, [θi,θr,θf,I,R,E] = RayTracing( h, n )

    if draw_droplet:
        DrawDroplet( fig, ax )

    ### At I
    # Incident ray
    if alpha_indicent > 1e-4:
        ax.plot( *Line( (-2, I[1]), I ), "-", color = color, alpha=alpha_indicent )

    # Local axis
    if draw_local_axis_I:
        ax.plot( *Line( (0,0), 2*I ) , "k-", alpha=0.5 )
        ax.plot( *Line( I+0.5*FindTangent(I), I-0.5*FindTangent(I) ), "k-", alpha=0.5 )

    # Refracted
    if alpha_refracted_I > 1e-4:
        ax.plot( *Line( I, R ), "-", color=color, alpha=alpha_refracted_I )
    
    # Relfected
    if alpha_reflected_I > 1e-4:
        ax.plot( *Line( I, I+5*Rotation( np.array([-1,0]), 2*θi ) ), "-", color=color, alpha=alpha_reflected_I )
    
    # Angles
    if draw_angles_I:
        # Incident
        ax.plot( *Arc( I, 0.2, np.pi, np.pi-θi ), color="coral", alpha=alpha_indicent )

        # Refracted
        ax.plot( *Arc( I, 0.2, θr-θi, -θi ), color="royalblue", alpha=alpha_refracted_I )

        # Reflected
        ax.plot( *Arc( I, 0.2, np.pi-2*θi, np.pi-θi ), color="coral", alpha=alpha_reflected_I )
        
    # Arrows
    if draw_arrows_I:
        # Incident
        ax.annotate( "", xy = I-np.array([0.41,0]), xytext=I-np.array([0.4,0]), arrowprops=dict(arrowstyle= '<-',
                    color=color, alpha=alpha_indicent ) )

        # Refracted
        ax.annotate( "", xy = I+0.39*(R-I)/np.linalg.norm(R-I), xytext=I+0.4*(R-I)/np.linalg.norm(R-I), arrowprops=dict(arrowstyle= '<-',
                    color=color, alpha=alpha_refracted_I ) )
        
        # Reflected
        ax.annotate( "", xy = I+0.39*Rotation( np.array([-1,0]), 2*θi ), xytext=I+0.4*Rotation( np.array([-1,0]), 2*θi ), arrowprops=dict(arrowstyle= '<-',
                    color=color, alpha=alpha_reflected_I ) )

    ### At R
    # Local axis
    if draw_local_axis_R:
        ax.plot( *Line( (0,0), 2*R ), "k-", alpha=0.5 )
        ax.plot( *Line( R+0.5*FindTangent(R), R-0.5*FindTangent(R) ), 'k-', alpha=0.5 )

    # Refracted
    if alpha_refracted_R > 1e-4:
        ax.plot( *Line( R, R+5*Rotation( I-R, np.pi-θr+θi ) ), "-", color=color, alpha=alpha_refracted_R )
    
    # Reflected
    if alpha_reflected_R > 1e-4:
        ax.plot( *Line( R, E ), "-", color=color, alpha=alpha_reflected_R )
    
    # Angles
    if draw_angles_R:
        # Incident
        ax.plot( *Arc( R, 0.2, np.pi-θi+θr, np.pi-θi+2*θr  ), color="coral", alpha=alpha_refracted_I )

        # Refracted
        ax.plot( *Arc( R, 0.2, 2*np.pi-θi+2*θr, 2*np.pi-2*θi+2*θr  ), color="royalblue", alpha=alpha_refracted_R )

        # Reflected
        ax.plot( *Arc( R, 0.2, np.pi-θi+2*θr, np.pi-θi+3*θr  ), color="coral", alpha=alpha_reflected_R )
        
    # Arrows
    if draw_arrows_R:
        # Incident
        ax.annotate( "", xy = R-0.41*(R-I)/np.linalg.norm(R-I), xytext=R-0.4*(R-I)/np.linalg.norm(R-I), arrowprops=dict(arrowstyle= '<-',
                    color=color, alpha=alpha_refracted_I ) )
        
        # Refracted
        ax.annotate( "", xy = R+0.39*Rotation( (I-R)/np.linalg.norm(I-R), np.pi-θr+θi ), xytext=R+0.4*Rotation( (I-R)/np.linalg.norm(I-R), np.pi-θr+θi ), arrowprops=dict(arrowstyle= '<-',
                    color=color, alpha=alpha_refracted_R ) )
        
        # Reflected
        ax.annotate( "", xy = R+0.39*(E-R)/np.linalg.norm(E-R), xytext=R+0.4*(E-R)/np.linalg.norm(E-R), arrowprops=dict(arrowstyle= '<-',
                    color=color, alpha=alpha_reflected_R ) )

    ### At E
    # Local axis
    if draw_local_axis_E:
        ax.plot( *Line( (0,0), 2*E ), "k-", alpha=0.5 )
        ax.plot( *Line( E+0.5*FindTangent(E), E-0.5*FindTangent(E) ), 'k-', alpha=0.5 )

    # Refracted
    if alpha_refracted_E > 1e-4:
        ax.plot( *Line( E, (-2, E[1]+(-2-E[0])*np.tan(θf) ) ), "-", color=color, alpha=alpha_refracted_E )
    
    # Reflected
    if alpha_reflected_E > 1e-4:
        ax.plot( *Line( E, E+Rotation( (R-E), 2*np.pi-2*θr ) ), "-", color=color, alpha=alpha_reflected_E  )

    # Angles
    if draw_angles_E:
        # Incident
        ax.plot( *Arc( E, 0.2, 3*θr-θi, 4*θr-θi  ), color="coral", alpha=alpha_reflected_R  )
        
        # Refracted
        ax.plot( *Arc( E, 0.2, np.pi-θi+4*θr, np.pi-2*θi+4*θr  ), color="royalblue", alpha=alpha_refracted_E )

        # Reflected
        ax.plot( *Arc( E, 0.2, 4*θr-θi, 5*θr-θi  ), color="coral", alpha=alpha_reflected_E )

    # Arrows
    if draw_arrows_E:
        # Incident
        ax.annotate( "", xy = E-0.41*(E-R)/np.linalg.norm(E-R), xytext=E-0.4*(E-R)/np.linalg.norm(E-R), arrowprops=dict(arrowstyle= '<-',
                    color=color, alpha=alpha_reflected_R ) )
        
        # Refracted
        ax.annotate( "", xy = E+0.39*Rotation( (R-E)/np.linalg.norm(R-E), np.pi-θr+θi ), xytext=E+0.4*Rotation( (R-E)/np.linalg.norm(R-E), np.pi-θr+θi ), arrowprops=dict(arrowstyle= '<-',
                    color=color, alpha=alpha_refracted_E ) )

        # Reflected
        ax.annotate( "", xy = E+0.39*Rotation( (R-E)/np.linalg.norm(R-E), -2*θr ), xytext=E+0.4*Rotation( (R-E)/np.linalg.norm(R-E), -2*θr ), arrowprops=dict(arrowstyle= '<-',
                    color=color, alpha=alpha_reflected_E ) )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)


def PlotCaustic( alphas ):
    '''
    Plot the formation of the caustic
    '''

    N = 150

    fig, ax = plt.subplots( figsize=(6,6), tight_layout=True )
    DrawDroplet( fig, ax )

    def update(frame):
        PlotRaysDroplet( fig, ax, h = 0.7 + 0.3*frame/N, n = RefrIndex(750), color = "red", draw_droplet = False, alpha_indicent = alphas[0],
                        draw_local_axis_I = False, draw_angles_I = False, draw_arrows_I = False, alpha_refracted_I = alphas[1], alpha_reflected_I = 0,
                        draw_local_axis_R = False, draw_angles_R = False, draw_arrows_R = False, alpha_refracted_R = 0.0, alpha_reflected_R = alphas[2],
                        draw_local_axis_E = False, draw_angles_E = False, draw_arrows_E = False, alpha_refracted_E = alphas[3], alpha_reflected_E = 0  )
        return ax

    plt.close()
    
    return animation.FuncAnimation(fig, func=update, frames=N, interval=70)


def PlotRainbow( alphas: list[float], mode: str ):
    '''
    Plot the formation of the rainbow from a single droplet
    '''

    fig, ax = plt.subplots( figsize=(6,6), tight_layout=True )
    DrawDroplet( fig, ax )

    # Visible spectrum: red at 750 nm and blue at 380 nm (water)
    Spectrum = np.arange(750,380,-10)

    def update(frame):
        color = mpl.colormaps["rainbow_r"](frame/len(Spectrum))
        n = RefrIndex( Spectrum[frame] )

        if mode == "caustic":
            PlotRaysDroplet( fig, ax, h = MaxH(n), n = n, color = color, draw_droplet = False, alpha_indicent = alphas[0],
                            draw_local_axis_I = False, draw_angles_I = False, draw_arrows_I = False, alpha_refracted_I = alphas[1], alpha_reflected_I = 0,
                            draw_local_axis_R = False, draw_angles_R = False, draw_arrows_R = False, alpha_refracted_R = 0.0, alpha_reflected_R = alphas[2],
                            draw_local_axis_E = False, draw_angles_E = False, draw_arrows_E = False, alpha_refracted_E = alphas[3], alpha_reflected_E = 0  )

        else:
            for h in np.arange(0.9,0.999,0.01):
                PlotRaysDroplet( fig, ax, h = h, n = n, color = color, draw_droplet = False, alpha_indicent = alphas[0],
                                draw_local_axis_I = False, draw_angles_I = False, draw_arrows_I = False, alpha_refracted_I = alphas[1], alpha_reflected_I = 0,
                                draw_local_axis_R = False, draw_angles_R = False, draw_arrows_R = False, alpha_refracted_R = 0.0, alpha_reflected_R = alphas[2],
                                draw_local_axis_E = False, draw_angles_E = False, draw_arrows_E = False, alpha_refracted_E = alphas[3], alpha_reflected_E = 0  )

        return ax
    
    plt.close()
    
    return animation.FuncAnimation(fig, func=update, frames=len(Spectrum), interval=70)


def PlotCausticFreq( f: float, alphas: list[float] ):
    '''
    Plot the caustic for a certain frequency
    '''

    fig, ax = plt.subplots( figsize=(6,6), tight_layout=True )
    DrawDroplet( fig, ax )

    wl = C / ( f * 1e12 ) / (1e-9)

    n = RefrIndex( wl )
    color = mpl.colormaps["rainbow_r"]((750-wl)/(750-380))
    
    for h in np.arange(0.75,0.999,0.01):
        PlotRaysDroplet( fig, ax, h = h, n = n, color = color, draw_droplet = False, alpha_indicent = alphas[0],
                        draw_local_axis_I = False, draw_angles_I = False, draw_arrows_I = False, alpha_refracted_I = alphas[1], alpha_reflected_I = 0,
                        draw_local_axis_R = False, draw_angles_R = False, draw_arrows_R = False, alpha_refracted_R = 0.0, alpha_reflected_R = alphas[2],
                        draw_local_axis_E = False, draw_angles_E = False, draw_arrows_E = False, alpha_refracted_E = alphas[3], alpha_reflected_E = 0  )
    
    print( f"La longitud de onda de la luz es: {wl:.2f} nm" ) 
    print( f"La frecuencia de la luz es: {f:.2f} THz" )
    print( f"La posición angular de la cáustica es: {MaxCaustic(n):.2f}º" )

    plt.show()

    

'''
Functions defined for: Lenses
'''


def PlotRaysDroplet_Paraxial( alphas: list[float] = [ 0.1, 0.96, 0.92 ] ):
    '''
    Plot the trajectory of lightrays through a droplet in the paraxial approximation
    '''

    fig, ax = plt.subplots( figsize=(13,6), ncols=2, tight_layout=True )
    N = 100

    for a in ax:
        a.set_aspect("equal")

    # First subplot: static
    DrawDroplet( fig, ax[0] )

    for h in np.arange(0,0.25,0.05):
        PlotRaysDroplet( fig, ax[0], h = h, n = RefrIndex(750), color = "red", draw_droplet = False, alpha_indicent = alphas[0],
                        draw_local_axis_I = False, draw_angles_I = False, alpha_refracted_I = alphas[1], alpha_reflected_I = 0,
                        draw_local_axis_R = False, draw_angles_R = False, alpha_refracted_R = alphas[2], alpha_reflected_R = 0,
                        draw_local_axis_E = False, draw_angles_E = False, alpha_refracted_E = 0, alpha_reflected_E = 0  )

    ax[0].set_aspect("equal")
    ax[0].set_xlim(-2,2.5)
    ax[0].set_ylim(-2.25,2.25) 

    # Second plot: animated
    DrawDroplet( fig, ax[1] )

    def update(frame):
        ax[1].set_aspect("equal")

        PlotRaysDroplet( fig, ax[1], 0.2*frame/N, n = RefrIndex(750), color = "red", draw_droplet = False, alpha_indicent = alphas[0],
                        draw_local_axis_I = False, draw_angles_I = False, draw_arrows_I = False, alpha_refracted_I = alphas[1], alpha_reflected_I = 0,
                        draw_local_axis_R = False, draw_angles_R = False, draw_arrows_R = False, alpha_refracted_R = alphas[2], alpha_reflected_R = 0,
                        draw_local_axis_E = False, draw_angles_E = False, draw_arrows_E = False, alpha_refracted_E = 0, alpha_reflected_E = 0  )
        
        ax[1].set_aspect("equal")
        ax[1].set_xlim(-2,2.5)
        ax[1].set_ylim(-2.25,2.25) 

        return ax

    plt.close()

    return animation.FuncAnimation(fig, func=update, frames=N, interval=70)


def Lens( R1: float, R2: float, c1: float = 0, c2: float = 0, ymax: float = 1 ) -> tuple:
    '''
    Coordiantes of a lens given the radii and centers of the spherical surfaces
    '''

    # Surface S1
    if R1 == np.inf:
        x, y = np.array([c1,c1]), np.array([ymax,-ymax])
    elif R1<0:
        x, y = Arc( (R1+c1,0), abs(R1), np.arcsin(ymax/abs(R1)), -np.arcsin(ymax/abs(R1)) )
    else:
        x, y = Arc( (R1+c1,0), abs(R1), -np.arcsin(ymax/abs(R1))+np.pi, np.arcsin(ymax/abs(R1))+np.pi )

    # Surface S2
    if R2 == np.inf:
        x2, y2 = np.array([c2,c2]), np.array([-ymax,ymax])
    elif R2<0:
        x2, y2 = Arc( (R2+c2,0), abs(R2), -np.arcsin(ymax/abs(R2)), np.arcsin(ymax/abs(R2)) )
    else:
        x2, y2 = Arc( (R2+c2,0), abs(R2), np.arcsin(ymax/abs(R2))+np.pi, -np.arcsin(ymax/abs(R2))+np.pi )

    # Concatenate surfaces and generate closed curve
    x = np.append( x, x2 )
    y = np.append( y, y2 )

    x = np.append( x, [x[0]] )
    y = np.append( y, [y[0]] )

    return x, y


def IntersectionLineCirc( P1: np.ndarray, P2: np.ndarray, c: tuple = (0,0), r: float = 1 ) -> list:
    '''
    Find the intersection(s) between a line (defined by two points) and a circumpherence
    '''

    dx, dy = P2 - P1

    # Auxiliary coefficients for the 2nd degree eq
    a2 = dx*dx + dy*dy
    a1 = 2*( dx*( P1[0] - c[0] ) + dy*( P1[1] - c[1] ) )
    a0 = ( P1[0] - c[0] )**2 + ( P1[1] - c[1] ) **2 - r*r

    discr = a1*a1 - 4*a2*a0

    if discr < 0:
        return [] # No intersection
    
    elif discr == 0:
        t = -a1 / (2*a2)

        return [ ( P1[0]+t*dx, P1[1]+t*dy ) ]  # Tangent point
    
    else:
        sqrt_disc = discr**0.5
        t1 = (-a1 + sqrt_disc) / (2*a2)
        t2 = (-a1 - sqrt_disc) / (2*a2)

        return [ ( P1[0]+t1*dx, P1[1]+t1*dy ), ( P1[0]+t2*dx, P1[1]+t2*dy ) ]


def line_circle_intersection(circle_center, radius, line_point1, line_point2):
    """
    Find intersection points between a line and a circle.
    
    Parameters:
    circle_center: tuple (h, k)
    radius: float r
    line_point1, line_point2: tuples (x, y) defining the line
    
    Returns:
    List of intersection points (empty if none, one point if tangent, two if secant)
    """
    h, k = circle_center
    x1, y1 = line_point1
    x2, y2 = line_point2
    
    dx = x2 - x1
    dy = y2 - y1
    
    a = dx**2 + dy**2
    b = 2 * (dx * (x1 - h) + dy * (y1 - k))
    c = (x1 - h)**2 + (y1 - k)**2 - radius**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return []  # No intersection
    elif discriminant == 0:
        t = -b / (2*a)
        x = x1 + t*dx
        y = y1 + t*dy
        return [(x, y)]  # Tangent point
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2*a)
        t2 = (-b - sqrt_disc) / (2*a)
        
        x1_int = x1 + t1*dx
        y1_int = y1 + t1*dy
        x2_int = x1 + t2*dx
        y2_int = y1 + t2*dy
        
        return [(x1_int, y1_int), (x2_int, y2_int)]
    


def RaysLens( h: float, n: float, R1: float, R2: float, c1: float = 0, c2: float = 0, xlims: tuple = (-2,2) ) -> tuple:
    '''
    The trajetory of a ray of light through a lens
    '''

    with np.errstate(divide='ignore'):

        ### Incident ray contact point with S1
        if R1 == np.inf:
            xi = c1
        else:
            xi = (R1+c1)-R1*np.cos(np.arcsin(h/R1))


        x = np.array( [xlims[0], xi] )
        y = np.array( [ h, h] )

        ### Refraction of the ray entering the lens
        theta_i = np.arcsin(h/R1)
        theta_r = np.arcsin(h/(R1*n))

        # The deflection depends on the orientation of the surface
        theta_n = -np.sign(R1) * (theta_i - theta_r)

        ### Intersection of the refracted ray with S2
        if R2 == np.inf:
            xf, yf = c2, h+np.tan(theta_n)*(c2-xi)
        else:
            # Intersections between the refracted ray and the sphere defining S2
            cut = IntersectionLineCirc( P1=np.array([ x[-1], y[-1] ]), P2=np.array([ x[-1]+np.cos(theta_n), y[-1]+np.sin(theta_n) ]), c=(R2+c2, 0), r=abs(R2) )

            # The ray emerges just after xi
            xf, yf = min( (a, b) for a, b in cut if a > xi )

        x = np.append( x, [ xf ] )
        y = np.append( y, [ yf ] )
        
        ### Orientation of the emergent ray
        theta_if = np.sign(h)*np.sign(R1)*abs(theta_n) - np.sign(R2)*np.arcsin( yf/abs(R2) ) # incident angle with respect to the surface normal at the contact point
        theta_rf = np.arcsin( n * np.sin(theta_if) ) # refracted angle with respect to the surface normal at the contact point
        theta_f = np.sign(-R2)*np.arcsin( yf/abs(R2) ) - theta_rf # deflection angle, difference between the orientation of the normal and the reflected angle

        # Compute the focal distance
        f = xf - yf/np.tan(theta_f)

        x = np.append( x, [ max( f, 1.1*xlims[1] ) if (R1 != np.inf or R2 != np.inf) and n!=1 else 1.1*xlims[1] ] )
        y = np.append( y, [ yf+(x[-1]-x[-2])*np.tan(theta_f) ] )

        if f<0:
            x = np.append( [ 1.1*f ], x )
            y = np.append( [ h ], y )

        # Virtual rays (only relevant for divergent lenses)
        xv = np.array( [ xf, 1.1*f if (R1 != np.inf or R2 != np.inf) and n!=1 else 1.1*xlims[0]  ] )
        yv = np.array( [ yf, yf+(xv[-1]-xf)*np.tan(theta_f) ] )

        return x, y, f, xv, yv


def PlotThinLens( fig: mpl.figure, ax: mpl.axes, h: list, n: float, R1: float, R2: float, c1: float = 0, c2: float = 0, ymax: float = 1, xlims: tuple = (-2,8), color: str = "red", alpha_rays: float = 1, alpha_virtual: float = 0.2, report: bool = False ):
    '''
    Plot a lens and the rays of light in the paraxial approximation
    '''

    ax.set_xlim( *xlims )
    ax.set_ylim( -1.1*ymax, 1.1*ymax )

    h *= ymax

    # Draw the lens
    x, y = Lens( R1, R2, c1 = c1, c2 = c2, ymax = 1  ) 
    ax.plot( x, y, "-", color="lightsteelblue" )
    ax.fill( x, y, "-", color="lightsteelblue", alpha=0.5 )

    for h_ in h:
        x, y, f, xv, yv = RaysLens( h_, n, R1, R2, c1=c1, c2=c2, xlims=xlims )
        ax.plot( x, y, "-", color=color, alpha=alpha_rays )
        ax.plot( xv, yv, ":", color=color, alpha=alpha_virtual )

    ax.plot( f, 0 , "ro" )

    ax.set_xlim( [ min( xlims[0], 1.1*f ), max( xlims[1], 1.1*f ) ] if (R1 != np.inf or R2 != np.inf) and n!=1 else xlims )

    # Draw the axis
    ax.plot( ax.get_xlim(), [0, 0], "--", color="lightgray" )

    with np.errstate(divide='ignore'):

        if report:
            print( f"El radio de la superficie S1 es aproximadamente R1={R1:.2f}")
            print( f"El radio de la superficie S2 es aproximadamente R2={R2:.2f}")

            if n != 1:
                if  (R1 != np.inf or R2 != np.inf):
                    print( f"La lente es {'convergente' if f>0 else 'divergente'}. La distancia focal es aproximadamente f={f-x[-2]:.2f} ({1/( (n-1)*(1/R1-1/R2) ):.2f})" )
                else:
                    print( "Como las dos superficies son planas, la trayectoria de los rayos de luz que inciden perpendicularmente no se vé afectada.")
            else:
                print( f"Dado que n=1, el material tiene las mismas propiedades que el aire y no afecta a la propagación de la luz." )

        plt.show()


def PlotLensImageFocusRays( f: float = 2, ymax: float = 1 ):
    '''
    Draw a schemme with the rays passing through the image focus
    '''

    xlim = (-2*f, 2*f)

    fig, ax = plt.subplots( figsize=(13,5), ncols=2, tight_layout=True )

    ax[0].set_xlim( *xlim )

    ax[0].plot( ax[0].get_xlim(), (0,0), "--", color="lightgray" )
    ax[0].plot( (0,0), (-ymax,ymax), "-", color="lightsteelblue" )
    ax[0].plot( 0, ymax, "^", color="lightsteelblue", markersize=10 )
    ax[0].plot( 0, -ymax, "v", color="lightsteelblue", markersize=10 )
    ax[0].plot( f, 0, "|", color="red", markersize=10 )
    ax[0].text( f, -0.02*abs(ax[0].get_ylim()[1]-ax[0].get_ylim()[0]), "$f'$", color="red", va = "top", ha="center", size=15 )

    for i in [0.5, 0.25, 0, -0.25, -0.5]:
        ax[0].plot( ( xlim[0], 0, xlim[1]), (i, i, i - i/f*xlim[1] ), color="yellow", alpha=0.4 )
        ax[0].annotate( "", xy = ( -3.1*f/2, i ), xytext = ( -3*f/2, i ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[0].annotate( "", xy = ( -1.1*f/2, i ), xytext = ( -f/2, i ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[0].annotate( "", xy = ( f/2, i/f*(f/2) ), xytext = ( 1.1*f/2, i - i/f*(1.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[0].annotate( "", xy = ( 3*f/2, -i + i/f*(f/2) ), xytext = ( 3.1*f/2, -i/f*(1.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

    ax[1].set_xlim(*xlim)

    ax[1].plot( ax[1].get_xlim(), (0,0), "--", color="lightgray" )
    ax[1].plot( (0,0), (-ymax,ymax), "-", color="lightsteelblue" )
    ax[1].plot( 0, ymax, "v", color="lightsteelblue", markersize=10 )
    ax[1].plot( 0, -ymax, "^", color="lightsteelblue", markersize=10 )
    ax[1].plot( -f, 0, "|", color="red", markersize=10 )
    ax[1].text( -f, -0.02*abs(ax[1].get_ylim()[1]-ax[1].get_ylim()[0]), "$f'$", color="red", va = "top", ha="center", size=15 )

    for i in [0.5, 0.25, 0, -0.25, -0.5]:
        ax[1].plot( ( xlim[0], 0, xlim[1]), (i, i, i + i/f*xlim[1] ), color="yellow", alpha=0.4 )
        ax[1].annotate( "", xy = ( -3.1*f/2, i ), xytext = ( -3*f/2, i ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[1].annotate( "", xy = ( -1.1*f/2, i ), xytext = ( -f/2, i ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[1].annotate( "", xy = ( f/2, i+i/f*(f/2) ), xytext = ( 1.1*f/2, i + i/f*(1.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[1].annotate( "", xy = ( 3*f/2, i + i/f*(3*f/2) ), xytext = ( 3.1*f/2, i+i/f*(3.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        
        ax[1].plot( ( -f, 0 ), ( 0, i ), ":", color="yellow", alpha=0.4 )


    plt.show()


def PlotLensCenterRays( f: float = 2, ymax: float = 1 ):
    '''
    Draw a schemme with the rays passing through the center of the lens
    '''

    xlim = (-2*f, 2*f)

    fig, ax = plt.subplots( figsize=(13,5), ncols=2, tight_layout=True )

    ax[0].set_xlim( *xlim )

    ax[0].plot( ax[0].get_xlim(), (0,0), "--", color="lightgray" )
    ax[0].plot( (0,0), (-ymax,ymax), "-", color="lightsteelblue" )
    ax[0].plot( 0, ymax, "^", color="lightsteelblue", markersize=10 )
    ax[0].plot( 0, -ymax, "v", color="lightsteelblue", markersize=10 )
    ax[0].plot( f, 0, "|", color="red", markersize=10 )
    ax[0].text( f, -0.02*abs(ax[0].get_ylim()[1]-ax[0].get_ylim()[0]), "$f'$", color="red", va = "top", ha="center", size=15 )

    for i in [0.5, 0.25, 0, -0.25, -0.5]:
        ax[0].plot( ( xlim[0], 0, xlim[1]), (i, 0, -i ), color="yellow", alpha=0.4 )

        ax[0].annotate( "", xy = ( -3.1*f/2, i/xlim[0]*(-3.1*f/2) ), xytext = ( -3*f/2, i/xlim[0]*(-3*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[0].annotate( "", xy = ( -1.1*f/2, i/xlim[0]*(-1.1*f/2) ), xytext = ( -f/2, i/xlim[0]*(-f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[0].annotate( "", xy = ( f/2, i/xlim[0]*(f/2) ), xytext = ( 1.1*f/2, i/xlim[0]*(1.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[0].annotate( "", xy = ( 3*f/2, i/xlim[0]*(3*f/2) ), xytext = ( 3.1*f/2, i/xlim[0]*(3.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

    ax[1].set_xlim( *xlim )

    ax[1].plot( ax[0].get_xlim(), (0,0), "--", color="lightgray" )
    ax[1].plot( (0,0), (-ymax,ymax), "-", color="lightsteelblue" )
    ax[1].plot( 0, ymax, "v", color="lightsteelblue", markersize=10 )
    ax[1].plot( 0, -ymax, "^", color="lightsteelblue", markersize=10 )
    ax[1].plot( -f, 0, "|", color="red", markersize=10 )
    ax[1].text( -f, -0.02*abs(ax[1].get_ylim()[1]-ax[1].get_ylim()[0]), "$f'$", color="red", va = "top", ha="center", size=15 )

    for i in [0.5, 0.25, 0, -0.25, -0.5]:
        ax[1].plot( ( xlim[0], 0, xlim[1]), (i, 0, -i ), color="yellow", alpha=0.4 )

        ax[1].annotate( "", xy = ( -3.1*f/2, i/xlim[0]*(-3.1*f/2) ), xytext = ( -3*f/2, i/xlim[0]*(-3*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[1].annotate( "", xy = ( -1.1*f/2, i/xlim[0]*(-1.1*f/2) ), xytext = ( -f/2, i/xlim[0]*(-f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[1].annotate( "", xy = ( f/2, i/xlim[0]*(f/2) ), xytext = ( 1.1*f/2, i/xlim[0]*(1.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[1].annotate( "", xy = ( 3*f/2, i/xlim[0]*(3*f/2) ), xytext = ( 3.1*f/2, i/xlim[0]*(3.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

    plt.show()


def PlotLensObjectFocusRays( f: float = 2, ymax: float = 1 ):
    '''
    Draw a schemme with the rays passing through the object focus
    '''

    xlim = (-2*f, 2*f)

    fig, ax = plt.subplots( figsize=(13,5), ncols=2, tight_layout=True )

    ax[0].set_xlim( *xlim )

    ax[0].plot( ax[0].get_xlim(), (0,0), "--", color="lightgray" )
    ax[0].plot( (0,0), (-ymax,ymax), "-", color="lightsteelblue" )
    ax[0].plot( 0, ymax, "^", color="lightsteelblue", markersize=10 )
    ax[0].plot( 0, -ymax, "v", color="lightsteelblue", markersize=10 )
    ax[0].plot( -f, 0, "|", color="red", markersize=10 )
    ax[0].plot( f, 0, "|", color="red", markersize=10 )
    ax[0].text( -f, -0.02*abs(ax[0].get_ylim()[1]-ax[0].get_ylim()[0]), "$f$", color="red", va = "top", ha="center", size=15 )
    ax[0].text( f, -0.02*abs(ax[0].get_ylim()[1]-ax[0].get_ylim()[0]), "$f'$", color="red", va = "top", ha="center", size=15 )

    for i in [0.5, 0.25, 0, -0.25, -0.5]:
        ax[0].plot( ( xlim[0], 0, xlim[1]), (i, i, i - i/f*xlim[1] )[::-1], color="yellow", alpha=0.4 )

        ax[0].annotate( "", xy = ( -3.1*f/2, i - i/f*(3.1*f/2) ), xytext = ( -3*f/2, i - i/f*(3*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[0].annotate( "", xy = ( -1.1*f/2, i - i/f*(1.1*f/2) ), xytext = ( -f/2, i - i/f*(f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        
        ax[0].annotate( "", xy = ( f/2, i ), xytext = (1.1*f/2, i ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[0].annotate( "", xy = ( 3*f/2, i ), xytext = ( 3.1*f/2, i ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )


    ax[1].set_xlim( *xlim )

    ax[1].plot( ax[1].get_xlim(), (0,0), "--", color="lightgray" )
    ax[1].plot( (0,0), (-ymax,ymax), "-", color="lightsteelblue" )
    ax[1].plot( 0, ymax, "^", color="lightsteelblue", markersize=10 )
    ax[1].plot( 0, -ymax, "v", color="lightsteelblue", markersize=10 )
    ax[1].plot( f, 0, "|", color="red", markersize=10 )
    ax[1].plot( -f, 0, "|", color="red", markersize=10 )
    ax[1].text( f, -0.02*abs(ax[1].get_ylim()[1]-ax[1].get_ylim()[0]), "$f$", color="red", va = "top", ha="center", size=15 )
    ax[1].text( -f, -0.02*abs(ax[1].get_ylim()[1]-ax[1].get_ylim()[0]), "$f'$", color="red", va = "top", ha="center", size=15 )

    for i in [0.5, 0.25, 0, -0.25, -0.5]:
        ax[1].plot( ( xlim[0], 0, xlim[1]), ( i - i/f*xlim[0], i, i ), color="yellow", alpha=0.4 )

        ax[1].annotate( "", xy = ( -3.1*f/2, i + i/f*(3.1*f/2) ), xytext = ( -3*f/2, i + i/f*(3*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[1].annotate( "", xy = ( -1.1*f/2, i + i/f*(1.1*f/2) ), xytext = ( -f/2, i + i/f*(f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

        ax[1].annotate( "", xy = ( f/2, i ), xytext = (1.1*f/2, i ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[1].annotate( "", xy = ( 3*f/2, i ), xytext = ( 3.1*f/2, i ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

        ax[1].plot( ( 0, xlim[1]), ( i, i - i/f*xlim[1] ), ":", color="yellow", alpha=0.4 )

        ax[1].annotate( "", xy = ( f/2, i - i/f*(f/2) ), xytext = ( 1.1*f/2, i - i/f*(1.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax[1].annotate( "", xy = ( 3*f/2, i - i/f*(3*f/2) ), xytext = ( 3.1*f/2, i - i/f*(3.1*f/2) ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

    plt.show()



def PlotLensImage( s: float, f: float, y: float, color_object: str = "forestgreen", color_image: str = "yellowgreen", report: bool = False ):
    '''
    Draw the formation of a image by a lens
    '''

    def ImageLens( s: float, f: float ) -> float:
        return np.float64(1)/( 1/f + np.float64(1)/s ) if abs(f+s)>1e-6 else np.inf
    
    fig, ax = plt.subplots( figsize=(8,5), tight_layout=True )

    # Draw object
    ax.plot( (s, s), (0, y), "-", color=color_object )
    ax.plot( s, y, "o", color=color_object )

    # Draw image
    sp = ImageLens( s, f )

    yp = y * sp / s if sp != np.inf else 0

    if sp != np.inf:
        ax.plot( (sp, sp), (0, yp), "-", color=color_image )
        ax.plot( sp, yp, "o", color=color_image )


    # Draw the lenss
    ymax = max( abs(y), abs(yp) )
    ax.plot( (0,0), (-1.2*ymax,1.2*ymax), "-", color="lightsteelblue" )
    ax.plot( 0, 1.2*ymax, "^" if f>0 else "v", color="lightsteelblue", markersize=10 )
    ax.plot( 0, -1.2*ymax, "v" if f>0 else "^", color="lightsteelblue", markersize=10 )
    ax.plot( -f, 0, "|", color="red", markersize=10 )
    ax.plot( f, 0, "|", color="red", markersize=10 )
    ax.text( -f, -0.02*abs(ax.get_ylim()[1]-ax.get_ylim()[0]), "$f$", color="red", va = "top", ha="center", size=15 )
    ax.text( f, -0.02*abs(ax.get_ylim()[1]-ax.get_ylim()[0]), "$f'$", color="red", va = "top", ha="center", size=15 )

    # Set axis
    xlim = ax.get_xlim()
    ax.set_xlim( *xlim )

    # Draw the axis
    ax.plot( xlim, (0,0), "--", color="lightgray" )

    # Draw rays
    if sp != np.inf:
        # Draw focal image ray
        ax.plot( (xlim[0], 0, xlim[1]), (y, y, y+(yp-y)/(sp)*xlim[1]), color="yellow", alpha=0.3 )
        ax.annotate( "", xy = ( 1.01*xlim[0]/2, y ), xytext = ( xlim[0]/2, y ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( xlim[1]/2, y+(yp-y)/(sp)*xlim[1]/2 ) , xytext = ( 1.01*xlim[1]/2, y+(yp-y)/(sp)*1.01*xlim[1]/2 ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

        # Draw center image ray
        ax.plot( (xlim[0], 0, xlim[1]), ( (yp-y)/(sp-s)*xlim[0], 0, (yp-y)/(sp-s)*xlim[1] ), color="yellow", alpha=0.3 )
        ax.annotate( "", xy = ( 1.01*xlim[0]/2, (yp-y)/(sp-s)*1.01*xlim[0]/2 ), xytext = ( xlim[0]/2, (yp-y)/(sp-s)*xlim[0]/2 ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( xlim[1]/2, (yp-y)/(sp-s)*xlim[1]/2 ), xytext = ( 1.01*xlim[1]/2, (yp-y)/(sp-s)*1.01*xlim[1]/2 ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

        # Draw the focal object ray
        ax.plot( (xlim[0], 0, xlim[1]), ( yp+yp/f*xlim[0], yp, yp), color="yellow", alpha=0.3 )
        ax.annotate( "", xy = ( 1.01*xlim[0]/2, yp+yp/f*1.01*xlim[0]/2 ), xytext = ( xlim[0]/2, yp+yp/f*xlim[0]/2 ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( xlim[1]/2, yp ), xytext = ( 1.01*xlim[1]/2, yp ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

        if sp < 0:
            # Project focal image ray
            ax.plot( (xlim[0], 0), (y+(yp-y)/(sp)*xlim[0], y), ":", color="yellow", alpha=0.3 )
            # Project focal object ray
            ax.plot( (xlim[0], 0), (yp, yp), ":", color="yellow", alpha=0.3 )

    else:
        # Draw focal image ray
        ax.plot( (xlim[0], 0, xlim[1]), (y, y, y+(-y)/(f)*xlim[1]), color="yellow", alpha=0.3 )
        # Draw center image ray
        ax.plot( (xlim[0], 0, xlim[1]), ( (-y)/(f)*xlim[0], 0, (-y)/(f)*xlim[1] ), color="yellow", alpha=0.3 )

        # Project focal image ray
        ax.plot( (xlim[0], 0), (y+(-y)/(f)*xlim[0], y), ":", color="yellow", alpha=0.3 )


    ax.set_ylim( -1.4*max( y, abs(yp) ), +1.4*max( y, abs(yp) ) )

    if report:
        print( f"Tipo de lente: {'convergente' if f>0 else 'divergente'}" )
        print( f"Distancia focal: {f:.2f}" )
        print( f"Distancia objeto: {s:.2f}" )
        
        if sp != np.inf:
            print( f"Distancia imagen: {sp:.2f}" )
            print( f"Magnificación: {yp/y:.2f}")
            print( f"La imagen generada es {'real' if sp>0 else 'virtual'}, {'derecha' if yp>0 else 'invertida'} y {'aumentada' if abs(yp)-abs(y)>1e-6 else 'reducida' if abs(yp)-abs(y)<1e-6 else 'igual'}" )
        else:
            print( "Los rayos no convergen, no se forma imagen" )

    plt.show()


'''
Functions defined for: Lenses
'''

def PlotMirrorRays( fig: mpl.figure, ax: mpl.axes, r: float, y, xlim: tuple ):
    '''
    Draw the rays inciding on a mirror
    '''

    #fig, ax = plt.subplots( figsize=(8,5), tight_layout=True )

    # Focus and center of the mirror
    f = r/2
    c = r

    # Draw the mirror
    ymax = abs(y)
    
    if r != np.inf:
        mirror_coords = Arc( (c,0), r=abs(r), ang1=np.arcsin(1.2*ymax/r) if r<0 else np.pi+np.arcsin(1.2*ymax/r), ang2=np.arcsin(-1.2*ymax/r) if r<0 else np.pi+np.arcsin(-1.2*ymax/r) )
    else:
        mirror_coords = (0, 0), (1.2*ymax, -1.2*ymax)

    ax.plot( *mirror_coords, "-", linewidth=8, alpha=0.8, color="lightsteelblue" )

    
    if r != np.inf:
        ax.plot( f, 0, "|", color="red", markersize=10 )
        ax.text( f, -0.02*abs(ax.get_ylim()[1]-ax.get_ylim()[0]), "$f'$", color="red", va = "top", ha="center", size=15 )


    # Set axis
    ax.set_xlim( *xlim )
    ax.set_ylim( -abs(xlim[1]-xlim[0])/2, abs(xlim[1]-xlim[0])/2 )
    ax.set_aspect("equal")

    for h in [ 1, 0.5, 0, -0.5, -1 ]:
        theta = np.arcsin(h*y/r)
        #phi = 2*theta

        # Incident ray
        ax.plot( ( xlim[0], c-r*np.cos(theta) ), ( h*y, h*y ), color="yellow", alpha=0.3  )
        ax.annotate( "", xy = ( 3.01*xlim[0]/2, h*y ), xytext = ( 3*xlim[0]/2, h*y ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( 1.01*xlim[0]/2, h*y ), xytext = ( xlim[0]/2, h*y ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )

        # Reflected ray
        #print( ( xlim[0]/, h*y - h*y/(f-r+r*np.cos(theta))*(xlim[0]/6-r+r*np.cos(theta)) ), )
        ax.plot( ( xlim[0], c-r*np.cos(theta) ), ( h*y - h*y/(f-r+r*np.cos(theta))*(xlim[0]-r+r*np.cos(theta)), h*y ), color="yellow", alpha=0.3   )
        ax.annotate( "", xy = ( xlim[0]/4, h*y - h*y/(f-r+r*np.cos(theta))*(xlim[0]/4-r+r*np.cos(theta)) ), xytext = ( 1.01*xlim[0]/4, h*y - h*y/(f-r+r*np.cos(theta))*(1.01*xlim[0]/4-r+r*np.cos(theta)) ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( 3*xlim[0]/4, h*y - h*y/(f-r+r*np.cos(theta))*(3*xlim[0]/4-r+r*np.cos(theta)) ), xytext = ( 3.01*xlim[0]/4, h*y - h*y/(f-r+r*np.cos(theta))*(3.01*xlim[0]/4-r+r*np.cos(theta)) ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )


        # Projection of the reflected ray
        ax.plot( ( xlim[1], c-r*np.cos(theta) ), ( h*y - h*y/(f-r+r*np.cos(theta))*(xlim[1]-r+r*np.cos(theta)), h*y ), ":", color="yellow", alpha=0.3   )

    
'''
Functions defined for: Mirrors
'''


def PlotMirrorImage( s: float, r: float, y: float = 0.3, color_object: str = "forestgreen", color_image: str = "yellowgreen", report: bool = False ):
    '''
    Draw the formation of a image by a flat mirror
    '''

    def ImageMirror( s: float, f: float ) -> float:
        return np.float64(1)/( 1/f - np.float64(1)/s ) if abs(f-s)>1e-6 else np.inf
    
    fig, ax = plt.subplots( figsize=(8,6), tight_layout=True )

    # Draw object
    ax.plot( (s, s), (0, y), "-", color=color_object )
    ax.plot( s, y, "o", color=color_object )

    # Focus
    f = r/2
    c = r

    # Draw image
    sp = ImageMirror( s, f )
    yp = - y * sp / s

    ax.plot( (sp, sp), (0, yp), "-", color=color_image )
    ax.plot( sp, yp, "o", color=color_image )


    # Draw the mirror
    if r!=np.inf and abs(s-c)>1e-6:
        ymax = max( abs(y), abs(yp) if abs(s-f)>1e-6 else 0, abs(y-y/(r-s)*(c-r*np.cos(np.arcsin(y/r))-s) ) )
    else:
        ymax = max( abs(y), abs(yp) if abs(s-f)>1e-6 else 0 ) 

    if r != np.inf:
        mirror_coords = Arc( (c,0), r=abs(r), ang1=np.arcsin(1.2*ymax/r) if r<0 else np.pi+np.arcsin(1.2*ymax/r), ang2=-np.arcsin(1.2*ymax/r) if r<0 else np.pi-np.arcsin(1.2*ymax/r) )
    else:
        mirror_coords = (0, 0), (1.2*ymax, -1.2*ymax)

    ax.plot( *mirror_coords, "-", linewidth=8, alpha=0.8, color="lightsteelblue" )
    
    if r != np.inf:
        ax.plot( f, 0, "|", color="red", markersize=10 )
        ax.plot( c, 0, ".", color="red", markersize=10 )

        ax.text( f, -0.02*abs(ax.get_ylim()[1]-ax.get_ylim()[0]), "$f'$", color="red", va = "top", ha="center", size=15 )
        ax.text( c, -0.02*abs(ax.get_ylim()[1]-ax.get_ylim()[0]), "$C$", color="red", va = "top", ha="center", size=15 )

    # Set axis
    xlim = ax.get_xlim()
    ax.set_xlim( *xlim )
    

    ax.set_ylim( -1.4*ymax, 1.4*ymax )

    # Draw the axis
    ax.plot( xlim, (0,0), "--", color="lightgray" )

    # Draw rays
    if r != np.inf:
        theta = np.arcsin(y/r)

        # Parallel ray
        cut = IntersectionLineCirc( P1=np.array([ s, y ]), P2=np.array([ 0, y ]), c=(c, 0), r=abs(r) )
        xf_p, yf_p = min( (a, b) for a, b in cut if a > -abs(c) )
        ax.plot( ( xlim[0], xf_p, xlim[0] ), ( y, yf_p, yf_p + (yp-yf_p)/(sp-xf_p)*(xlim[0]-xf_p) ), color="yellow", alpha=0.3 )
        ax.annotate( "", xy = ( 2.01*s/3, y ), xytext = ( 2*s/3, y ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( 1.01*s/3, y ), xytext = ( s/3, y ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( 2*s/3, yf_p + (yp-yf_p)/(sp-xf_p)*(2*s/3-xf_p) ), xytext = ( 2.01*s/3, yf_p + (yp-yf_p)/(sp-xf_p)*(2.01*s/3-xf_p) ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( s/3, yf_p + (yp-yf_p)/(sp-xf_p)*(s/3-xf_p) ), xytext = ( 1.01*s/3, yf_p + (yp-yf_p)/(sp-xf_p)*(1.01*s/3-xf_p) ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )
        
        # Central ray
        ax.plot( ( xlim[0], 0, xlim[0] ) , ( y+y/(s)*(xlim[0]-s), 0, -y-y/(s)*(xlim[0]-s)), color="yellow", alpha=0.3 )
        ax.annotate( "", xy = ( 2.01*s/3, y+y/(s)*(2.01*s/3-s) ), xytext = ( 2*s/3, y+y/(s)*(2*s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( 1.01*s/3, y+y/(s)*(1.01*s/3-s) ), xytext = ( s/3, y+y/(s)*(s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( 2*s/3, -y-y/(s)*(2*s/3-s) ), xytext = ( 2.01*s/3, -y-y/(s)*(2.01*s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( s/3, -y-y/(s)*(s/3-s) ), xytext = ( 1.01*s/3, -y-y/(s)*(1.01*s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                    color="yellow", alpha=0.4 ) )

        # Ray through center of sphere
        if abs(s-c)>1e-6:
            cut = IntersectionLineCirc( P1=np.array([ s, y ]), P2=np.array([ c, 0 ]), c=(c, 0), r=abs(r) )
            xf_c, yf_c = min( (a, b) for a, b in cut if a > -abs(c) )
            ax.plot( ( xlim[0], xf_c ) , ( y-y/(c-s)*(xlim[0]-s), yf_c ), color="yellow", alpha=0.3 )
            ax.annotate( "", xy = ( 2.01*s/3, y-y/(c-s)*(2.01*s/3-s) ), xytext = ( 2*s/3, y-y/(c-s)*(2*s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                        color="yellow", alpha=0.4 ) )
            ax.annotate( "", xy = ( 1.01*s/3, y-y/(c-s)*(1.01*s/3-s) ), xytext = ( s/3, y-y/(c-s)*(s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                        color="yellow", alpha=0.4 ) )
            ax.annotate( "", xy = ( 1.75*s/3, y-y/(c-s)*(1.75*s/3-s) ), xytext = ( 1.76*s/3, y-y/(c-s)*(1.76*s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                        color="yellow", alpha=0.4 ) )
            ax.annotate( "", xy = ( 0.75*s/3, y-y/(c-s)*(0.75*s/3-s) ), xytext = ( 0.76*s/3, y-y/(c-s)*(0.76*s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                        color="yellow", alpha=0.4 ) )


        if abs(s-f)>1e-6:
            # Ray through the focus
            cut = IntersectionLineCirc( P1=np.array([ s, y ]), P2=np.array([ f, 0 ]), c=(c, 0), r=abs(r) )
            xf_f, yf_f = min( (a, b) for a, b in cut if a > -abs(c) )
            ax.plot( ( xlim[0], xf_f, xlim[0] ) , ( y-y/(f-s)*(xlim[0]-s), yf_f, yf_f ), color="yellow", alpha=0.3 )
            ax.annotate( "", xy = ( 2.01*s/3, y-y/(f-s)*(2.01*s/3-s) ), xytext = ( 2*s/3, y-y/(f-s)*(2*s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                        color="yellow", alpha=0.4 ) )
            ax.annotate( "", xy = ( 1.01*s/3, y-y/(f-s)*(1.01*s/3-s) ), xytext = ( s/3, y-y/(f-s)*(s/3-s) ), arrowprops=dict(arrowstyle= '<-',
                        color="yellow", alpha=0.4 ) )
            ax.annotate( "", xy = ( 2*s/3, yf_f ), xytext = ( 2.01*s/3, yf_f ), arrowprops=dict(arrowstyle= '<-',
                        color="yellow", alpha=0.4 ) )
            ax.annotate( "", xy = ( s/3, yf_f ), xytext = ( 1.01*s/3, yf_f ), arrowprops=dict(arrowstyle= '<-',
                        color="yellow", alpha=0.4 ) )

        if sp > 0:
            # Projection of the parallel ray reflected
            ax.plot( ( xf_p, xlim[1] ), ( yf_p, yf_p + (yp-yf_p)/(sp-xf_p)*(xlim[1]-xf_p) ), ":", color="yellow", alpha=0.3 )

            # Projection of the central rayy reflected
            ax.plot( ( 0, xlim[1] ) , ( 0, -y-y/(s)*(xlim[1]-s)), ":", color="yellow", alpha=0.3 )

            if abs(s-c)>1e-6:
                # Projection of the ray trhough the center reflected
                ax.plot( ( xf_c, xlim[1] ) , ( yf_c, y-y/(c-s)*(xlim[1]-s)  ), ":", color="yellow", alpha=0.3 )

            if abs(s-f)>1e-6:
                # Projection of the ray through the focus relfected
                ax.plot( ( xf_f, xlim[1] ) , ( yf_f, yf_f  ), ":", color="yellow", alpha=0.3 )

        pass

    else:
        # Direct ray
        ax.plot( ( xlim[0], 0 ), ( y, y ), color="yellow", alpha=0.3 )
        
        ax.annotate( "", xy = ( 1.11*xlim[0]/2, y ), xytext = ( 1.1*xlim[0]/2, y ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        ax.annotate( "", xy = ( xlim[0]/2, y ), xytext = ( 1.01*xlim[0]/2, y ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )

        # Projection of direct ray
        ax.plot( (0, xlim[1] ), (y,y), ":", color="yellow", alpha=0.3 )

        # Center ray
        ax.plot( ( xlim[0], 0, xlim[0] ), ( y/s*xlim[0], 0, -y/s*xlim[0] ), color="yellow", alpha=0.3 )
        ax.annotate( "", xy = ( 1.11*xlim[0]/2, y/s*1.11*xlim[0]/2 ), xytext = ( 1.1*xlim[0]/2, y/s*1.1*xlim[0]/2 ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )
        

        ax.annotate( "", xy = ( xlim[0]/2, -y/s*xlim[0]/2 ), xytext = ( 1.01*xlim[0]/2, -y/s*1.01*xlim[0]/2 ), arrowprops=dict(arrowstyle= '<-',
                color="yellow", alpha=0.4 ) )


        # Projection of the center ray
        ax.plot( ( 0, xlim[1] ), ( 0, -y/s*xlim[1] ), ":", color="yellow", alpha=0.3 )



    plt.show()


'''
Functions defined for: OpticalBench
'''


def ImageLens( s: float, f: float, tolerance: float = 1e-8 ) -> float:
    assert abs(f) > tolerance
    with np.errstate(divide='ignore'):
        return  np.float64(1.) / (1/f + 1/s)

def ImageMirror( s: float, f: float, tolerance: float = 1e-8 ) -> float:
    assert abs(f) > tolerance
    with np.errstate(divide='ignore'):
        return  np.float64(1.) / (1/f - 1/s)


def RaysLens_Simple( fig: mpl.figure, ax: mpl.axes, s: float, y: float, x: float, f:float ):
    '''
    Draw simple rays for the formation of an image by a lens
    '''

    sp = ImageLens( s, f )
    yp = sp / s * y

    if f>0:
        # Ray 1
        ax.plot( [x+s, x+sp if f>0 else x], [y, yp if f>0 else 0], "-", color="yellow", alpha=0.2 )

        # Ray 2
        ax.plot( [x+s, x,x+sp if f>0 else x+f], [y, y, yp if f>0 else 0], "-", color="yellow", alpha=0.2)

    else:
        # Ray 1
        ax.plot( [x+s, x], [y, 0], "-", color="yellow", alpha=0.2 )

        # Ray 2
        ax.plot( [x+s, x], [y, y], "-", color="yellow", alpha=0.2)
        ax.plot( [x, x+f], [y, 0], ":", color="yellow", alpha=0.2)


def RaysMirror_Simple( fig: mpl.figure, ax: mpl.axes, s: float, y: float, x: float, f:float ):
    '''
    Draw simple rays for the formation of an image by a mirror
    '''

    R = 2*f
    c = x+R

    sp = ImageMirror( s, f )
    yp = -sp / s * y

    if f>0 or abs(s)<abs(f):
        # Ray 1
        cut = IntersectionLineCirc( P1=np.array([ x+s, y ]), P2=np.array([ s, y ]), c=(c, 0), r=abs(R) )
        xf_c, yf_c = min( (a, b) for a, b in cut ) if f>0 else max( (a, b) for a, b in cut )

        ax.plot( [x+s,xf_c], [y,y], "-", color="yellow", alpha=0.2 )
        ax.plot( [xf_c, x+sp], [y,yp], ":", color="yellow", alpha=0.2 )

        # Ray 2
        ax.plot( [x+s,x], [y,0], "-", color="yellow", alpha=0.2 )
        ax.plot( [x,x+sp], [0,yp], ":", color="yellow", alpha=0.2 )

    else:
        # Ray 1
        cut = IntersectionLineCirc( P1=np.array([ x+s, y ]), P2=np.array([ s, y ]), c=(c, 0), r=abs(R) )
        xf_c, yf_c = min( (a, b) for a, b in cut ) if f>0 else max( (a, b) for a, b in cut )

        ax.plot( [x+s,xf_c,x+sp], [y,y,yp], "-", color="yellow", alpha=0.2 )

        # Ray 2
        ax.plot( [x+s,x,x+sp], [y,0,yp], "-", color="yellow", alpha=0.2 )

        

def PlotImageFormation( fig: mpl.figure, ax: mpl.axes, lenses, mirrors, y: float = 1e-3, color_object: str = "forestgreen", color_image: str = "yellowgreen", tolerance: float = 1e-8, report: bool = False ):
    '''
    Draw the formation of an image by a system of lenses and mirrors
    '''

    lenses = [ list(lenses[i]+(i+1,)) for i in range(len(lenses)) ]
    mirrors = [ list(mirrors[i]+(i+1,)) for i in range(len(mirrors)) ]

    # Draw object
    ax.plot( (0, 0), (0, y), "-", color=color_object )
    ax.plot( 0, y, "o", color=color_object )

    # Sort elements and remove anything before object or after image
    mirrors = sorted( ( xf for xf in mirrors if xf[0]>0 ), key=lambda x: x[0] )[:1]
    if len(mirrors):
        lenses = sorted( ( xf for xf in lenses if ( xf[0]>0 and xf[0]<mirrors[0][0] ) ), key=lambda x: x[0] )
    else:
        lenses = sorted( ( xf for xf in lenses if xf[0]>0 ), key=lambda x: x[0] )
    
    obj = [0, y]
    sp = []
    yp = []

    real_image=True

    # Formation of the image by lenses
    for (x, f, i) in lenses:

        # Object image
        s = obj[0]-x

        if s<0:        
            # Image formed
            sp.append( ImageLens( s, f ) )
            yp.append( sp[-1] / s * obj[1] )

            # Current image
            obj[0] = x + sp[-1]
            obj[1] = yp[-1]

            if sp[-1] != np.inf:
                ax.plot( (obj[0], obj[0]), (0, obj[1]), "-", color=color_image, alpha=0.3 )
                ax.plot( *obj, "o", color=color_image, alpha=0.3 )
                RaysLens_Simple( fig, ax, s, ([y]+yp)[-2], x, f )

                real_image = True if f>0 and obj[0]>x else False
        
        else:
            pass

    # Formation of the image by mirrors
    for (x, f, i) in mirrors:
        # Object image
        s = obj[0]-x

        if s<0:
            # Image formed
            sp.append( ImageMirror( s, f ) )
            yp.append( - sp[-1] / s * obj[1] )

            # Current image
            obj[0] = x + sp[-1]
            obj[1] = yp[-1]

            if sp[-1] != np.inf:
                ax.plot( (obj[0], obj[0]), (0, obj[1]), "-", color=color_image, alpha=0.3 )
                ax.plot( *obj, "o", color=color_image, alpha=0.3 )
                RaysMirror_Simple( fig, ax, s, ([y]+yp)[-2], x, f )

                real_image = True if f<0 and s<f else False
        
        else:
            pass

    # Draw final image
    if sp != np.inf:
        ax.plot( (obj[0], obj[0]), (0, obj[1]), "-", color=color_image )
        ax.plot( *obj, "o", color=color_image )

    all_y = np.concat( [ [y], yp ] )
    ymax = max( np.abs(all_y[np.isfinite(all_y)]) )

    # Draw lenses
    for (x, f, i) in lenses:
        ax.plot( (x, x), (-1.2*ymax, 1.2*ymax), "-", color="lightsteelblue" )
        ax.plot( x, 1.2*ymax, "^" if f>0 else "v", color="lightsteelblue", markersize=10 )
        ax.plot( x, -1.2*ymax, "v" if f>0 else "^", color="lightsteelblue", markersize=10 )
        ax.text( x+0.005*abs(ax.get_xlim()[1]-ax.get_xlim()[0]), 1.15*ymax, f"$L{i}$", color="lightsteelblue", va = "top", ha="left", size=10 )

        ax.plot( x+f, 0, "|", color="red", markersize=10 )
        ax.text( x+f, -0.02*abs(ax.get_ylim()[1]-ax.get_ylim()[0]), f"$f'({i})$", color="red", va = "top", ha="center", size=12 )

    # Draw mirrors
    for (x, f, i) in mirrors:

        r = 2*f 

        if r != np.inf:
            mirror_coords = Arc( (x+r,0), r=abs(r), ang1=np.arcsin(1.2*ymax/r) if r<0 else np.pi+np.arcsin(1.2*ymax/r), ang2=-np.arcsin(1.2*ymax/r) if r<0 else np.pi-np.arcsin(1.2*ymax/r) )
        else:
            mirror_coords = (0, 0), (1.2*ymax, -1.2*ymax)

        ax.plot( *mirror_coords, "-", linewidth=8, alpha=0.8, color="lightsteelblue" )
        ax.text( x+0.005*abs(ax.get_xlim()[1]-ax.get_xlim()[0]), 1.15*ymax, f"$M{i}$", color="lightsteelblue", va = "top", ha="left", size=10 )
        
        if r != np.inf:
            ax.plot( x+f, 0, "|", color="red", markersize=10 )
            ax.plot( x+r, 0, ".", color="red", markersize=10 )

            ax.text( x+f, -0.02*abs(ax.get_ylim()[1]-ax.get_ylim()[0]), f"$f'({i})$", color="red", va = "top", ha="center", size=15 )
            ax.text( x+r, -0.02*abs(ax.get_ylim()[1]-ax.get_ylim()[0]), f"$C{i}$", color="red", va = "top", ha="center", size=15 )

    # Set axis
    xlim = ax.get_xlim()
    ax.set_xlim( *xlim )

    # Draw the axis
    ax.plot( xlim, (0,0), "--", color="lightgray" )

    if report:
        print( f"El sistema está formado por {len(lenses+mirrors)} elementos ópticos" )
        print( f"Número de lentes: {len(lenses)} ({len([i for i in lenses if i[1]>0])} convergentes y {len([i for i in lenses if i[1]<0])} divergentes)" )
        print( f"Número de espejos: {len(mirrors)}\n" )
        if np.max(all_y)==np.inf:
            print("Este sistema no forma una imagen")
        else:
            print( f"La posición final de la imagen es {obj[0]:.2f}") 
            print( f"La magnificación del sistema es M={obj[1]/y:.2f}")
            print("\nLa imagen final es:" + f" {'Real' if real_image==True else 'Virtual'}" + f", {'Derecha' if obj[1]*y>0 else 'Invertida'}" + f", {'Aumentada' if np.abs(obj[1])-np.abs(y)>tolerance else 'Reducida' if np.abs(obj[1])-np.abs(y)<-tolerance else 'Igual'} ")














def DrawRaysLens( fig: mpl.figure, ax:mpl.axes, h: float, color: str, n: float, R1: float, R2: float, c1: float = 0, c2: float = 0, ymax: float = 1 ) -> float:

    min_x, max_x = ax.get_xlim()

    h *= ymax

    ### Incident ray contact point with S1
    if R1 == np.inf:
        xi, yi = c1, h
    else:
        xi, yi = (R1+c1)-R1*np.cos(np.arcsin(h/R1)), h

    # Incident ray
    ax.plot( *Line( (min_x-10, h), (xi, h) ), "-", color=color )

    ### Refraction of the ray entering the lens
    theta_i = np.arcsin(h/R1)
    theta_r = np.arcsin(h/(R1*n))

    # The deflection depends on the orientation of the surface
    theta_n = -np.sign(R1) * (theta_i - theta_r)

    ### Intersection of the refracted ray with S2
    if R2 == np.inf:
        xf, yf = c2, yi+np.tan(theta_n)*(c2-xi)
    else:
        cut = line_circle_intersection( (R2+c2, 0), abs(R2), (xi, yi), (xi+np.cos(theta_n), yi+np.sin(theta_n)))

        # The ray emerges just after xi
        xf, yf = min( (a, b) for a, b in cut if a > xi )

    # Refracted ray
    ax.plot( *Line( (xi, h), (xf, yf) ), "-", color=color )

    ### Orientation of the emergent ray
    theta_if = np.sign(R1)*abs(theta_n) - np.sign(R2)*np.arcsin( yf/abs(R2) ) # incident angle with respect to the surface normal at the contact point
    theta_rf = np.arcsin( n * np.sin(theta_if) ) # refracted angle with respect to the surface normal at the contact point
    theta_f = np.sign(-R2)*np.arcsin( yf/abs(R2) ) - theta_rf # deflection angle, difference between the orientation of the normal and the reflected angle

    ax.plot( *Line( (xf, yf), (max_x, yf+(max_x-xf)*np.tan(theta_f))) , "-", color=color )
    
    print( xi, xf, max_x )
    print( yi, yf, yf+(max_x-xf)*np.tan(theta_f))

    ### The focus is the point at which the emergent ray intersects the optical axis
    f = xf - yf/np.tan(theta_f)

    # If the focal distance is negative, the lens is divergent. In such case, virtual rays are added to show the intersection with the axis
    if f<0:
        ax.plot( *Line( (min_x, yf+(min_x-xf)*np.tan(theta_f)), (xf, yf)) , ":", color=color )

    return f