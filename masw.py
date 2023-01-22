import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pylab import rcParams
import cmath

class masw:
    def __init__(self, filename, dt, fs, N, dx, x1, direction, header=6):
        self.filename = filename
        self.dt       = dt
        self.header   = header
        self.data     = pd.read_csv(filename, header=self.header, delimiter="\t", skip_blank_lines=False)
        self.fs       = fs #Frecuencia de muestreo
        self.N        = N #Número de canales
        self.dx       = dx #Espacio entre geófonos
        self.x1       = x1 #Fuente compensada
        self.direction = direction #Forward o backward
        self.Lu        = self.data.shape[0]
        self.Tmax      = self.Lu/fs - 1.0/fs
        self.T         = np.linspace(0,self.Tmax, self.data.shape[0])
        self.L         = (N-1)*dx
        self.x         = np.arange(start=x1, stop=self.L+x1+dx,step=dx)
        self.A         = None
        self.Aplot     = None
        self.fplot     = None
        self.cplot     = None
        self.cT        = None
        self.f         = None
        self.LcT       = None
        self.f_curve0 = None
        self.c_curve0 = None
        self.lambda_curve0 = None
        self.f_curve0_up = None
        self.c_curve0_up = None
        self.lambda_curve0_up = None
        self.f_curve0_low = None
        self.c_curve0_low = None
        self.lambda_curve0_low = None

    def plot(self, scale=0.5):
        fig, ax = plt.subplots(1,1, figsize = (8,10))
        offset  = self.x1
        for col in self.data:
            y      = scale*self.data[col]/self.data[col].abs().max()+offset
            x      = self.T

            ax.plot(x,y,'k')
            ax.fill_between(x,y,y2=offset, where=(y>offset),color='k')
            ax.set_xlabel('Tiempo [s]')
            ax.set_ylabel('Distancia [m]')
            offset = offset + self.dx

        plt.grid()
        fig.tight_layout()
        return fig, ax

    def dispersion_imaging(self, cT_min, cT_max, delta_cT):
        #Conversión de la frecuencia de medición de Hz a rad/seg
        omega_fs = 2*np.pi*self.fs 
        #Matrices vacías con líneas de self.data
        U        = np.zeros_like(self.data)
        P        = np.zeros_like(self.data)
        #Unorm    = np.zeros_like(self.data)
        #Unp   = self.data.to_numpy()
        
        #Aplique la transformada discreta de Fourier al eje de tiempo de u (se usa la función np.fft.ff())
        U     = np.fft.fft(self.data, axis = 0 )
        
        #Normalizar U en dominios de frecuencia y desplazamiento
        #Calcule el espectro de fase de U
        #Unorm = U/abs(U)
        P     = np.exp(-1j*np.angle(U))

        #Rango de frecuencia para U
        omega = np.arange(self.Lu)*omega_fs/self.Lu
        
        self.cT    = np.arange(cT_min, cT_max+delta_cT, delta_cT)
        self.LcT   = len(self.cT)
        self.f     = omega/(2*np.pi)
        self.A     = np.zeros((self.Lu, self.LcT))

        for m, omega_test in enumerate(omega):
            for n, c_phase in enumerate(self.cT):
                delta = omega_test/c_phase
                temp = 0
                for l, x_test in enumerate(self.x):
                    temp = temp + np.exp(-1j*delta*x_test)*P[m,l]
                self.A[m,n] = np.abs(temp)/self.N
        return None

    def plot_dispersion_image_2D(self, fmin, fmax, resolution, FigWidth=8, FigHeight=8, FigFontsize=12):
        no_fmin    = np.argmax(self.f >= fmin)
        no_fmax    = np.argmax(self.f >= fmax)
        
        self.Aplot = self.A[no_fmin + 1 : no_fmax, :]        
        self.fplot      = self.f[no_fmin + 1 : no_fmax]
        self.cplot      = self.cT

        fig, ax = plt.subplots(1,1, figsize = (FigWidth,FigHeight))

        cntr1 = ax.contourf(self.fplot, self.cplot, self.Aplot.T, levels=resolution, cmap="RdBu_r")
        ax.set_xlabel('Frecuencia [Hz]')
        ax.set_ylabel('Velocidad de Fase [m/s]')
        fig.colorbar(cntr1, ax=ax)
        return fig, ax
    
    def plot_dispersion_image_3D(self, fmin, fmax, FigWidth=8, FigHeight=8, FigFontSize=12):
        X, Y = np.meshgrid(self.fplot, self.cplot)
        
        #Grafica la imagen de dispersión 3D
        fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
        cntr1 = ax2.plot_surface(X, Y, self.Aplot.T, rstride=1, cstride=1, cmap='jet', edgecolor='none')
        ax2.set_xlabel('Frecuencia [Hz]')
        ax2.set_ylabel('Velocidad de Fase [m/s]')
        ax2.set_zlabel('Amplitud Normalizada')
        fig2.colorbar(cntr1, ax=ax2)
        return fig2, ax2

    def extract_dispersion_curve(self, f_receivers, select, up_low_boundary, p, resolution=100, FigWidth=8, FigHeight=8):
        Aabsnorm2 = np.zeros_like(self.Aplot)
        Aabsnorm2 = (self.Aplot.T/self.Aplot.max(axis=1)).T
        c_loc, f_loc = np.where(Aabsnorm2.T == 1)
        
        Amax_fvec = np.zeros(len(f_loc))
        Amax_cvec = np.zeros(len(f_loc))
        for i in range(len(f_loc)):
            Amax_fvec[i] = self.fplot[f_loc[i]]
            Amax_cvec[i] = self.cplot[c_loc[i]]
            
        ii = np.where(Amax_fvec >= f_receivers)
        Amax_fvec = Amax_fvec[ii]
        Amax_cvec = Amax_cvec[ii]
        
        #ordenar puntos
        Amax_fvec_sort = np.sort(Amax_fvec)
        I = np.argsort(Amax_fvec)
        Amax_cvec_sort = Amax_cvec[I]
        
        fig, ax = plt.subplots(1,1, figsize = (10, 10))
        cntr1 = ax.contourf(self.fplot, self.cplot, self.Aplot.T, levels=resolution, cmap="RdBu_r")
        ax.set_xlabel('Frecuencia [Hz]')
        ax.set_ylabel('Velocidad de Fase [m/s]')
        fig.colorbar(cntr1, ax=ax)
        
        #Grafica los máximos encima de la imagen de dispersión
        ax.plot(Amax_fvec_sort, Amax_cvec_sort.T,'o', markersize = 4, markerfacecolor = 'k', color = 'k')
        ax.set_xlabel('Frecuencia [Hz]')
        ax.set_ylabel('Velocidad de Fase [m/s]')
        
        
        if up_low_boundary == 'yes':
            #Límites superior / inferior para la curva de dispersión del modo fundamental
            c_loc_p, f_loc_p = np.where(Aabsnorm2.T > p/100)
                        
            Amax_fvec_p = np.zeros(len(f_loc_p))
            Amax_cvec_p = np.zeros(len(f_loc_p))
            
            Amax_fvec_p = self.fplot[f_loc_p]
            Amax_cvec_p = self.cplot[c_loc_p]
            
            for i in range(len(f_loc_p)):
                Amax_fvec_p[i] = self.fplot[f_loc_p[i]]
                Amax_cvec_p[i] = self.cplot[c_loc_p[i]]
                
            ii = np.where(Amax_fvec_p >= f_receivers)
            Amax_fvec_p = Amax_fvec_p[ii]
            Amax_cvec_p = Amax_cvec_p[ii]
            
            #ordenar puntos
            Amax_fvec_sort_p = np.sort(Amax_fvec_p)
            I = np.argsort(Amax_fvec_p, axis =0, kind = 'mergesort')
            Amax_cvec_sort_p = Amax_cvec_p[I]
            
            Amax_fvec_sort_p_cell = {}
            Amax_cvec_sort_p_cell = {}
            f_curve0_up_temp = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))
            c_curve0_up_temp = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))
            f_curve0_low_temp = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))
            c_curve0_low_temp = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))

            U = np.unique(Amax_fvec_sort_p)
                      
            for i in range(len(U)):
                Amax_fvec_sort_p_cell[i] = Amax_fvec_sort_p[np.where(Amax_fvec_sort_p == U[i])]
                Amax_cvec_sort_p_cell[i] = Amax_cvec_sort_p[np.where(Amax_fvec_sort_p == U[i])]
                f_curve0_up_temp[i][0] = Amax_fvec_sort_p_cell[i][-1]
                c_curve0_up_temp[i][0] = Amax_cvec_sort_p_cell[i][-1]
                f_curve0_low_temp[i][0] = Amax_fvec_sort_p_cell[i][1]
                c_curve0_low_temp[i][0] = Amax_cvec_sort_p_cell[i][1]
            
            #Graficar los máximos encima de la imagen de dispersión
            ax.plot(Amax_fvec_sort_p, Amax_cvec_sort_p,'o',markersize = 1, markerfacecolor = 'k', color='k')
            ax.plot(f_curve0_up_temp,c_curve0_up_temp,'o',markersize = 4, markerfacecolor = 'k', color='k')
            ax.plot(f_curve0_low_temp,c_curve0_low_temp,'o',markersize = 4, markerfacecolor = 'k', color='k')
            ax.set_xlim(1, 50)
            ax.set_ylim(50, 400)
            
            if (select == 'numbers') or (select == 'both'):
                
                #Curva de dispersión del modo fundamental
                nP_start = int(input('Inicio de curva de dispersión del modo funamental: ')) -1
                nP_end = int(input('Fin de curva de dispersión del modo funamental: ')) 
                f_curve0 = Amax_fvec_sort[nP_start:nP_end]
                c_curve0 = Amax_cvec_sort[nP_start:nP_end]
                
                if up_low_boundary == 'yes':
                    f_curve0_up = f_curve0_up_temp[nP_start:nP_end]
                    c_curve0_up = c_curve0_up_temp[nP_start:nP_end]
                    f_curve0_low = f_curve0_low_temp[nP_start:nP_end]
                    c_curve0_low = c_curve0_low_temp[nP_start:nP_end]
                
                if select == 'both':
                    print ('Seleccione puntos adicionales para la curva de dispersión del modo fundamental. Presione el botón de en medio para dejar de seleccionar puntos.')
                    f_curve0_add, c_curve0_add = plt.ginputs()
                    f_curve0_temp = f_curve0 + f_curve0_add
                    c_curve0_temp = c_curve0 + c_curve0_add
                    
                    f_curve0 = f_curve0_temp.copy()
                    I = f_curve0_temp.argsort()
                    f_curve0.sort()     
                    c_curve0 = c_curve0_temp[I]
                    
                    if up_low_boundary == 'yes':
                        print ('Seleccione puntos adicionales para la curva de dispersión del límite superior. Presione el botón de en medio para dejar de seleccionar puntos.')
                        f_curve0_up_add, c_curve0_up_add = plt.ginputs()
                        f_curve0_up_temp = f_curve0_up + f_curve0_up_add
                        c_curve0_up_temp = c_curve0_up + c_curve0_up_add
                    
                        f_curve0_up = f_curve0_up_temp.copy()
                        I = f_curve0_up_temp.argsort()
                        f_curve0_up.sort()     
                        c_curve0_up = c_curve0_up_temp[I]
                        
                        print ('Seleccione puntos adicionales para la curva de dispersión del límite inferior. Presione el botón de en medio para dejar de seleccionar puntos.')
                        f_curve0_low_add, c_curve0_low_add = plt.ginputs()
                        f_curve0_low_temp = f_curve0_low + f_curve0_low_add
                        c_curve0_up_temp = c_curve0_low + c_curve0_low_add
                    
                        f_curve0_low = f_curve0_low_temp.copy()
                        I = f_curve0_low_temp.argsort()
                        f_curve0_low.sort()     
                        c_curve0_low = c_curve0_low_temp[I]
                        
                lambda_curve0 = c_curve0 / f_curve0
                
                if up_low_boundary == 'yes':
                    lambda_curve0_up = c_curve0_up / f_curve0_up
                    lambda_curve0_low = c_curve0_low / f_curve0_low
            
        if select == 'mouse':
            #Curva de dispersión del modo fundamental
            print ('Seleccione la curva de dispersión del modo fundamental. Presione el botón de en medio para dejar de seleccionar puntos.')
            f_curve0, c_curve0 = plt.ginputs()
            lambda_curve0 = c_curve0 / f_curve0
               
            if up_low_boundary == 'yes':
                print ('Seleccione puntos adicionales para la curva de dispersión del límite superior. Presione el botón de en medio para dejar de seleccionar puntos.')
                f_curve0_up, c_curve0_up = plt.ginputs()
                lambda_curve0_up = c_curve0_up / f_curve0_up
                   
                print ('Seleccione puntos adicionales para la curva de dispersión del límite inferior. Presione el botón de en medio para dejar de seleccionar puntos.')
                f_curve0_low, c_curve0_low = plt.ginputs()
                lambda_curve0_low = c_curve0_low / f_curve0_low
                   
        if up_low_boundary == 'no':
            f_curve0_up = []
            c_curve0_up = []
            lambda_curve0_up = []
            f_curve0_low = []
            c_curve0_low = []
            lambda_curve0_low = []
        
        self.f_curve0 = f_curve0
        self.c_curve0 = c_curve0
        self.lambda_curve0 = lambda_curve0
        self.f_curve0_up = f_curve0_up
        self.c_curve0_up = c_curve0_up
        self.lambda_curve0_up = lambda_curve0_up
        self.f_curve0_low = f_curve0_low
        self.c_curve0_low = c_curve0_low
        self.lambda_curve0_low = lambda_curve0_low
        
        return c_curve0
         
    def plot_dispersion_curve(self, type, up_low_boundaries, FigWidth, FigHeight, FigFontSize):
        fig, ax = plt.subplots()
        #Frecuencia frente a velocidad de fase de onda de Rayleigh
        if type == 'f_c':
    
            #Con límites superiores / inferiores
            if up_low_boundaries == 'yes':
                line, = ax.plot(self.f_curve0, self.c_curve0,'ko-',markersize = 3, markerfacecolor = 'k')
                line.set_label('Exp.')
                line2, = ax.plot(self.f_curve0_up, self.c_curve0_up,'r+--',markersize = 3, markerfacecolor = 'r')
                line2.set_label('Exp. arriba/abajo')
                line3, = ax.plot(self.f_curve0_low, self.c_curve0_low,'r+--',markersize = 3, markerfacecolor = 'r')
                line3.set_label('Exp. arriba/abajo')
                ax.legend(loc='upper right', fontsize = FigFontSize)
    
            #Sin límites superiores / inferiores
            ax.plot(self.f_curve0, self.c_curve0,'ko-',markersize = 3, markerfacecolor = 'k')
            if up_low_boundaries == 'no':
                ax.legend(['Exp.'], loc='upper right', fontsize = FigFontSize)
    
            #Etiquetas de eje y límites de eje
            ax.grid()
            ax.set_xlabel('Frecuencia [Hz]', fontsize = FigFontSize)
            ax.set_ylabel('Velocidad de onda de Rayleigh [m/s]', fontsize = FigFontSize)
    
            #Tamaño de la figura
            rcParams['figure.figsize'] = 2, 2
            fig.set_figheight(FigHeight)
            fig.set_figwidth(FigWidth)
            
        #Velocidad de fase de onda de Rayleigh frente a longitud de onda                     
        if type == 'c_lambda':
    
            #Con límites superiores / inferiores
            if up_low_boundaries == 'yes':
                line4, = ax.plot(self.c_curve0, self.lambda_curve0.T,'ko-',markersize = 3, markerfacecolor = 'k')
                line4.set_label('Exp.')                
                line5, = ax.plot(self.c_curve0_up, self.lambda_curve0_up,'r+--',markersize = 3, markerfacecolor = 'r')
                line5.set_label('Exp. arriba/abajo')
                line6, = ax.plot(self.c_curve0_low, self.lambda_curve0_low,'r+--',markersize = 3, markerfacecolor = 'r')
                line6.set_label('Exp. arriba/abajo')
                ax.legend(loc='lower left', fontsize = FigFontSize)
    
            #Sin límites superiores / inferiores
            ax.plot(self.c_curve0, self.lambda_curve0.T,'ko-', markersize = 3, markerfacecolor = 'k')
            if up_low_boundaries == 'no':
                ax.legend(['Exp.'], loc='lower left', fontsize = FigFontSize)
    
            #Etiquetas de eje y límites de eje
            #set(gca, 'FontSize', FigFontSize)
            ax.grid()
            ax.set_xlabel('Velocidad de onda de Rayleigh [m/s]', fontsize = FigFontSize)
            ax.set_ylabel('Longitud de onda [m]', fontsize = FigFontSize)
    
            #Tamaño de la figura
            rcParams['figure.figsize'] = 2, 2
            fig.set_figheight(FigHeight)
            fig.set_figwidth(FigWidth)
            plt.gca().invert_yaxis()
        
#PARTE 2: INVERSIÓN

    def Ke_layer(self, h, alpha, beta, rho, c_test, k):
        r = cmath.sqrt(1 - c_test**2 / alpha**2)
        s = cmath.sqrt(1 - c_test**2 / beta**2)
        
        Cr = np.cosh(k*r*h)
        Sr = np.sinh(k*r*h)
        Cs = np.cosh(k*s*h)
        Ss = np.sinh(k*s*h)
        D = 2 * (1 - Cr * Cs) + (1 / (r * s) + r * s) * Sr * Ss
        
        k11_e = (k*rho*c_test**2)/D * (s**(-1)*Cr*Ss - r*Sr*Cs)
        k12_e = (k*rho*c_test**2)/D * (Cr*Cs - r*s*Sr*Ss - 1) - k*rho*beta**2*(1+s**2)
        k13_e = (k*rho*c_test**2)/D * (r*Sr - s**(-1)*Ss)
        k14_e = (k*rho*c_test**2)/D * (-Cr + Cs)
        k21_e = k12_e
        k22_e = (k*rho*c_test**2)/D * (r**(-1)*Sr*Cs - s*Cr*Ss)
        k23_e = -k14_e
        k24_e = (k*rho*c_test**2)/D * (-r**(-1)*Sr + s*Ss)
        k31_e = k13_e
        k32_e = k23_e
        k33_e = k11_e
        k34_e = -k12_e
        k41_e = k14_e
        k42_e = k24_e
        k43_e = -k21_e
        k44_e = k22_e
        
        Ke = np.real(np.array([[k11_e, k12_e, k13_e, k14_e], 
                       [k21_e, k22_e, k23_e, k24_e],
                       [k31_e, k32_e, k33_e, k34_e],
                       [k41_e, k42_e, k43_e, k44_e]]))
        
        Ke = np.where(np.isnan(Ke), np.nan, Ke)
        return Ke
    
    def Ke_halfspace(self, alpha,beta,rho,c_test,k):

        r = cmath.sqrt(1 - c_test**2 / alpha**2)
        s = cmath.sqrt(1 - c_test**2 / beta**2)

        k_11 = k*rho*beta**2*(r*(1 - s**2)) / (1 - r*s)
        k_12 = k*rho*beta**2*(1 - s**2) / (1 - r*s) - 2*k*rho*beta**2
        k_21 = k_12
        k_22 = k*rho*beta**2*(s*(1 - s**2)) / (1  -r*s)

        Ke_halfspace = np.real(np.array([[k_11, k_12], [k_21, k_22]]))
        Ke_halfspace = np.where(np.isnan(Ke_halfspace), np.nan, Ke_halfspace)
        
        return Ke_halfspace
                    
    def stiffness_matrix(self, c_test, k, h, alpha, beta, rho, n):
        #Matriz de rigidez del sistema
        K = np.zeros((2*(n+1), 2*(n+1)))#14x14
        
        #Compruebe si la velocidad de la fase de prueba es igual a la velocidad de la onda de corte 
        #o la velocidad de la onda de compresión de una de las capas
        epsilon = 0.0001
        
        while any(abs(c_test - beta) < epsilon) or any(abs(c_test - alpha) < epsilon):
            c_test = c_test * (1 - epsilon)
        
        #Capas de espesor finito j = 1, ..., n
        for j in range (n):
            #Calcular la matriz de rigidez del elemento para la capa j
            Ke = self.Ke_layer(h[j], alpha[j], beta[j], rho[j], c_test, k)#4x4
            #Agregar a la matriz de rigidez del sistema
            DOFS = slice(2*(j+1)-2, 2*(j+1)+2)
            K[DOFS][:, DOFS] += Ke
            
        #Medio espacio
        #Calcular la matriz de rigidez del elemento para la mitad del espacio
        Ke_halfspace = self.Ke_halfspace(alpha[-1], beta[-1], rho[-1], c_test, k)

        #Agregar a la matriz de rigidez del sistema
        DOFS = slice(2*(n+1)-2, 2*(n+1))
        K[DOFS][:, DOFS] += Ke_halfspace
        
        #Evaluar determinante de la matriz de rigidez del sistema
        D = np.real(np.linalg.det(K))
        return D
    
    def theoretical_dispersion_curve(self, c_test, h, alpha, beta, rho, n):
        k = (2*np.pi) / self.lambda_curve0
        D = np.zeros((len(k), len(c_test)))
        c_t = np.zeros((len(k), 1))
        
        sign_old = np.array(np.nan)
        signD = np.array(np.nan)
        lambda_t = np.zeros((len(k), 1))
        for l in range(len(k)):
            for m in range(len(c_test)):
                D[l][m] = self.stiffness_matrix(c_test[m], k[l], h, alpha, beta, rho, n)
                if m == 0:
                    sign_old = np.sign(D[l][m])
                else:
                    sign_old = signD
                signD = np.sign(D[l][m])
                if sign_old * signD == -1:
                    c_t[l] = c_test[m]
                    lambda_t[l] = 2*np.pi / k[l]
                    break
                
        return c_t, lambda_t
    
    
    def plot_theor_exp_dispersion_curves(self, c_t, lambda_t, up_low_boundaries, FigWidth, FigHeight, FigFontSize):
        fig, ax = plt.subplots()
        
    #Con límites superiores / inferiores
        if up_low_boundaries == 'yes':
            obs, = ax.plot(self.c_curve0,self.lambda_curve0,'ko-', markersize = 3, markerfacecolor = 'k')
            obs.set_label('Exp.')
            obs_up, = ax.plot(self.c_curve0_up,self.lambda_curve0_up,'k+--', markersize = 3, markerfacecolor = 'k')
            obs_up.set_label('Exp. arriba/abajo')
            calc, = ax.plot(c_t, lambda_t,'r+--', markersize = 10, markerfacecolor = 'r')
            calc.set_label('Theor.')
            obs_low, = ax.plot(self.c_curve0_low,self.lambda_curve0_low,'k+--', markersize = 3, markerfacecolor = 'k')
            obs_up.set_label('Exp. arriba/abajo')
            ax.legend(loc = 'lower left', fontsize = FigFontSize)

    #Sin límites superiores / inferiores
        if up_low_boundaries == 'no':
            obs, = ax.plot(self.c_curve0,self.lambda_curve0, 'ko-', markersize = 3, markerfacecolor = 'k')
            obs.set_label('Exp.')
            calc, = ax.plot(c_t, lambda_t,'r+--', markersize = 10, markerfacecolor = 'r')
            calc.set_label('Theor.')
            ax.legend(loc = 'lower left', fontsize = FigFontSize)

    #Etiquetas de eje y límites de eje
        ax.set_xlabel('Velocidad de la onda de Rayleigh [m/s]', fontsize = FigFontSize, fontweight = 'normal')
        ax.set_ylabel('Longitud de onda [m]', fontsize = FigFontSize, fontweight = 'normal')
        

    #Tamaño de la figura
        ax.grid()
        rcParams['figure.figsize'] = 2, 2
        fig.set_figheight(FigHeight)
        fig.set_figwidth(FigWidth)
        plt.gca().invert_yaxis()
        
    def modelo_de_velocidades(self, n, h, beta, FigWidth, FigHeight, FigFontSize):
        #Calcule el vector de profundidad z
        z = np.zeros(n+1);
        for i in range(n):
            z[i+1] = np.sum(h[0:i])

        #Graficar el perfil de velocidad de la onda de corte
        fig, ax = plt.subplots()
        for i in range(n):
            ax.plot(np.array([beta[i], beta[i]]), np.array([z[i], z[i+1]]), 'k', markersize = 3, markerfacecolor = 'k')
            ax.plot(np.array([beta[i], beta[i+1]]), np.array([z[i+1], z[i+1]]), 'k', markersize = 3, markerfacecolor = 'k')
        ax.plot(np.array([beta[n], beta[n]]), np.array([z[n], z[n]+5]), 'k', markersize = 3, markerfacecolor = 'k')
        #Establecer los ejes y los límites de los ejes
        ax.set_xlabel('Velocidad de onda de corte [m/s]', fontsize = FigFontSize)
        ax.set_ylabel('Espesor [m]', fontsize = FigFontSize)
        
        #Tamaño de la figura
        ax.grid()
        rcParams['figure.figsize'] = 2, 2
        fig.set_figheight(FigHeight)
        fig.set_figwidth(FigWidth)
        plt.gca().invert_yaxis()
        
    # algoritmo de recocido simulado
    #def recocido_simulado(self, n_iteraciones, limites, n_pasos, temp, c_test, h, alpha, beta, rho, n):
        #funcion teórica inicial
        #c_t, lambda_t = self.theoretical_dispersion_curve(c_test, h, alpha, beta, rho, n)
         
        # evaluar el punto inicial para la Velocidad de Onda
        #mejor_eval = self.error(c_t, lambda_t)
         
        #vel_esp = np.concatenate((h, beta))
        # solución de trabajo actual
        #curr, curr_eval = vel_esp, mejor_eval

        # ejecutar el algoritmo
        #for i in range(n_iteraciones):
            # Da un paso
            #candidato = curr + np.random.randn(len(limites)) * n_pasos
            #evalua en el modelo
            #c_t, lambda_t = self.theoretical_dispersion_curve(c_test, candidato[0:7], alpha, candidato[7:14], rho, n)
            # evaluar punto candidato
            #candidato_eval = self.error(c_t, lambda_t)

            # comprobar si hay una nueva mejor solución
            #if candidato_eval < mejor_eval:
                # almacenar nuevo mejor punto
                #mejor, mejor_eval = candidato, candidato_eval
                
            # diferencia entre evaluación de puntos de candidato y actual
            #diff = candidato_eval - curr_eval
            # calcular la temperatura para el punto actual
            #t = temp / float(i + 1)
            # calcular el criterio de aceptación de la metrópoli
            #metropolis = np.exp(-diff / t)
            
            # comprobar si debemos mantener el nuevo punto
            #if diff < 0 or np.random.rand() < metropolis:
                # almacenar el nuevo punto actual
                #curr, curr_eval = candidato, candidato_eval

        #return mejor, mejor_eval