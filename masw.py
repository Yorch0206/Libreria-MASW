import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

class masw:
    def __init__(self, filename, dt, fs, N, dx, x1, direction, header=6):
        # fs - Sampling frequency
        # N  - Number of channels
        # dx - spacing beetween geophones
        # x1 - offset soure
        # direction - Forward or backward
        self.filename = filename
        self.dt       = dt
        self.header   = header
        self.data     = pd.read_csv(filename, header=self.header, delimiter="\t", skip_blank_lines=False)
        self.fs       = fs
        self.N        = N
        self.dx       = dx
        self.x1       = x1
        self.direction = direction
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
        omega_fs = 2*np.pi*self.fs 
        U        = np.zeros_like(self.data)
        P        = np.zeros_like(self.data)
        Unorm    = np.zeros_like(self.data)

        Unp   = self.data.to_numpy()
        U     = np.fft.fft(self.data, axis = 0 )
        
        Unorm = U/abs(U)
        P     = np.exp(-1j*np.angle(U))

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
        self.Aplot = self.A[ no_fmin:no_fmax+1, :]
        self.fplot      = self.f[ no_fmin:no_fmax+1]
        self.cplot      = self.cT

        fig, ax = plt.subplots(1,1, figsize = (FigWidth,FigHeight))

        cntr1 = ax.contourf(self.fplot, self.cplot, self.Aplot.T, levels=resolution, cmap="RdBu_r")
        ax.set_xlabel('Frecuencia [Hz]')
        ax.set_ylabel('Velocidad de Fase [m/s]')
        fig.colorbar(cntr1, ax=ax)
        return fig, ax
    
    def plot_dispersion_image_3D(self, f, c, A,fmin, fmax, FigWidth = 8, FigHeight = 8, FigFontSize = 12):
        #Límites del eje de frecuencia
        no_fmin    = np.argmax(self.f >= fmin)
        no_fmax    = np.argmax(self.f >= fmax)
        
        #Seleccione los datos correspondientes al rango de frecuencia [fmin, fmax]
        #Calcular el valor absoluto (longitud) de números complejos
        Aplot = self.A[ no_fmin:no_fmax+1, :]
        fplot = self.f[ no_fmin:no_fmax+1]
        cplot = self.cT
        
        #Grafica la imagen de dispersión 3D
        fig, ax = plt.axes(projection='3d')
        cntr1 = ax.plot_surface(fplot, cplot, Aplot, rstride=1, cstride=1, cmap='jet', edgecolor='none')
        ax.set_xlabel('Frecuencia [Hz]')
        ax.set_ylabel('Velocidad de Fase [m/s]')
        ax.set_zlabel('Amplitud Normalizada')
        fig.colorbar(cntr1, ax=ax)

    def extract_dispersion_curve(self, fig, ax, f_receivers, select, up_low_boundary, p, resolution=100, FigWidth=8, FigHeight=8):
        Aabsnorm2 = np.zeros_like(self.Aplot)
        Aabsnorm2 = (self.Aplot.T/self.Aplot.max(axis=1)).T
        freq_ind, c_ind = np.where(Aabsnorm2 == 1)

        fvec = []
        for fi in freq_ind:
            if self.fplot[fi] > f_receivers:
                fvec.append(self.fplot[fi])
        fvec = np.array(fvec)
        
        cvec = []                
        for m, fi in enumerate(freq_ind):
            if self.fplot[fi] > f_receivers:
                cvec.append(self.cplot[c_ind[m]])
        cvec = np.array(cvec)
        
        ind = np.where(fvec > f_receivers)
        fvec = fvec[ind]
        cvec = cvec[ind]
        
        
        #Ordenar los puntos
        fvec_sort = fvec.copy()
        I = fvec_sort.argsort()
        fvec_sort.sort()
               
        cvec_sort = cvec[I]
        
        X, Y = np.meshgrid(fvec_sort, cvec_sort)
        
        #Grafica los máximos encima de la imagen de dispersión
        #fig, ax = plt.subplots(1, 1, figsize = (FigWidth,FigHeight))
        #cntr1 = ax.contourf(self.fplot,self.cplot,self.Aplot.T,levels=resolution, cmap="RdBu_r")
        #print("fvec_sort: ",fvec_sort)
        #print("cvec_sort: ", cvec_sort)
        ax.plot(fvec_sort, cvec_sort,'o', markersize = 4, markerfacecolor = 'k', color = 'k')
        #ax.set_xlabel('Frecuencia [Hz]')
        #ax.set_ylabel('Velocidad de Fase [m/s]')
        
        
        if up_low_boundary == 'yes':
            
            #Límites superior / inferior para la curva de dispersión del modo fundamental
            c_p, f_p = np.where(Aabsnorm2.T > (p/100))
            
            fvec_p = []
            for fi_2 in f_p: 
                 if self.fplot[fi_2] > f_receivers:
                     fvec_p.append(self.fplot[fi_2])
            fvec_p = np.array(fvec_p)
                     
            cvec_p = []
            for m_2, fi_2 in enumerate(f_p):
                 if self.fplot[fi_2] > f_receivers:
                     cvec_p.append(self.cplot[c_p[m_2]])
            cvec_p = np.array(cvec_p)
    
            fvec_p = fvec_p[fvec_p > f_receivers]
            cvec_p = cvec_p[fvec_p > f_receivers]
            
            #Ordenar puntos
            Amax_fvec_sort_p = fvec_p.copy() 
            I = fvec_p.argsort()
            Amax_fvec_sort_p.sort()
            
            Amax_cvec_sort_p = cvec_p[I]
            
            #Amax_fvec_sort_p_cell = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))
            Amax_fvec_sort_p_cell = {}
            #Amax_cvec_sort_p_cell = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))
            Amax_cvec_sort_p_cell = {}
            f_curve0_up_temp = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))
            c_curve0_up_temp = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))
            f_curve0_low_temp = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))
            c_curve0_low_temp = np.zeros((len(np.unique(Amax_fvec_sort_p)),1))

            U = np.unique(Amax_fvec_sort_p)
            
            for i in range(len(U)):
                Amax_fvec_sort_p_cell[i] = Amax_fvec_sort_p[np.where(Amax_fvec_sort_p == U[i])]
                Amax_cvec_sort_p_cell[i] = Amax_cvec_sort_p[np.where(Amax_fvec_sort_p == U[i])]
                f_curve0_up_temp[i] = Amax_fvec_sort_p_cell[i][-1]
                c_curve0_up_temp[i] = Amax_cvec_sort_p_cell[i][-1]
                f_curve0_low_temp[i] = Amax_fvec_sort_p_cell[i][0]
                c_curve0_low_temp[i] = Amax_cvec_sort_p_cell[i][0]
            
            #Graficar los máximos encima de la imagen de dispersión
            ax.plot(Amax_fvec_sort_p, Amax_cvec_sort_p,'o',markersize = 1, markerfacecolor = 'k', color='k')
            ax.plot(f_curve0_up_temp,c_curve0_up_temp,'o',markersize = 4, markerfacecolor = 'k', color='k')
            ax.plot(f_curve0_low_temp,c_curve0_low_temp,'o',markersize = 4, markerfacecolor = 'k', color='k')
            
            if (select == 'numbers') or (select == 'both'):
                for label in Amax_fvec_sort_p:
                    ax.text(fvec_sort, cvec_sort, str(label), verticalalignment = 'bottom',horizontalalignment = 'right')
                #clear figure (hold off)
                
                #Curva de dispersión del modo fundamental
                nP_start = input('Inicio de curva de dispersión del modo funamental: ')
                nP_end = input('Fin de curva de dispersión del modo funamental: ')
                nP0 = np.arange(int(nP_start), int(nP_end)+1)
                f_curve0 = fvec_sort[nP0]
                c_curve0 = cvec_sort[nP0]
                
                if up_low_boundary == 'yes':
                    f_curve0_up = f_curve0_up_temp[nP0]
                    c_curve0_up = c_curve0_up_temp[nP0]
                    f_curve0_low = f_curve0_low_temp[nP0]
                    c_curve0_low = c_curve0_low_temp[nP0]
                
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
                        
                lambda_curve0 = np.array([c_curve0]) / np.array([f_curve0])
                
                if up_low_boundary == 'yes':
                    lambda_curve0_up = np.array([c_curve0_up]) / np.array([f_curve0_up])
                    lambda_curve0_low = np.array([c_curve0_low]) / np.array([f_curve0_low])
            
        if select == 'mouse':
               #hold off
               
               #Curva de dispersión del modo fundamental
               print ('Seleccione la curva de dispersión del modo fundamental. Presione el botón de en medio para dejar de seleccionar puntos.')
               f_curve0, c_curve0 = plt.ginputs()
               lambda_curve0 = np.array([c_curve0]) / np.array([f_curve0])
               
               if up_low_boundary == 'yes':
                   print ('Seleccione puntos adicionales para la curva de dispersión del límite superior. Presione el botón de en medio para dejar de seleccionar puntos.')
                   f_curve0_up, c_curve0_up = plt.ginputs()
                   lambda_curve0_up = np.array([c_curve0_up]) / np.array([f_curve0_up])
                   
                   print ('Seleccione puntos adicionales para la curva de dispersión del límite inferior. Presione el botón de en medio para dejar de seleccionar puntos.')
                   f_curve0_low, c_curve0_low = plt.ginputs()
                   lambda_curve0_low = np.array([c_curve0_low]) / np.array([f_curve0_low])
                   
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
        
    def MASWaves_Ke_layer(h, alpha, beta, rho, c_test, k):
        r = np.sqrt(1 - c_test**2 / alpha**2)
        s = np.sqrt(1 - c_test**2 / beta**2)
        
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
        
        Ke = [[k11_e, k12_e, k13_e, k14_e],
              [k21_e, k22_e, k23_e, k24_e],
              [k31_e, k32_e, k33_e, k34_e],
              [k41_e, k42_e, k43_e, k44_e]]
        
        return Ke

    def MASWaves_stiffness_matrix(self, c_test, k, h, alpha, beta, rho, n):
        #Matriz de rigidez del sistema
        K = np.zeros(2*(n+1), 2*(n+1))
        
        #Compruebe si la velocidad de la fase de prueba es igual a la velocidad de la onda de corte 
        #o la velocidad de la onda de compresión de una de las capas
        epsilon = 0.0001
        while any(abs(c_test-beta)<epsilon) or any(abs(c_test-alpha)<epsilon):
            c_test = c_test*(1-epsilon)
        
        #Capas de espesor finito j = 1, ..., n
        for j in range (0, n):
            #Compute element stiffness matrix for layer j
            Ke = self.MASWaves_Ke_layer(h[j], alpha[j], beta[j], rho[j], c_test, k)
            
            #Agregar a la matriz de rigidez del sistema
            DOFS = np.arange(2*j-1, 2*j+2)
            K[DOFS, DOFS] = K[DOFS, DOFS] + Ke
        
        #Medio espacio
        #Calcular la matriz de rigidez del elemento para la mitad del espacio
        Ke_halfspace = self.MASWaves_Ke_halfspace(alpha(end), beta(end), rho(end), c_test,k)
        
        #Agregar a la matriz de rigidez del sistema
        DOFS = np.arange(2*(n+1)-1, 2*(n+1))
        K[DOFS, DOFS] = K[DOFS, DOFS] + Ke_halfspace

        #Evaluar determinante de la matriz de rigidez del sistema
        k_det = np.linalg.det(K)
        D = k_det.real
        
        return None
    
    def theorical_dispersion_curve(self, c_test, lambda_curve0, h, alpha, beta, rho, n):
        k = (2*np.pi) + lambda_curve0        
        D = np.zeros(len(c_test), len(k))
        c_t = np.zeros(len(k), 1)
        
        for l in range(len(k)):
            for m in range(len(c_test)):
                D[l, m] = self.stiffness_matrix(c_test(m), k(l), h, alpha, beta, rho, n)
                if m ==1:
                    sign_old = np.sign(D[l, m])
                else:
                    sign_old = np.sign(D)
                signD = np.sign(D[l, m])
                
                if sign_old * signD == -1:
                    c_t[l] = c_test[m]
                    lambda_t[l] = 2*np.pi / k[l]                    
            
        return None

    def inversion(self, c_test, h, alpha, beta, rho, n, up_low_boundaries, c_curve0, lambda_curve0, c_curve0_up, lambda_curve0_up, c_curve0_low, lambda_curve0_low):
        
        return None