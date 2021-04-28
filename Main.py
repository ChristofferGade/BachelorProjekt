import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


class OLG:
    """ The Class contain the OLG model

    """
    #########################
    # The M Definitions #
    #########################

    def __init__(self, **kwargs):

        self.baseline_parameters() # A struct containing the parameters
        self.primitive_functions()
        self.update_parameters(kwargs)
        self.population_growth() # Gather the natual and predicted growth rate (1950-2100)

    def baseline_parameters(self):

        # Demographics
        #self.n = 0.51                                   # Initial growth rate
        #self.N_iniY = 0.35
        #self.N = [self.N_iniY,self.N_iniY/(1+self.n),self.N_iniY/(1+self.n)**2]       # Initial population
                          

        # Household
        self.rho = 0.2                # Discount rate
        self.beta = 1/(1+self.rho)      # Discount factor
        self.phi = 0.8                  # Leisure preference intensity

        # Initial allocation assumption of middel aged
        self.l1_m = 0.4
        self.s1_m = 2
        self.t = 0

        # Production
        self.alpha = 1/3                # Capital share in production
        self.A = 5                     # Tecnology level
        self.delta = 0.2               # Depreciation rate

        # Human Capital
        self.delta_h = 0.072            # Depreciation
        self.theta = 1                  # Positive Scale
        self.kappa = 0.8               # HC inheritance share
        self.H = 1

        self.E_share = 0.05
        self.E = self.E_share * 5
        self.epsilon = 0.10            # elasticity of education spending
        self.gamma = 0.70                 # elasticity of time allocation (gives weird results)


        # Government
        self.tau_k = 0.35               # Taxation of capital
        self.tau_w = 0.2                 # Taxation of labour
        self.pi = 0.3                  # Pension contribution rate

        # Code Convenience
        self.k_min = 1e-10
        self.k_max = 20                 # Note: Make sure capital wont reach this level and that the convergence is true (not to low density)

        self.seed = 1999
        self.T = 20 

        # Plot
        self.Density = 30
        self.y_axis_min = 0 
        self.x_axis_min = 0

    def primitive_functions(self):

        eps = 1e-12 # Minimum evaluation

        # Utility Function (note: add leisure as a optimal function of c in same period)
        self.u = lambda c,l,h: np.log(np.fmax(c,eps))+self.phi*np.log(np.fmax(l,eps)*np.fmax(h,eps)) 

        # Production Function
        self.f = lambda k: self.A*np.fmax(k,eps)**self.alpha*self.L**(1-self.alpha)
        self.fprime = lambda k: self.alpha*self.A*np.fmax(k,eps)**(self.alpha-1)*self.L**(1-self.alpha)

        # Human Capital Accumulation
        self.h = lambda h,e: (1-self.delta_h) * h + self.theta * e**self.gamma * h * self.E**self.epsilon
        
        # Effictive Wages
        self.W = lambda w,h,tau_p: (1-self.tau_w-tau_p) * w * h

    def update_parameters(self, kwargs):
        # For interactive plots (widgets)
        for key, value in kwargs.items():
            setattr(self, key, value)


    ##########################
    # Gather the growth rate #
    ##########################    
    
    def population_growth(self):

        # a. Import the CSV file with population from UN population division
        df = pd.read_csv('WPP2019_TotalPopulationBySex.csv')

        # b. Choose the World population and Medium-variant projection based
        A = df.loc[df['Location']=='World']
        A = A.loc[df['VarID']==2]

        # c. Find the Growth rate from 1950-2100
        Pop = pd.concat([A['Time'],A['PopTotal']],axis=1)
        Growth = Pop.PopTotal.pct_change().rename('Difference')
        Growth = Growth[1:] # removes the first NaN (150 observation left)

        # d. Find the 25 year average growth rate 
        five_year_growth_rate = Growth.groupby(np.arange(len(Growth))//25).mean()
        self.n_data = (1+five_year_growth_rate.to_numpy())**25-1 
        
        # Setting the last periods to a constant growth rate
        #self.n_data = np.append(self.n_data,[0.02]*(self.T-len(self.n_data)))

        #self.n_data = np.append([0.1,0.18947953,0.33223047,0.50601531],self.n_data)


        # Baseline Model
        self.nfuture = -0.06
        self.n_data = self.n_data[:]
        self.n_data = np.append(self.n_data,[self.nfuture]*(self.T+1-len(self.n_data)))


        # Setting the first growth rate
        #self.n = self.n_data[0]
        Init_young = 0.35
        Init_growth = 0.4
        self.N = [Init_young,Init_young/(1+Init_growth),Init_young/(1+Init_growth)**2]
        # Creaating the Population 
        self.N_Pop = np.ndarray([self.T+2,3])
        self.N_Pop[0,:]= self.N
        for i in range(self.T+1):
            self.N[2] = self.N[1]
            self.N[1] = self.N[0]
            self.N[0] = self.N[0]*(1+self.n_data[i])
            self.N_Pop[i+1,:] = self.N

    #####################
    # Solving the Model #
    #####################

    def Pop_work(self,l0,e0,l1_m,h0):
        
        self.L = self.N[0]*(1-l0-e0)*h0 + self.N[1]*(1-l1_m)*self.h(self.H/self.kappa,0)
        return self.L


    def solve_firm_problem(self,k,TA,h0,t=0):
        
        # Unpack time allocations
        l0, e0 ,l1_m = TA

        # Update working population
        self.Pop_work(l0,e0,l1_m,h0)

        # Interest Rate
        R = 1+self.fprime(k)-self.delta

        # Wage Rate
        w = self.f(k)*(1/self.L)-self.fprime(k)*(k/self.L)

        return R, w

    def Household_variables(self,c0,l0,e0,k,k_plus,kg):

        # Gather the middel aged allocations
        l1_m = self.l1_m

        # Human capital
        h0 = self.h(self.H,e0)
        h1 = self.h(h0,0)
        h2 = self.h(h1,0)

        # Growth in h
        hg =  kg*0 #1.2 # ((self.H/self.kappa)-h0)/h0
        k_plus2 = k_plus*1.04

        # Define Timeallocations (We assume optimal allocation and doesnt change with time)
        TA = [l0,e0,l1_m] # Current time allocations
        
        # Current and future prices
        tau_p = (self.N[2]/(self.N[0]+self.N[1]))*self.pi
        R0, w0 = self.solve_firm_problem(k,TA,h0)

        self.N = self.N_Pop[self.t+1,:]
        tau_p1 = (self.N[2]/(self.N[0]+self.N[1]))*self.pi
        h01 = self.h(h0*self.kappa,e0)
        R1, w1 = self.solve_firm_problem(k_plus,TA,h01)
        R2 = R1
        w2 = w1

        self.N = self.N_Pop[self.t,:]

        # Future pension benefits
        h1_mid = self.h(self.H/self.kappa,0)

        Pens2 = self.pi * w2 * (h1 * (1-l1_m) + h0 * (1-l0-e0))         

        # Find leisure middel age (Optimal rule used)
        W0 = self.W(w0, h0, tau_p)
        W1 = self.W(w1, h1, tau_p1)
        l1 = self.beta * (1+R1*(1-self.tau_k)) * (W0/W1)* l0

        # Define Consumption middel age (Optimal rule used)
        c1 = self.beta * (1+R1*(1-self.tau_k)) * c0

        # Define savings for the two periods 
        s0 = (1-self.tau_w-tau_p) * (1-l0-e0) * w0 * h0 - c0
        s1 = (1+R1*(1-self.tau_k))*s0 + (1-self.tau_w-tau_p1)*(1-l1)*w1*h1-c1        

        # Define consumption in the last period
        c2 = (1+R2*(1-self.tau_k))*s1+Pens2

        return h0, h1, h2, l1, c1, c2, s0, s1, tau_p
        
    
    def lifetime_utility_young(self,x,k,k_plus,kg):
        
        # Unpack the allocation parameters
        c0 = x[0]
        l0 = x[1]
        e0 = x[2]
        
        # gather the implication of the choices
        I = self.Household_variables(c0,l0,e0,k,k_plus,kg)

        # Find human capital initial
        h0 = I[0]
        h1 = I[1]
        h2 = I[2]

        # Future leisure
        l1 = I[3] 
        l2 = 1

        # Future consumption
        c1 = I[4]
        c2 = I[5]

        U = self.u(c0,l0,h0)+self.beta*self.u(c1,l1,h1) + self.beta**2*self.u(c2,l2,h2)

        return -U

    def solve_household_problem(self,k,k_plus):
        

        # Assume we are in steady state
        kg = ((k_plus-k)/k)#((self.N[0]-self.N[1])/self.N[0])
        if kg >=2:
            kg = 1

        # Initial Guess
        x0 = [1,0.2,0.2]

        # Bounds
        bound_c = (0,k)
        bound_l = (0,0.9)
        bound_e = (0,1)
        bnds = (bound_c,bound_l,bound_e)

        # Constraints
        def constraint1(x,k):
            # Constraint c to be maximum equal wage ( w >= c)
            TA = [x[1],x[2],self.l1_m]
            h0 = self.h(self.H,x[2])
            return self.solve_firm_problem(k,TA,h0)[1]*(1-self.tau_w)-x[0]

        def constraint2(x): # (1 >= l + e)
            return 1-x[1]-x[2]
        
        con1 = {'type': 'ineq', 'args': (k, ), 'fun':constraint1}
        con2 = {'type': 'ineq', 'fun':constraint2}
        cons = [con1,con2]

        # Optimization
        result = optimize.minimize(self.lifetime_utility_young, x0, method = "SLSQP",\
            args = (k, k_plus,kg, ), bounds = bnds, constraints=cons)

        # a. Unpack
        c0,l0,e0 = result.x
        
        # b. Gather the savings
        Answer = self.Household_variables(c0,l0,e0,k,k_plus,kg)

        s0 = Answer[6]
        s1 = self.s1_m # current saving of middel aged

        # e. Aggregated savings
        S = s0 * self.N[0] + s1*self.N[1]

        return S, s0, s1, c0, l0, e0, Answer

    def find_equilibrium(self, k_plus,disp=0):



        # b objective function to minimize
        def obj(k):
            

            # saving
            S = self.solve_household_problem(k,k_plus)[0]

            # deviation of capital to day vs tomorrow
            return (k_plus-S)**2

        k_min = 0
        k_max = self.k_max+1

        k = optimize.fminbound(obj,k_min,k_max,disp=disp)

        # Update mid age  

        return k


    ##############################
    # Find the transition Curve  #
    ##############################


    def find_transition_curve(self):

        # a determine the k_plus grid as all possible points 
        self.k_plus_grid = np.linspace(self.k_min, self.k_max, self.Density)

        # b. implid current capital
        self.k_grid = np.empty(self.Density)
        for i, k_plus in enumerate(self.k_plus_grid):
            k = self.find_equilibrium(k_plus)
            self.k_grid[i] = k    



    #########################
    # Simulating the Model  #
    #########################

    def simulate(self, reset_seed=True, k_initial=1, shock = False, shock_permanent = True):
        if reset_seed:
            np.random.seed(self.seed)


        self.find_transition_curve()

        # a. initialize

        # Capital and output
        self.sim_k = np.empty(self.T)
        self.sim_k[0] = k_initial

        self.y_output = np.empty(self.T)
        self.y_output[0] = self.f(k_initial)

        # Population
        self.pop = np.empty(self.T)
        self.pop[0] = np.sum(self.N_Pop[0,:])
        self.sim_n = np.empty(self.T)
        self.sim_n[0] = self.n_data[0]

        #self.N_overview = np.ndarray((self.T,3))
        #self.N_overview[0,:] = self.N

        # Variables at interest
        self.sim_k_plus = np.empty(self.T)
        self.sim_w = np.empty(self.T)
        self.sim_r = np.empty(self.T)
        self.sim_s0 = np.empty(self.T)
        self.sim_s1 = np.empty(self.T)        
        self.sim_c0 = np.empty(self.T)
        self.sim_c1 = np.empty(self.T)
        self.sim_c2 = np.empty(self.T)                
        self.sim_l0 = np.empty(self.T)
        self.sim_l1 = np.empty(self.T)
        self.sim_e0 = np.empty(self.T)  

        self.sim_h0 = np.empty(self.T) 
        self.sim_h1 = np.empty(self.T) 
        self.sim_h2 = np.empty(self.T) 

        self.sim_L = np.empty(self.T)  
        self.sim_L_force = np.empty(self.T)
        self.sim_L_eff = np.empty(self.T)           


        # Human capital 
        self.sim_E = np.empty(self.T)
        self.sim_E[0] = 0.25

        # Pension scheme
        self.sim_tau_p = np.empty(self.T)
        self.pens_con = np.empty(self.T)
        self.pens_ben = np.empty(self.T)
         

        # b. time loop
        for t in range(self.T - 1):
            
            # Set the period global
            self.t = t+1

            # Decides the population in the next period to determine current capital.
            self.sim_n[t+1] = self.n_data[t+1]
            self.pop[t+1] = np.sum(self.N_Pop[t+1,:])
            

            self.N = self.N_Pop[t+1,:]
            
            # Decide tomorrow education spending dependint on todays output
            #self.E = self.y_output[t]*self.E_share
            #self.sim_E[t+1] = self.y_output[t]*self.E_share
            
            # Decides the transition curve for which tomorrows capital is decided
            self.find_transition_curve()
            
            # i. current
            k = self.sim_k[t]

            # ii. list of potential future values
            k_plus_list = []

            for i in range(1, self.Density):

                if (
                    k >= self.k_grid[i - 1] and k < self.k_grid[i]
                ):  # between grid points

                    # o. linear interpolation
                    dy = self.k_plus_grid[i] - self.k_plus_grid[i - 1]
                    dx = self.k_grid[i] - self.k_grid[i - 1]
                    k_plus_interp = self.k_plus_grid[i - 1] + dy / dx * (
                        k - self.k_grid[i - 1]
                    )

                    # oo. append
                    k_plus_list.append(k_plus_interp)

            # iii. random draw of future value
            if len(k_plus_list) > 0:
                self.sim_k[t + 1] = np.random.choice(k_plus_list, size=1)[0]
            else:
                self.sim_k[t + 1] = 0

            # CAPITAL TOMORROW IS FOUND

            # Finds the output
            self.y_output[t+1]=self.f(self.sim_k[t+1])
            #self.sim_E[t+1] = self.y_output[t]*self.E_share
            # Problem: Sends the labour force and growth rate in for the next period
            #pop_reset = self.Pop(t+1,prev=True)

            #if t == 0:
            #    self.E = self.sim_E[0]
            #else:
            #    self.E = self.sim_E[t-1]
            
            # Set the population back to the current so we can determine 
            self.N = self.N_Pop[t,:]
            self.t = t
            self.find_transition_curve()

            # runs the equilibrium for the given capital future
            x = self.solve_household_problem(self.sim_k[t],self.sim_k[t+1])
            Answer = x[6]
            # S, s0, s1, c0, l0, e0, Answer
            # Answer = h0, h1, h2, l1, c1, c2, s0, s1, tau_p

            self.sim_s0[t] = x[1]
            self.sim_s1[t] = x[2] # The current savings      
            self.sim_c0[t] = x[3]   
            self.sim_l0[t] = x[4]
            self.sim_e0[t] = x[5]    
            self.sim_l1[t] = self.l1_m # Current mid aged leisure time
            self.sim_c1[t] = Answer[4] # This is the optimal for the next gen
            self.sim_c2[t] = Answer[5]  

            self.sim_h0[t] = Answer[0]              
            self.sim_h1[t] = Answer[1]              
            self.sim_h2[t] = Answer[2]

            self.sim_tau_p[t] = Answer[8]          

            # Determine labour supply and labour force
            self.sim_L[t] = self.N[0]*(1-x[4]-x[5])+self.N[1]*(1-self.l1_m)
            self.sim_L_force[t] = self.N[0]+self.N[1]

            # Determine effective labour supply
            self.sim_L_eff[t] = self.Pop_work(x[4],x[5],self.l1_m, Answer[0])

            # Wage and interest rate
            TA = [x[4],x[5],self.l1_m]
            self.sim_r[t], self.sim_w[t] =  self.solve_firm_problem(self.sim_k[t],TA,Answer[0])

            # Update middel aged optimal allocation
            self.s1_m = Answer[7]
            self.l1_m = Answer[3]
            self.H = self.kappa * Answer[0]

            self.E = self.y_output[t]*self.E_share
            


        # Since we dont have k_plus in 11 we dont have k in 10 
        # and thus we dont have s, l, c in 9 or rather they are zero
        index = self.T-1
        self.sim_s0 = np.delete(self.sim_s0, index)
        self.sim_s1 = np.delete(self.sim_s1, index)      
        self.sim_c0 = np.delete(self.sim_c0, index) 
        self.sim_l0 = np.delete(self.sim_l0, index)
        self.sim_e0 = np.delete(self.sim_e0, index)    
        self.sim_l1 = np.delete(self.sim_l1, index)
        self.sim_c1 = np.delete(self.sim_c1, index)
        self.sim_c2 = np.delete(self.sim_c2, index) 

        self.sim_h0 = np.delete(self.sim_h0, index)
        self.sim_h1 = np.delete(self.sim_h1, index)
        self.sim_h2 = np.delete(self.sim_h2, index)

        self.sim_L = np.delete(self.sim_L, index)
        self.sim_L_force = np.delete(self.sim_L_force, index)
        self.sim_L_eff = np.delete(self.sim_L_eff, index)

        self.sim_r = np.delete(self.sim_r, index) # since we dont have leisure
        self.sim_w = np.delete(self.sim_w, index) # since we dont have leisure

        self.sim_tau_p = np.delete(self.sim_tau_p, index)
        
        # Pension
        #self.tau_p = (1/(1+self.sim_n[:-1])*self.pi)
        #self.sim_pension_contribution = self.pop[:-1] * self.tau_p * self.sim_w * (1-self.sim_l)
        #self.sim_pension_benefits = self.pop_lag[:-1] * self.pi * self.sim_w * (1-self.sim_l)

    
    def plot_simulation(self, k_initial=1, shock = False, shock_permanent = True, **kwargs):


        self.simulate(k_initial=k_initial, shock=shock, shock_permanent=shock_permanent)

        if not "ls" in kwargs:
            kwargs["ls"] = "-"
        if not "marker" in kwargs:
            kwargs["marker"] = "o"
        if not "MarkerSize" in kwargs: 
            kwargs["MarkerSize"] = 2


        ### THE ECONOMY ###

        fig = plt.figure(figsize=(6, 8), dpi=200)
        fig.subplots_adjust(hspace = 0.5)

        ax_e = fig.add_subplot(6,1,1) # Population and Labour supply

        ax_e.plot(self.pop, **kwargs)
        ax_e.plot(self.sim_L, **kwargs)
        ax_e.plot(self.sim_L_force, **kwargs)
        ax_e.set_xlim([0, self.T])
        ax_e.set_ylim([0,self.pop[-1]+0.5])
        ax_e.set_xlabel("time")
        ax_e.set_ylabel("$N_t$, $L_t$ and labour force")

        ax_e1 = fig.add_subplot(6,1,2) # pension tax

        ax_e1.plot(self.sim_n, **kwargs)
        ax_e1.set_xlim([0, self.T])
        ax_e1.set_xlabel("time")
        ax_e1.set_ylabel("Growth rate: $n_t$")           

 
        ax_e2 = fig.add_subplot(6,1,3) # Kapital

        ax_e2.plot(self.sim_k, **kwargs)
        ax_e2.plot(self.y_output, **kwargs)
        ax_e2.plot(self.sim_L_eff, **kwargs)
        ax_e2.set_xlim([0, self.T])
        ax_e2.set_xlabel("time")
        ax_e2.set_ylabel("$K_t$  $Y_t$") 


        ax_e3 = fig.add_subplot(6,1,4) # Capital and output pr. effective capita

        ax_e3.plot((self.sim_k[:-1]/(self.pop[:-1]*(self.sim_h0[:]+self.sim_h1[:]+self.sim_h2[:]))), **kwargs)
        ax_e3.plot((self.y_output[:-1]/(self.pop[:-1]*(self.sim_h0[:]+self.sim_h1[:]+self.sim_h2[:]))), **kwargs)
        ax_e3.set_xlim([0, self.T])
        ax_e3.set_xlabel("time")
        ax_e3.set_ylabel("$k_t$,  $y_t$") 

        ax_e4 = fig.add_subplot(6,1,5) # Wage and interest rate

        ax_e4.plot(self.sim_r, **kwargs)
        ax_e4.plot(self.sim_w, **kwargs)
        ax_e4.set_xlim([0, self.T])
        ax_e4.set_xlabel("time")
        ax_e4.set_ylabel("$r_t$  $w_t$")    

        ax_e5 = fig.add_subplot(6,1,6) # pension tax

        ax_e5.plot(self.sim_tau_p, **kwargs)
        ax_e5.set_xlim([0, self.T])
        ax_e5.set_xlabel("time")
        ax_e5.set_ylabel("$tau_p$")           



        ### THE HOUSEHOLD ###


        fig1 = plt.figure(figsize=(6, 8), dpi=200)
        fig1.subplots_adjust(hspace = 0.5)        

        ax_h1 = fig1.add_subplot(5,1,1) # Leisure allocation

        ax_h1.plot(self.sim_l0, **kwargs)
        ax_h1.plot(self.sim_e0, **kwargs)
        ax_h1.set_xlim([0, self.T])
        ax_h1.set_ylim([0,1])
        ax_h1.set_xlabel("time")
        ax_h1.set_ylabel("Young time allocation: $l_t$ $e_t$ ")       

        ax_h2 = fig1.add_subplot(5,1,2) # Consumption and savings

        ax_h2.plot(self.sim_c0, **kwargs)
        ax_h2.plot(self.sim_s0, **kwargs)
        ax_h2.set_xlim([0, self.T])
        ax_h2.set_xlabel("time")
        ax_h2.set_ylabel("Young: $c_t$ $s_t$")

        ax_h3 = fig1.add_subplot(5,1,3) # Consumption and savings

        ax_h3.plot(self.sim_h0, **kwargs)
        ax_h3.plot(self.sim_h1, **kwargs)
        ax_h3.plot(self.sim_h2, **kwargs)
        ax_h3.set_xlim([0, self.T])
        ax_h3.set_xlabel("time")
        ax_h3.set_ylabel("$h_t$")

        ax_h4 = fig1.add_subplot(5,1,4) # Consumption and savings

        ax_h4.plot(self.sim_c1, **kwargs)
        ax_h4.plot(self.sim_s1, **kwargs)
        ax_h4.set_xlim([0, self.T])
        ax_h4.set_xlabel("time")
        ax_h4.set_ylabel("Mid age: $c_t$ $s_t$") 

        ax_h5 = fig1.add_subplot(5,1,5) # Consumption and savings

        ax_h5.plot(self.sim_l1, **kwargs)
        ax_h5.set_xlim([0, self.T])
        ax_h5.set_ylim([0,1])
        ax_h5.set_xlabel("time")
        ax_h5.set_ylabel("Middel aged leisure")            



    ###########################
    # Plot Transition Curve   #
    ###########################




    def plot_45(self,ax_e,**kwargs):

        if not "color" in kwargs:
            kwargs["color"] = "black"
        if not "ls" in kwargs:
            kwargs["ls"] = "--"

        ax_e.plot([self.k_min,self.k_max],[self.k_min,self.k_max], **kwargs)

            
    def plot_transition_curve(self,ax_e,**kwargs):
        self.find_transition_curve()

        ax_e.plot(self.k_grid, self.k_plus_grid, **kwargs)

        lim = 10 # self.k_max

        ax_e.set_xlim([self.x_axis_min, lim])
        ax_e.set_ylim([self.y_axis_min, lim])
        ax_e.set_xlabel("$k_t$")
        ax_e.set_ylabel("$k_{t+1}$")

#########
# Plots #
#########

def plot_Pop(ax,df, Region="World"):
    
    W = df.loc[df['Location']==Region]
    W = W.loc[df['VarID']==2]
    W = pd.concat([W['Time'],W['PopTotal']],axis=1)
    Pop = W.PopTotal.to_numpy()
    #Pop = np.log(Pop)

    ax.plot(Pop)

def plot_Pop_growth(ax,df, Region="World"):
    
    W = df.loc[df['Location']==Region]
    W = W.loc[df['VarID']==2]
    W = pd.concat([W['Time'],W['PopTotal']],axis=1)

    Growth = W.PopTotal.pct_change().rename('Difference')
    Growth = Growth[1:] # removes the first NaN (150 observation left)
    Growth = Growth.to_numpy()


    ax.plot(Growth)
    #ax.legend(Region)

def plot(ax,data,T=20):

    ax.plot(data)
    ax.set_xlim([0, T])
    ax.set_xlabel("time") 

    
def gr(x):
    return ((1+x)**(1/25)-1)*100