import numpy as np
import matplotlib.pyplot as plt

class PLA:

    def pla(self, N, plt = False):
        dataset = self.D(N)
        f_ln    = self.rnd()

        w0      = np.zeros((1, 3))
        w       = np.zeros((1, 3))

        for j in range(10000):
            for it, i in enumerate(w):
                w0[it] = i

            self.runPLA(dataset, f_ln, w)
            dist = np.linalg.norm(w - w0)

            if (dist < 1E-4):
#                print("Converged")
                N_iter = j
                break

        if plt:
            self.plot(dataset, f_ln, w0)

        self.w    = w
        self.f    = f_ln

        return N_iter


    def classifyPLA(self, w, x): 
        w_eval       = np.zeros((1, 2))
        w_eval[0][0] = w[0][1]
        w_eval[0][1] = w[0][2]

        dot_prod = np.dot(w_eval[0], x) + w[0][0]

        classify = np.sign(dot_prod)

        return classify
 
    def runPLA(self, d, f_ln, w): 
        w_eval       = np.zeros((1, 2))
        w_eval[0][0] = w[0][1]
        w_eval[0][1] = w[0][2]

        x     = [] # Get all misclassified values
        for it, i in enumerate(d):
            y        = self.eval_f(i, f_ln) # Evaluate fn at randomly selected point

            classify = self.classifyPLA(w, i) 
    
            if ( classify != y):
                x.append(i)

        x     = np.array(x)
        N     = x.shape[0]

#        print(w, N)

        if (N > 0):
            randN = np.random.randint(N)
    
            x_r     = x[randN]
            y       = self.eval_f(x_r, f_ln) # Evaluate fn at randomly selected point
            w_eval  = w_eval + y*x_r

            w[0][0] = w[0][0] + y*1.0
            w[0][1] = w_eval[0][0]
            w[0][2] = w_eval[0][1]
            


    def D(self, N): #Create data set of size N
        rd = 2*np.random.random_sample((N, 2)) - 1 #Random sample in [-1, 1]

        return rd 

    def eval_f(self, d, f): #evaluate function
        ln = f[1] - f[0]

        y  = 0.0 

        sgn = np.cross(d - f[0], ln)
        if (sgn > 0):
            y =  1
        else:
            y = -1

        return y

    def plot(self, d, f, w):
        N = d.shape[0]

        x_ln = np.zeros(2)
        y_ln = np.zeros(2)
        for it, i in enumerate(f):
            x_ln[it] = i[0]
            y_ln[it] = i[1]


        x = np.zeros(N)
        y = np.zeros(N)

        for it, i in enumerate(d):
            x[it] = i[0]
            y[it] = i[1]

        x_w = np.zeros(2)
        y_w = np.zeros(2)

        x_w[0] = 0.0 
        y_w[0] = -w[0][0]/w[0][2] 

        x_w[1] = 1.0 
        y_w[1] =  -w[0][0]/w[0][2] - w[0][1]/w[0][2] 

        plt.plot(x_ln, y_ln)
        plt.plot(x_w, y_w, color = 'r')
        plt.scatter(x, y)

        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])

        plt.show()


    def rnd(self): #Create random function
        ln = 2*np.random.random_sample((2, 2)) - 1 #End points of line
        return ln


if __name__=="__main__":
    run        = PLA()
    N_sample   = 100
    N_runs     = 1000

    N_sum      = 0
    N_disagree = 0

    for j in range(N_runs):    
        N_iter = run.pla(N_sample)
        N_sum  = N_sum + N_iter
        
        x_rnd = run.D(1)
        y_PLA = run.classifyPLA(run.w, x_rnd[0]) 
        y     = run.eval_f(x_rnd, run.f)

        if (int(y) != int(y_PLA)):
            N_disagree = N_disagree + 1

    print(N_sum/N_runs)
    print(N_disagree, N_disagree/N_runs)
