class Config:
    def __init__(self,N):
        # data size
        self.N = N
        self.V = 60
        
        # dimensions
        self.D = 50
        self.H = 15
        self.K = 2
        self.M = 15
        
        #experiment
        self.seed = 66632532
        self.seed_validation = 333
        