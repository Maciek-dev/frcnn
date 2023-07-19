class Data:
    
    def __init__(self, x,y):
        self.x=x
        self.y=y

    def __iter__(self):
        self.n=0
        return self
        #self.batch_size=batch_size

    def __next__(self):
        if self.n<=len(self.x):
            x=self.x[self.n]
            y=self.y[self.n]
            self.n+=1
            return x,y
        

x=[0,1,2,3,4,5,6,7,8,9]
y=[10,11,12,13,14,15,16,17,18,19]
data=Data(x,y)
for i in data:
    print(i)