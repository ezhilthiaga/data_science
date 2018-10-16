import numpy as np
a=np.random.rand(3,2)
print ("a=",a)
b=np.random.rand(2,3)
print("b=",b)
c = np.dot(a,b)
print("vectorization",c)
z=[[0,0,0],
   [0,0,0],
   [0,0,0]]
for i in range(len(a)):
    for j in range(len(b[0])):
        for k in range(len(b)):
            z[i][j]+=a[i][k]*b[k][j]

print(z)

