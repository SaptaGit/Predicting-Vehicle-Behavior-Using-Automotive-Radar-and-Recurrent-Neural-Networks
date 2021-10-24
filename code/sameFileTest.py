import numpy as np


a = np.array([1,2,3])



first = [1,2,3,4]
second = [4,3,2,1]

for item1,item2 in zip(first,second):
  print(str(item1) + str(item2))

  
try:
  print(a[5])
except Exception as e:
  print("Something went wrong")
  print(e)
finally:
  print("The 'try except' is finished")