import tensorflow as tf

gf = tf.compat.v1.GraphDef()   
m_file = open('models/ConvNet_10Classes_070220.pb','rb')
gf.ParseFromString(m_file.read())

with open('FindConvertInputOutputArrays.txt', 'a') as the_file:
    for n in gf.node:
        the_file.write(n.name+'\n')

file = open('FindConvertInputOutputArrays.txt','r')
data = file.readlines()
print("output name = ")
print(data[len(data)-1])

print("Input name = ")
file.seek ( 0 )
print(file.readline())