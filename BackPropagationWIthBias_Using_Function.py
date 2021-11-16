import numpy as np

#taking input of the input layer

'''input_node_number = 3
input_value=[1,1,0]
hidden_layer_number= 1
node_count = [3,2,1]
node_weight = [[0.2, 0.4, -0.5], [-0.3, 0.1, 0.2], [-0.3, -0.2]]
bias=[[-0.4, 0.2], [0.1]]'''
 
def inputValue():
    input_value=[]
    input_node_number= eval(input('enter the number of input nodes: '))
    print('enter the input of the ',input_node_number,' different nodes:')
    for i in range (input_node_number):
        node_value=eval(input())
        input_value.append(node_value)
    return input_value, input_node_number



#taking weight of all nodes in hidden layer
def weightHidden(input_node_number):
    node_count = []
    node_count.append(input_node_number)
    print('\nNode count : ',node_count)
    hidden_layer_number=eval(input('enter the total number of hidden layer : '))
    
    node_weight=[]
    bias=[]
    for i in range (hidden_layer_number):
        print('Enter the number of nodes in layer ',i+1)
        hidden_node_number=eval(input())
        node_count.append(hidden_node_number)
        print('\nNode count : ',node_count)
        bias_h=[]
        for j in range(hidden_node_number):
            temp=[]
            print('\nEnter the weight for node',j+1,'of ',i+1,'layer:')
            for k in range (node_count[i]):
                hidden_value= eval(input())
                temp.append(hidden_value)
            node_weight.append(temp)
            print('\nEnter the bias value for node ',j+1,'of ',i+1,'layer:')
            bias_val = eval(input())
            bias_h.append(bias_val)
        bias.append(bias_h)
    return node_count, bias, node_weight, hidden_layer_number
        


#taking the weights of all nodes in output layer
def weightOutputLayer(node_count, bias, node_weight):
    output_layer_nodes=eval(input('\nEnter the number of nodes in output layer:'))
    node_count.append(output_layer_nodes)
    bias_val2=[]
    for i in range(output_layer_nodes):
        tmp=[]
        a=len(node_count)-2
        print('\nEnter the weight for node ',i+1,'of output layer:')
        for j in range(node_count[a]):
            output_wt=eval(input())
            tmp.append(output_wt)
        node_weight.append(tmp)
        print('\nEnter the bias value for node ',i+1,'of output layer:')
        bias_val = eval(input())
        bias_val2.append(bias_val)
        bias.append(bias_val2)
    return node_weight, bias, node_count

    


# Feed-Forward Calculations
def feedforward(input_value,node_count,node_weight):
    net_input_value=0
    net_input=[]
    value= []
    value.append(input_value)
    node_weight_temp=node_weight[:]
    
    for i in range(len(node_count)-1):
        tmp=[]
        for j in range(node_count[i+1]): 
            for k in range(node_count[i]):
                net_input_value +=value[i][k] * node_weight_temp[j][k]
            net_input_value2 = net_input_value + bias[i][j]
            tmp.append(net_input_value2)
            net_input_value=0
        net_input.append(tmp)
        actual_output=[]
        for l in range(len(net_input[i])):
            #result=(1-np.exp(-net_input[i][l]))/(1+np.exp(-net_input[i][l]))
            result=1/(1+np.exp(-net_input[i][l]))
            actual_output.append(result)
        value.append(actual_output)
        for m in range(node_count[i+1]):
            node_weight_temp.pop(0)
    return value, net_input


        
#forward propagation display
def feedforwardValueDisplay(input_value,node_count,node_weight, hidden_layer_number, value):
    print('\nThe weights are:',node_weight)
    print('\nNode count : ',node_count) 
    print('\nThe input value for the input nodes are:',input_value)
    for i in range(hidden_layer_number):
        #print('\nThe net input for each node in layer',i+1,' are:\n ',net_input[i])
        print('\nThe actual output for each node in layer',i+1,' are:\n ',value[i+1])
    #print('\nThe net input value for the nodes in output layer are:',net_input[len(net_input)-1])
    print('\nThe actual output value for the nodes in output layer are:',value[len(value)-1])



#backpropagation
#error calculation
def errorCalculations(node_weight, node_count, value, target):
    error_list=[]
    wt=[]
    delta_list=[]
    node_weight_temp=node_weight[:]
    node_weight_temp.reverse()
    #print(node_weight_temp)
    for i in range(len(node_count)-1):
        err_temp=[]
        error_list_temp=[]
        a = value[len(value)-(i+1)]
        b=len(value)-(i+1)
        if((len(value)-1)==b):
            for j in range(node_count[len(node_count)-(i+1)]):
                err = (target-a[j]) * a[j] * (1-a[j])
                err_temp.append(err)
        else:
            for j in range(node_count[len(node_count)-(i+1)]):
                err = delta_list[i-1][j]* a[j] * (1-a[j])
                err_temp.append(err)
        error_list_temp.append(err_temp)
        error_list.append(err_temp)
        if(i!=(len(node_count)-2)):
            for k in range(node_count[len(node_count)-(i+1)]):
                node_wt=node_weight_temp[k]
                wt.append(node_wt)
            wt.reverse()
            for l in range(node_count[len(node_count)-(i+1)]):
                node_weight_temp.pop(0)
            
            delta_temp=[]
            for j in range(len(error_list_temp[0])):
                temp_list=[]
                for k in range(len(wt[j])):
                    c = error_list_temp[0][j] * wt[j][k]
                    temp_list.append(c)
                delta_temp.append(temp_list)   
            delta_array=np.array(delta_temp)
            delta_array_sum=delta_array.sum(axis=0)
            delta = delta_array_sum.tolist()
            delta_list.append(delta)
    #print('\nDeltaList ',delta_list)
    return error_list



#new weight update
def weightUpdate(error_list, node_count, node_weight, value, learning_rate):
    error_list_temp=error_list[:]
    error_list_temp.reverse()
    updated_weight=[]
    node_weight_temp=node_weight[:]
    #print('error', error_list_temp)
    for i in range(len(node_count)-1):
        weight_update_temp=[]
        for j in range(node_count[i]): 
            for k in range(node_count[i+1]):
                change_in_weight = learning_rate * error_list_temp[i][k] * value[i][j]
                v_new = node_weight_temp[k][j] +change_in_weight
                weight_update_temp.append(v_new)
        updated_weight.append(weight_update_temp)
        for l in range(node_count[i+1]):
            node_weight_temp.pop(0)
    return updated_weight


#display    
def updateDisplay(node_count, error_list, updated_weight):
    for i in range(len(node_count)-1):
        print('\nThe error in the',len(node_count)-(i+1),'layer is : ',error_list[i])
    for i in range(len(node_count)-1):
        print('\nThe weight update in the layer',i+1,'is : ',updated_weight[i]) 
        

#function call
input_value, input_node_number = inputValue()
node_count, bias, node_weight, hidden_layer_number = weightHidden(input_node_number)
node_weight, bias, node_count = weightOutputLayer(node_count, bias, node_weight)

#setting the target value
target=0
learning_rate=0.9

#function call
value, net_input = feedforward(input_value,node_count,node_weight)
feedforwardValueDisplay(input_value,node_count,node_weight, hidden_layer_number, value)
error_list = errorCalculations(node_weight, node_count, value, target)
updated_weight = weightUpdate(error_list, node_count, node_weight, value, learning_rate)           
updateDisplay(node_count, error_list, updated_weight)
