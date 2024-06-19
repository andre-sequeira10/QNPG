import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import pennylane as qml 
from pennylane import numpy as np 
from torch.distributions import Categorical
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n_qubits", type=int, default=4)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--matrix", type=str, default="fim")
parser.add_argument("--policy", type=str, default="Born")
parser.add_argument("--n_actions", type=int, default=2)

args = parser.parse_args()
n_qubits = args.n_qubits
n_layers = args.n_layers
matrix = args.matrix
policy = args.policy
n_actions = args.n_actions


def ansatz_flatten(state, flat_weights, n_qubits, n_layers=1, change_of_basis=False, entanglement="all2all"):
    #flat_weights = weights.flatten()
    num_weights_per_layer = n_qubits * 2
        
    for l in range(n_layers):
        for i in range(n_qubits):
            index = l * num_weights_per_layer + i * 2
            qml.RZ(flat_weights[index], wires=i)
            qml.RY(flat_weights[index + 1], wires=i)


        qml.broadcast(unitary=qml.CNOT, pattern="all_to_all", wires=range(n_qubits), parameters=None)

        if l < n_layers-1:
            qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Y")
            qml.AngleEmbedding(state, wires=range(n_qubits),rotation="Z")

def measurement(n_qubits,policy):
    if policy == "Born":
        return qml.probs(wires=range(n_qubits))
    elif policy == "softmax":
        #return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        op = qml.operation.Tensor(qml.PauliZ(wires=0),qml.PauliZ(wires=1))
        for i in range(2,n_qubits):
            op = qml.operation.Tensor(op,qml.PauliZ(wires=i))
        
        ops = [op]

        if n_actions <= n_qubits:
            for a in range(1,n_actions):
                if a == 1:
                    op = qml.operation.Tensor(qml.PauliX(wires=0),qml.PauliZ(wires=1))
                else:
                    op = qml.operation.Tensor(qml.PauliZ(wires=0),qml.PauliZ(wires=1))
                for j in range(2,n_qubits):
                    if a == j:
                        op = qml.operation.Tensor(op,qml.PauliX(wires=j))
                    else:
                        op = qml.operation.Tensor(op,qml.PauliZ(wires=j))
                ops.append(op)
        else:
            for a in range(1,n_qubits):
                if a == 1:
                    op = qml.operation.Tensor(qml.PauliX(wires=0),qml.PauliZ(wires=1))
                else:
                    op = qml.operation.Tensor(qml.PauliZ(wires=0),qml.PauliZ(wires=1))
                for j in range(2,n_qubits):
                    if a == j:
                        op = qml.operation.Tensor(op,qml.PauliX(wires=j))
                    else:
                        op = qml.operation.Tensor(op,qml.PauliZ(wires=j))
                ops.append(op)
            for a in range(n_actions-n_qubits):
                op = qml.operation.Tensor(qml.PauliX(wires=a),qml.PauliX(wires=(a+1)%n_qubits))
                for j in range(n_qubits):
                    if j != a and j != a+1:
                        op = qml.operation.Tensor(op,qml.PauliZ(wires=j))

                ops.append(op)
            
        return [qml.expval(op) for op in ops]
    
def qcircuit_fisher(inputs, weights0):
                                    
    for q in range(n_qubits):
        qml.Hadamard(wires=q)

    ansatz_flatten(inputs, weights0,n_qubits, n_layers=n_layers)
    
    return measurement(n_qubits,policy)



weights_np = np.zeros(n_qubits*2*n_layers,requires_grad=True)#0.1*np.random.randn(n_qubits * 2 * n_layers,requires_grad=True)
print("num_params - ",len(weights_np))
weights = torch.tensor(weights_np, requires_grad=True)
inpt_np = np.random.randn(n_qubits,requires_grad=False)
inpt = torch.tensor(inpt_np, requires_grad=False)


def get_fisher_matrix(inputs, weights, n_qubits=n_qubits,m=None,policy=None,n_actions=2):
    torch.autograd.set_detect_anomaly(True)

    dev = qml.device("default.qubit", wires=n_qubits+1, shots=1024)
 
    qnode = qml.QNode(qcircuit_fisher, dev, diff_method=qml.gradients.param_shift,grad_on_execution=False,cache=False)

    with qml.Tracker(dev,persistent=True) as tr:

        if m=="qfim":
            mt = qml.metric_tensor(qnode,approx=None, aux_wire=n_qubits)
            f = mt(inpt_np,weights_np)
            #f = qml.qinfo.quantum_fisher(qnode)(inpt_np,weights_np)
        elif m=="fim":

            out = qnode(inputs,weights)

            if policy == "Born" or policy == "Born_softmax_activation":
                partitions = list(map(len, np.array_split(list(range(2**n_qubits)),n_actions)))
                indexes = torch.split(out, partitions)
                pi = torch.stack([torch.sum(i) for i in indexes])
                pi /= torch.sum(pi)

            elif policy == "softmax":
                out = torch.stack(out)
                pi = torch.exp(out)/torch.sum(torch.exp(out))
                if weights.grad is not None:

                    print("grad not none")
                    weights.grad.zero_()

            dist = Categorical(pi)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_prob.backward()

            grad = weights.grad.view(-1)
            f = torch.ger(grad, grad)

        print(tr.totals)

    
get_fisher_matrix(inpt, weights,n_qubits=n_qubits, m=matrix, policy=policy, n_actions=n_actions)