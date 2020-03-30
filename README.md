# prost--neuralnetwork
Search engine for roll out policy is 'NNRollOut -it <int>'
  
  
'-it' specifies the number of iteration.

The current parameters are for:  "elevators_inst_mdp__1"

run:

python3 prost.py elevators_inst_mdp__1 "[PROST -s 1 -se [UCTStar  -init [Expand -h [NNRollOut -it 3 ]]]]"




Manually set the topology of the network in :
(1)  NNRollOut.cc   in line 19;
(2)  action_selection.h  in line 66;
(3) NN.h line 18;
