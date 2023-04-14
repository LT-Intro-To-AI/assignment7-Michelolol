from neural import *

XOR = [
    ([0.0, 0.0],    [0.0]), # [0, 0] => 0
    ([0.0, 1.0],    [1.0]), # [0, 1] => 1
    ([1.0, 0.0],    [1.0]), # [1, 1] => 1
    ([1.0, 1.0],    [0.0]) 
]



voter_opinion = [
    ([.9, .6, .8, .3, .1],[1]),
    ([.8, .8, .4, .6, .4],[1]),
    ([.7, .2, .4, .6, .3],[1]),
    ([.5, .5, .8, .4, .8],[0]),
    ([.3, .1, .6, .8, .8],[0]),
    ([.6, .3, .4, .3, .6],[0])
]

von = NeuralNet(5, 6, 1)
von.train(voter_opinion)
print(von.test_with_expected(voter_opinion))
test_data = [
    [1, 1, 1, .1, .1]
]
print(f"case 1: {von.evaluate(test_data[0])}")