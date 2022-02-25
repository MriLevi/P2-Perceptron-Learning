import activation as act
import perceptron_network as pn
import itertools
import unittest
from collections import defaultdict


class Tests(unittest.TestCase):

    def test_and_gate(self):
        """the and gate outputs 1 if both x1 and x2 (inputs) are 1, else it outputs 0"""
        inputs = list(itertools.product([0, 1], repeat=2))
        weights = [1, 1]
        threshold = 1
        perceptrons = [pn.Perceptron(threshold, act.binary_step, weights, input) for input in inputs]

        for p in perceptrons:
            if p.inputs == (1, 1):
                self.assertEqual(1, p.determine_output(), f"AND gate: Result should be 1, since input is {p.inputs}")
            else:
                self.assertEqual(0, p.determine_output(), f"AND gate: Result should be 0, since input is {p.inputs}")

    def test_or_gate(self):
        """The or gate returns 1 if either x1 or x2 is 1, or if both are 1."""
        inputs = list(itertools.product([0, 1], repeat=2))
        weights = [2, 2]
        threshold = 1
        perceptrons = [pn.Perceptron(threshold, act.binary_step, weights, input) for input in inputs]
        for p in perceptrons:
            if p.inputs == (0, 0):
                self.assertEqual(0, p.determine_output(), f"OR gate: Result should be 0, since input is {p.inputs}")
            else:
                self.assertEqual(1, p.determine_output(), f"OR gate: Result should be 1, since input is {p.inputs}")

    def test_not_gate(self):
        """The NOT gate returns 1 if inputs are x1 and x2 are 0, else returns 0"""
        inputs = list(itertools.product([0, 1], repeat=2))
        weights = [-1, -1]
        threshold = -1
        perceptrons = [pn.Perceptron(threshold, act.binary_step, weights, input) for input in inputs]
        for p in perceptrons:
            if p.inputs == (0, 0):
                self.assertEqual(1, p.determine_output(), f"NOT gate: Result should be 1, since input is {p.inputs}")
            else:
                self.assertEqual(0, p.determine_output(), f"NOT gate: Result should be 0, since input is {p.inputs}")

    def test_nor_gate(self):
        """The NOR gate returns 1 only if x1 and x2 are 0, else returns 0"""
        inputs = list(itertools.product([0, 1], repeat=3))
        weights = [-1, -1, -1]
        threshold = -1  # bias == -threshold
        perceptrons = [pn.Perceptron(threshold, act.binary_step, weights, input) for input in inputs]
        for p in perceptrons:
            if p.inputs == (0, 0, 0):
                self.assertEqual(1, p.determine_output(), f"nor: Result should be 1, since input is {p.inputs}")
            else:
                self.assertEqual(0, p.determine_output(), f"nor: Result should be 0, since input is {p.inputs}")

    def test_party_gate(self):
        """The party gate is a gate I copied over from the reader."""
        inputs = list(itertools.product([0, 1], repeat=3)) # generates all possible combos of 0-0-1
        weights = [0.6, 0.3, 0.2]
        threshold = 0.4  # bias == -treshold

        # generate the perceptrons with every possible input combination
        perceptrons = [pn.Perceptron(threshold, act.binary_step, weights, input) for input in inputs]
        for p in perceptrons:
            if p.inputs in [(0, 0, 0), (0, 0, 1), (0, 1, 0)]:
                self.assertEqual(0, p.determine_output(), f"party Result should be 0, since input is {p.inputs}")
            else:
                self.assertEqual(1, p.determine_output(), f"party Result should be 1, since input is {p.inputs}")

    def test_xor_gate(self):
        """The XOR gate returns 1 when x1 or x2 EXCLUSIVELY is equal to 1, else returns 0"""
        inputs = list(itertools.product([0, 1], repeat=2))  # generate all combinations of inputs
        results = defaultdict(int) # to save the result in

        for input_values in inputs: # for every input, generate these layers
            layer1 = pn.PerceptronLayer([pn.Perceptron(0.5, act.binary_step, [1, 1], input_values),
                                         pn.Perceptron(-1.5, act.binary_step, [-1, -1], input_values)])
            layer2 = pn.PerceptronLayer([pn.Perceptron(1.5, act.binary_step, [1, 1])])
            network = pn.PerceptronNetwork([layer1, layer2])
            results[input_values] = network.feed_forward()  # store input as key and the output of the network as value

        for k, v in results.items(): # check all the results
            if k in [(0, 1), (1, 0)]:  # 0,1 and 1,0 should yield 1 according to XOR truth table
                self.assertEqual(1, v)
            else:
                self.assertEqual(0, v)


if __name__ == '__main__':
    unittest.main()
