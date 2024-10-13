from graphviz import Digraph

dot = Digraph()

dot.node('A', 'Age')
dot.node('B', 'Family History')
dot.node('C', 'Breast Cancer')
dot.node('D', 'Mammogram Result')
dot.node('E', 'Lump')

dot.edges(['AC', 'BC', 'CD', 'CE'])
dot.render('bayesian_network', format='png', cleanup=True)  # Saves as 'bayesian_network.png'
