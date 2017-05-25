How to run:
------------
python uCEQ_algo.py 		# will generate files (including .png) with uCE_Q prefix
python FoeQ_algo.py		# will generate files (including .png) with Foe_Q prefix
python FriendQ_algo.py		# will generate files (including .png) with Friend_Q prefix
python Q_learning_algo.py	# will generate files (including .png) with Q_learning prefix

Files description:
----------------------
Environment
___________
SoccerMDP.py

Equilibrium constraints and Objective functions
_______________________________________________
get_uCorrelatedEquilibrium.py (used for uCorrelatedEquilibriumQ)
get_AdversarialEquilibrium.py (used for Foe-Q)
get_CoordinatedEquilibrium.py (used for Friend-Q and Q-learning)

Multi Q Agent
_______________
MultiQ_agent.py

Algorithms
__________________
uCEQ_algo.py
FoeQ_algo.py
FriendQ_algo.py
Q_learning_algo.py

