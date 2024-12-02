"""mcts.py: A class for the Monte Carlo Tree Search algorithm and player.

Matt Shumway
"""





class MCTSPlayer():
    def __init__(self, **kwargs):
        """
        Initialize the MCTS player.
        
        :param kwargs: The keyword arguments for the MCTS player.
        """
        self.name = kwargs.get('name', 'MCTS Player')
        self.num_simulations = kwargs.get('num_simulations', 100)
        self.C = kwargs.get('C', 1.4)
        self.mcts = MCTS(num_simulations=self.num_simulations, C=self.C)
    
    def get_action(self, game):
        """
        Get the action to take.
        
        :param game: The game to play.
        :return: The action to take.
        """
        return self.mcts.search(game)
    
    