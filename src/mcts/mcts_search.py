import copy
import numpy as np

class MCTSNode:
    """
    Represents a node in the MCTS tree.
    Each node is characterized by:
      - game: copy of the Game state at this node
      - prior: the initial policy probability for the action
      - parent: parent node in the search tree
      - children: dictionary (action -> child node)
      - visit_count: number of visits
      - total_value: sum of value estimates for this node
      - Q: average value Q(s,a)
    """
    def __init__(self, game, prior, parent=None):
        self.game = copy.deepcopy(game)
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.Q = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, action_priors):
        """
        For each valid action, create a new child node
        with the corresponding prior probability.
        """
        for action, prob in action_priors.items():
            # Only create a child if the probability > 0
            if prob > 0:
                # Copy game and make the move
                new_game_state = copy.deepcopy(self.game)
                new_game_state.make_move(action, new_game_state.current_player)
                child_node = MCTSNode(new_game_state, prob, parent=self)
                self.children[action] = child_node

    def update_stats(self, value):
        """
        Update this node's statistics given a simulated value from a child.
        """
        self.visit_count += 1
        self.total_value += value
        self.Q = self.total_value / self.visit_count

    def fully_qualified_path(self):
        """
        Useful for debugging - returns a string describing the path of actions from root to here.
        """
        path = []
        node = self
        while node.parent is not None:
            path.append(node)
            node = node.parent
        return path[::-1]


class MCTS:
    """
    MCTS class: For each call to 'run()', we perform a certain number
    of simulations starting from the given game state. We use the neural
    network to guide the search (policy, value).
    """
    def __init__(self, network, c_puct=1.0, num_simulations=50):
        """
        Args:
          network: Neural network with a predict(state) -> (policy, value)
          c_puct: Exploration constant
          num_simulations: Number of MCTS simulations for each move
        """
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def run(self, game):
        """
        Perform MCTS search for the current 'game' state.
        Return the visit count distribution over valid moves.
        """
        root = MCTSNode(game, prior=1.0, parent=None)

        # Expand root node
        if not game.game_over:
            policy, _ = self._evaluate(root)
            root.expand(policy)

        # Run N simulations
        for _ in range(self.num_simulations):
            node = self._select(root)
            value = self._simulate(node)
            self._backpropagate(node, value)

        # At the root node, collect visit counts for each valid action
        visit_counts = [0 for i in range(game.board_width)]
        actions = sorted(root.children.keys())
        for action in actions:
            visit_counts[action] = root.children[action].visit_count

        # Convert visit counts to probabilities
        visit_counts = np.array(visit_counts, dtype=float)
        if visit_counts.sum() > 0:
            probs = visit_counts / visit_counts.sum()
        else:
            # edge case: no expansions or all zero
            probs = np.ones(len(visit_counts)) / len(visit_counts)

        # Map probabilities back to the move indexes
        action_to_prob = []
        for i, action in enumerate(sorted(root.children.keys())):
            action_to_prob.append(probs[i])

        return action_to_prob, visit_counts

    def _select(self, node):
        """
        Traverse the tree from the node until a leaf is found.
        At each step, choose the action that maximizes the UCB.
        """
        current = node
        while current.is_expanded() and not current.game.game_over:
            best_score = -float('inf')
            best_action = None
            for action, child in current.children.items():
                # UCB formula
                U = child.Q + self.c_puct * child.prior * \
                    np.sqrt(current.visit_count + 1) / (1 + child.visit_count)
                if U > best_score:
                    best_score = U
                    best_action = action
            current = current.children[best_action]
        return current

    def _simulate(self, node):
        """
        Evaluate the leaf node. If it's terminal, return the outcome.
        Otherwise, use the network to predict (policy, value),
        expand children, and return the value estimate.
        """
        # If game is already over at this node, return outcome
        if node.game.game_over:
            if node.game.winner == 1:
                return 1.0
            elif node.game.winner == 2:
                return -1.0
            else:
                return 0.0

        # Evaluate the position with the network
        policy, value = self._evaluate(node)

        # Expand children with the predicted policy
        node.expand(policy)

        return value

    def _backpropagate(self, node, value):
        """
        Propagate the value back up the tree, alternating signs
        if needed to account for different players (this is optional,
        depends on how you handle perspective).
        """
        current = node
        while current is not None:
            current.update_stats(value)
            current = current.parent
            # For Connect4 in a zero-sum setting, we can invert value each step
            # if we want each parent's perspective. For simplicity, skip that:
            # value = -value

    def _evaluate(self, node):
        """
        Uses the neural network to get (policy, value) for the given state.
        The state must be represented in a suitable input format.
        For now we use a placeholder function 'get_state_representation'.
        """
        state_input = node.game.get_valid_nn_input()
        policy, value = self.network.predict(state_input)
        # 'policy' might be for all columns. We only keep the valid columns.
        valid_moves = node.game.available_moves()
        masked_policy = {}
        policy_sum = 0
        for action in valid_moves:
            masked_policy[action] = policy[action]
            policy_sum += policy[action]
        # Normalize
        if policy_sum > 0:
            for action in masked_policy.keys():
                masked_policy[action] /= policy_sum
        else:
            # if total policy sum is 0, just assign uniform
            for action in masked_policy.keys():
                masked_policy[action] = 1.0 / len(valid_moves)

        return masked_policy, value


class MockNetwork:
    """
    Mock network that returns random policy & random value.
    Replace with a real trained model.
    """
    def predict(self, state_input):
        # state_input shape: (1, board_height, board_width, 2)
        _, h, w, _ = state_input.shape
        # random policy for w columns
        random_policy = np.random.rand(w)
        random_value = np.random.uniform(-1, 1)
        return random_policy, random_value
